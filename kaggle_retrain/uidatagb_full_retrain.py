"""
uidatagb_full_retrain.py
=========================
Full retraining pipeline for the UIdataGB 9-class gallbladder-ultrasound
classifier, designed to run top-to-bottom in a single Kaggle GPU notebook
(T4 x2). Rebuilt from scratch to fix three problems an external review
found in the earlier local pipeline:

  1. No held-out test set. The old pipeline used a 70/30 train/validation
     split and reported every number (including model-selection decisions
     like epoch count and thresholds) on that same validation set. This
     script uses a proper 70/20/10 train/validation/test split: the test
     set is touched exactly once, at the very end, after every design
     decision is already frozen.
  2. "Replicate" models were not reproducibility replicates. The old
     consistency-score replicates were trained on three DISJOINT SUBSETS
     of the training data (different data AND different seed), which
     conflates data-sensitivity with true seed-to-seed reproducibility.
     This script trains the primary model and 3 replicates on the SAME
     full training set, varying ONLY the random seed/initialisation.
  3. No epoch-level training curves were saved anywhere, only final-epoch
     numbers. This script logs train/validation loss and accuracy after
     every epoch and saves them so overfitting/underfitting can actually
     be diagnosed, not just asserted.

It also adds the two null-control experiments a review called mandatory
before trusting the XAI consistency numbers:
  - Untrained-model control: run the same cross-replicate consistency
    battery on 4 randomly-initialised (never-trained) ResNet-50 models,
    to see what the "floor" consistency score is with no learning at all.
  - Random-heatmap control: compute cosine similarity between pairs of
    independent random [0,1] heatmaps of the same shape, since Grad-CAM-
    family methods end in ReLU + min-max normalisation, which can inflate
    cosine similarity between unrelated non-negative maps.

WHY each major section exists, WHAT it takes as input, and WHAT it
produces are documented in that section's own header comment, per the
"explain what/why/input/output" convention requested for this project.

--------------------------------------------------------------------
HOW TO RUN THIS ON KAGGLE
--------------------------------------------------------------------
1. Upload the corrected dataset as a private Kaggle Dataset:
   - Zip fedgb/data/uidatagb/ locally (contains training/ and
     validation/ subfolders, 9 class folders each, already
     pHash-deduplicated -- do not re-run deduplication on this data).
   - On kaggle.com -> Datasets -> New Dataset -> upload the zip.
   - Name it e.g. "uidatagb-corrected".
2. Create a new Kaggle Notebook, attach that dataset as input, and
   turn on GPU T4 x2 (Settings -> Accelerator).
3. Copy this file's contents into notebook cells (the SECTION markers
   below are natural cell boundaries) or upload it as a notebook and
   run all cells.
4. Update DATA_ROOT below to match the attached dataset's mount path
   (Kaggle mounts datasets under /kaggle/input/<dataset-name>/).
5. Run all. Expect several hours of GPU time for the full pipeline
   (primary + 3 replicates + null controls); Kaggle's free tier gives
   30 GPU-hours/week, so budget accordingly and consider running the
   classification training and the XAI battery as two separate
   sessions if you are close to the weekly limit.

Every step below checks for its own output file first and skips if
already present, so this script is safe to re-run/resume after a
Kaggle session timeout.
"""

import os
import sys
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image

# =====================================================================
# SECTION 0: Configuration
# WHAT: all the knobs a rerun might need to change, in one place.
# WHY: keeping this at the top means nobody has to hunt through the
#      script to change the data path or seed list.
# =====================================================================

# Kaggle mounts an attached dataset named "uidatagb-corrected" here.
# CHANGE THIS to match your actual dataset slug if you name it
# differently when uploading.
DATA_ROOT = "/kaggle/input/datasets/sayed227/uidatagb/uidatagb-corrected/uidatagb"
OUT_ROOT = "/kaggle/working/uidatagb_retrain_outputs"

CLASS_NAMES = [
    "01_gallstones",
    "02_abdomen_and_retroperitoneum",
    "03_cholecystitis",
    "04_membranous_and_gangrenous_cholecystitis",
    "05_perforation",
    "06_polyps_and_cholesterol_crystals",
    "07_adenomyomatosis",
    "08_carcinoma",
    "09_various_causes_of_gallbladder_wall_thickening",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Split fractions. Applied to the UNION of the old training/ and
# validation/ folders, then re-split 70/20/10, so the test set is
# genuinely new rather than being the old validation set relabelled.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
TEST_FRAC = 0.10
SPLIT_SEED = 42  # fixed once; the split itself is NOT varied across
                  # the replicate-model seeds below, so all 4 models
                  # (primary + 3 replicates) see identical data.

# Seeds: primary model uses SPLIT_SEED-derived training seed 42 for
# continuity with the earlier paper; the 3 replicates vary ONLY the
# training seed (weight init + data loader shuffling order), on the
# SAME training data as the primary -- this is the Review-3 fix.
PRIMARY_SEED = 42
REPLICATE_SEEDS = [7, 123, 2027]

EPOCHS = 20                 # up from the old fixed 10; see Section 3
BATCH_SIZE = 160            # pushed up from 96 to use most of a T4's
                             # 16GB; ResNet-50 at 224x224 with AMP fp16
                             # activations fits comfortably at this size
                             # with headroom for the CUDA context + OS
                             # overhead. If this OOMs on your instance,
                             # drop to 128 then 96 then 64.
LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5      # epochs with no val-macroF1 improvement

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT_ROOT, exist_ok=True)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT_ROOT, "run.log"), "a") as f:
        f.write(line + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================================
# SECTION 1: Dataset
# WHAT: an ImageFolder-style dataset over the pHash-deduplicated
#       UIdataGB release, reading from whichever of the old
#       training/validation subfolders each file lives in (the
#       corrected data is already de-duplicated; we are only
#       RE-SPLITTING it 70/20/10, not touching image content).
# WHY:  a custom Dataset (not torchvision.ImageFolder) is used so the
#       class order matches CLASS_NAMES exactly, keeping label indices
#       identical to the original paper's tables.
# INPUT: DATA_ROOT/training/<class>/*.{png,jpg,jpeg,bmp} and
#        DATA_ROOT/validation/<class>/*.{png,jpg,jpeg,bmp}
# OUTPUT: a list of (path, label) pairs, later split 70/20/10.
# =====================================================================

def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])


class UIdataGBFiles(Dataset):
    """Thin dataset over a fixed list of (path, label) pairs, so the
    same underlying file list can be wrapped with either train or
    eval transforms for different DataLoaders without re-scanning
    disk each time."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def collect_all_samples():
    samples = []
    for split_dir in ("training", "validation"):
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(DATA_ROOT, split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = CLASS_TO_IDX[class_name]
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    samples.append((os.path.join(class_dir, fname), label))
    return samples


# Path to the cluster manifest generated locally by
# generate_dihedral_cluster_manifest.py (fedgb/outputs/phase0/
# dihedral_cluster_manifest.json), uploaded alongside the image data as
# a second Kaggle Dataset (or included in the same dataset zip). This
# corrects a flip/rotation-augmented-duplicate leakage gap found by
# augmentation_aware_leakage_check.py: 46% of the corpus (4,918 of
# 10,692 images) had at least one cross-split duplicate the original
# single-orientation pHash pipeline could not detect. Splitting must
# happen at the CLUSTER level -- every image in a duplicate cluster
# assigned to the same one of train/val/test -- or this correction is
# silently lost the moment a fresh per-image split is drawn.
CLUSTER_MANIFEST_PATH = "/kaggle/input/datasets/sayed227/uidatagb/uidatagb-cluster-manifest.json"


def normalize_rel_key(path):
    """Reduces any path (Windows-mixed-separator, Linux, absolute,
    relative) down to 'training/<class>/<filename>' or
    'validation/<class>/<filename>', forward-slashed and lowercase for
    the split segment, so the manifest (built locally, Windows paths)
    and the Kaggle-side file listing (Linux paths under a different
    root) can be matched reliably without depending on exact prefixes
    matching between two different machines' filesystem layouts."""
    norm = path.replace("\\", "/")
    parts = norm.split("/")
    for i, p in enumerate(parts):
        if p.lower() in ("training", "validation"):
            return "/".join(parts[i:])
    # fallback: last 3 components (split/class/filename) if the split
    # folder name itself wasn't found verbatim
    return "/".join(parts[-3:])


def load_cluster_manifest():
    if not os.path.exists(CLUSTER_MANIFEST_PATH):
        log(f"[split] WARNING: cluster manifest not found at "
            f"{CLUSTER_MANIFEST_PATH} -- falling back to a per-image split "
            f"with NO duplicate-cluster protection. This re-opens the "
            f"flip/rotation-duplicate leakage gap the manifest exists to "
            f"close. Only proceed without it if you have independently "
            f"confirmed this dataset copy has no such duplicates.")
        return None
    with open(CLUSTER_MANIFEST_PATH) as f:
        manifest = json.load(f)
    # rebuild the lookup keyed on the normalized relative path, since the
    # manifest's stored paths came from a different machine/filesystem
    rel_to_cluster = {}
    for raw_path, cluster_id in manifest["path_to_cluster_id"].items():
        rel_to_cluster[normalize_rel_key(raw_path)] = cluster_id
    log(f"[split] loaded cluster manifest: {manifest['n_clusters']} clusters "
        f"from {manifest['n_raw_images']} raw images "
        f"(method: {manifest['method']})")
    return rel_to_cluster


def make_70_20_10_split():
    """Stratified 70/20/10 split, computed ONCE at the DUPLICATE-CLUSTER
    level (not the raw-image level) and cached to disk so every model
    (primary + all 3 replicates + both null controls) trains/evaluates/
    tests on the IDENTICAL split -- required both for the replicate-
    consistency fix (isolate seed effects only) AND for the flip/
    rotation-duplicate leakage fix (no cluster ever crosses a split
    boundary)."""
    split_path = os.path.join(OUT_ROOT, "split_70_20_10.json")
    if os.path.exists(split_path):
        log(f"[split] loading cached split from {split_path}")
        with open(split_path) as f:
            cached = json.load(f)
        return cached["train"], cached["val"], cached["test"]

    log("[split] building fresh stratified 70/20/10 split (cluster-aware)")
    all_samples = collect_all_samples()
    rel_to_cluster = load_cluster_manifest()

    # group images into clusters: images sharing a cluster ID from the
    # manifest go together; any image the manifest doesn't cover (should
    # not happen if the manifest was built from this exact dataset, but
    # handled defensively) becomes its own singleton cluster so it still
    # participates in the split rather than crashing the script.
    cluster_members = {}   # cluster_key -> [(path, label), ...]
    next_singleton_id = -1
    for path, label in all_samples:
        rel_key = normalize_rel_key(path)
        if rel_to_cluster is not None and rel_key in rel_to_cluster:
            cluster_key = ("manifest", rel_to_cluster[rel_key])
        else:
            cluster_key = ("singleton", next_singleton_id)
            next_singleton_id -= 1
        cluster_members.setdefault(cluster_key, []).append((path, label))

    n_unmatched = sum(1 for k in cluster_members if k[0] == "singleton") \
        if rel_to_cluster is not None else len(all_samples)
    if rel_to_cluster is not None and n_unmatched > 0:
        log(f"[split] WARNING: {n_unmatched} images had no manifest entry "
            f"and were treated as their own singleton cluster -- check "
            f"normalize_rel_key() output against the manifest's stored "
            f"paths if this number is large (expected: 0, or very small).")

    # majority class per cluster (a cluster could technically span two
    # class folders if the underlying duplicate was mislabeled -- see
    # dihedral_mixed_class_clusters.json for the ones already flagged)
    cluster_majority_label = {}
    for ckey, members in cluster_members.items():
        labels = [lbl for _, lbl in members]
        cluster_majority_label[ckey] = max(set(labels), key=labels.count)

    rng = random.Random(SPLIT_SEED)
    by_class_clusters = {}
    for ckey, maj_label in cluster_majority_label.items():
        by_class_clusters.setdefault(maj_label, []).append(ckey)

    train, val, test = [], [], []
    for label, ckeys in by_class_clusters.items():
        ckeys = ckeys[:]
        rng.shuffle(ckeys)
        n = len(ckeys)
        n_train = round(n * TRAIN_FRAC)
        n_val = round(n * VAL_FRAC)
        train_ckeys = set(ckeys[:n_train])
        val_ckeys = set(ckeys[n_train:n_train + n_val])
        test_ckeys = set(ckeys[n_train + n_val:])
        for ckey in ckeys:
            target = (train if ckey in train_ckeys
                       else val if ckey in val_ckeys
                       else test)
            target.extend(cluster_members[ckey])

    log(f"[split] cluster-level split: {len(cluster_members)} clusters -> "
        f"train={len(train)} val={len(val)} test={len(test)} images")

    with open(split_path, "w") as f:
        json.dump({"train": train, "val": val, "test": test}, f)

    log(f"[split] train={len(train)} val={len(val)} test={len(test)} "
        f"(total={len(train) + len(val) + len(test)})")
    return train, val, test


# =====================================================================
# SECTION 2: Model
# WHAT: ResNet-50, ImageNet-pretrained, 9-class head, RAW LOGITS
#       output (no Softmax layer in the model itself -- softmax is
#       applied only at inference time; nn.CrossEntropyLoss expects
#       raw logits and applies log-softmax internally during training).
# WHY:  matches the architecture already documented in the paper's
#       Methods section, kept identical here so results are comparable
#       to the pipeline's earlier runs.
# INPUT: none (loads ImageNet weights from torchvision).
# OUTPUT: an nn.Module mapping 224x224x3 -> 9 raw class logits.
# =====================================================================

def build_model():
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    return model


def get_target_layer(model):
    """layer4 is the standard Grad-CAM-family tap point: the last
    convolutional stage before global average pooling, encoding the
    richest semantic features."""
    return model.layer4[-1]


# =====================================================================
# SECTION 3: Training loop with full per-epoch history
# WHAT: standard supervised training with early stopping on
#       validation macro-F1, saving BOTH the best-val checkpoint and
#       a full per-epoch history of train/val loss and accuracy.
# WHY:  this is the fix for "no epoch curves were saved" -- every
#       epoch's numbers are appended to a JSON list so overfitting
#       (train acc high, val acc low/falling) or underfitting (both
#       low) can be read directly off a plot afterward, not asserted
#       from a single final-epoch number.
# INPUT: train_loader, val_loader, a fresh model, a seed.
# OUTPUT: outputs/<tag>/best.pt (checkpoint with best val macro-F1),
#         outputs/<tag>/history.json (per-epoch loss/accuracy/F1).
# =====================================================================

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    accuracy = float((all_preds == all_labels).mean())

    # macro-F1 by hand (avoids requiring sklearn to be preinstalled,
    # though Kaggle images normally have it)
    f1s = []
    for c in range(NUM_CLASSES):
        tp = int(((all_preds == c) & (all_labels == c)).sum())
        fp = int(((all_preds == c) & (all_labels != c)).sum())
        fn = int(((all_preds != c) & (all_labels == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))

    return total_loss / n, accuracy, macro_f1


def train_one_model(tag, seed, train_samples, val_samples, epochs=EPOCHS):
    """Trains one model (primary or a replicate) on a FIXED sample
    list, varying only `seed` (weight init + shuffle order) -- this is
    the direct fix for Review 3's disjoint-subset problem: every
    replicate sees the SAME train_samples, just a different seed."""
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "best.pt")
    history_path = os.path.join(out_dir, "history.json")

    if os.path.exists(ckpt_path) and os.path.exists(history_path):
        log(f"[{tag}] already trained, skipping (found {ckpt_path})")
        return ckpt_path, history_path

    set_seed(seed)

    train_ds = UIdataGBFiles(train_samples, get_transforms(train=True))
    val_ds = UIdataGBFiles(val_samples, get_transforms(train=False))
    # persistent_workers + prefetch_factor keep the CPU-side worker pool
    # alive and pre-loading batches between epochs instead of restarting
    # it every epoch, which matters more once BATCH_SIZE is large enough
    # that the GPU can otherwise sit idle waiting for the next batch.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True,
                               persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True,
                             persistent_workers=True, prefetch_factor=4)

    model = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    # Mixed precision: uses the T4's fp16 tensor cores, cutting memory
    # per sample roughly in half so the larger BATCH_SIZE above fits
    # comfortably, and speeding up each step. Falls back to a no-op on
    # CPU automatically via the `enabled=` flag.
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    history = []
    best_val_f1 = -1.0
    epochs_since_improve = 0

    log(f"[{tag}] training start (seed={seed}, n_train={len(train_samples)}, "
        f"n_val={len(val_samples)})")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        correct = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
            correct += int((logits.argmax(dim=1) == y).sum())

        train_loss = running_loss / n_seen
        train_acc = correct / n_seen
        val_loss, val_acc, val_f1 = evaluate(model, val_loader)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        })
        log(f"[{tag}] epoch {epoch:02d}/{epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        # checkpoint on best validation macro-F1, not just accuracy,
        # since accuracy alone can hide poor performance on rare classes
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_since_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            log(f"[{tag}]   -> new best val_macro_f1={val_f1:.4f}, checkpoint saved")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= EARLY_STOP_PATIENCE:
                log(f"[{tag}] early stopping at epoch {epoch} "
                    f"(no val improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

        # save history after every epoch, not just at the end, so a
        # Kaggle session timeout does not lose already-completed epochs
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    log(f"[{tag}] training done. best_val_macro_f1={best_val_f1:.4f}")
    return ckpt_path, history_path


# =====================================================================
# SECTION 4: Final test-set evaluation
# WHAT: loads each model's BEST checkpoint (selected on validation,
#       Section 3) and evaluates it EXACTLY ONCE on the held-out test
#       set, which no design decision (epoch count, seed choice,
#       threshold values) was ever allowed to see.
# WHY:  this is the actual point of a three-way split -- an unbiased
#       final number. Running this more than once per model and
#       picking the best result would silently reintroduce the same
#       bias a validation-only split has; call it once and report
#       what it says.
# INPUT: a trained checkpoint, the frozen test_samples list.
# OUTPUT: outputs/<tag>/test_metrics.json
# =====================================================================

def evaluate_on_test(tag, ckpt_path, test_samples):
    out_path = os.path.join(OUT_ROOT, tag, "test_metrics.json")
    if os.path.exists(out_path):
        log(f"[{tag}] test metrics already computed, skipping")
        with open(out_path) as f:
            return json.load(f)

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

    test_ds = UIdataGBFiles(test_samples, get_transforms(train=False))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    test_loss, test_acc, test_f1 = evaluate(model, test_loader)
    result = {
        "tag": tag,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "n_test": len(test_samples),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"[{tag}] TEST SET (touched once): "
        f"acc={test_acc:.4f} macro_f1={test_f1:.4f}")
    return result


# =====================================================================
# SECTION 5: Null control 1 -- untrained-model consistency floor
# WHAT: builds 4 ResNet-50 models with ONLY ImageNet-pretrained
#       weights (classification head randomly initialised, NEVER
#       fine-tuned on gallbladder data), then runs the same Grad-CAM
#       cross-model cosine-consistency computation the trained
#       replicates use.
# WHY:  a reviewer flagged this as the decisive, mandatory control:
#       if untrained models ALSO show high (~0.9+) cross-model
#       consistency, the consistency metric is not measuring anything
#       about learned, disease-specific behaviour -- it would be an
#       artefact of the CAM math (ReLU + [0,1] normalisation) or the
#       architecture's inductive bias alone.
# INPUT: the same frozen test/val image sample list used elsewhere,
#        a small stratified subset for speed.
# OUTPUT: outputs/null_control_untrained/consistency.json
# =====================================================================

def gradcam_heatmap(model, target_layer, x, class_idx):
    """Minimal single-purpose Grad-CAM (not Grad-CAM++/Score-CAM here
    -- the null control only needs one representative CAM-family
    method to test whether the consistency METRIC itself has a
    trivial floor; if Grad-CAM's floor is already high, the same
    ReLU + normalisation argument applies to the other CAM variants)."""
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    logits = model(x)
    score = logits[0, class_idx]
    score.backward()

    h1.remove()
    h2.remove()

    act = activations["value"][0]        # [C, H, W]
    grad = gradients["value"][0]          # [C, H, W]
    weights = grad.mean(dim=(1, 2))       # [C]
    cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
    cam = cam.detach().cpu().numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam.flatten()


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def run_untrained_null_control(test_samples, n_images=40):
    out_path = os.path.join(OUT_ROOT, "null_control_untrained", "consistency.json")
    if os.path.exists(out_path):
        log("[null-untrained] already computed, skipping")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    log(f"[null-untrained] building 4 untrained models "
        f"(ImageNet backbone, random 9-class head, no fine-tuning)")
    set_seed(999)
    models_list = [build_model().to(DEVICE).eval() for _ in range(4)]
    for m in models_list:
        for p in m.parameters():
            p.requires_grad_(True)  # need grads for Grad-CAM even though untrained

    rng = random.Random(1234)
    subset = rng.sample(test_samples, min(n_images, len(test_samples)))
    transform = get_transforms(train=False)

    sims = []
    for path, label in subset:
        image = Image.open(path).convert("RGB")
        x = transform(image).unsqueeze(0).to(DEVICE)

        heatmaps = []
        for m in models_list:
            target_layer = get_target_layer(m)
            heatmaps.append(gradcam_heatmap(m, target_layer, x, label))

        # pairwise cosine similarity across the 4 untrained models
        for i in range(len(heatmaps)):
            for j in range(i + 1, len(heatmaps)):
                sims.append(cosine_sim(heatmaps[i], heatmaps[j]))

    result = {
        "n_images": len(subset),
        "n_pairs": len(sims),
        "mean_cosine_similarity": float(np.mean(sims)),
        "std_cosine_similarity": float(np.std(sims)),
        "note": ("Cross-model Grad-CAM cosine similarity between 4 UNTRAINED "
                 "(ImageNet backbone only, random classification head) "
                 "ResNet-50 models. This is the floor the trained "
                 "replicate-consistency scores must clear to mean anything."),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"[null-untrained] mean cosine similarity = "
        f"{result['mean_cosine_similarity']:.4f} +/- {result['std_cosine_similarity']:.4f} "
        f"over {result['n_pairs']} pairs")


# =====================================================================
# SECTION 6: Null control 2 -- random-heatmap cosine-similarity floor
# WHAT: generates pairs of independent random [0,1]-normalised
#       224x224 maps (matching the shape/range Grad-CAM-family outputs
#       use after ReLU + min-max normalisation) and computes their
#       cosine similarity.
# WHY:  isolates whether high cosine similarity is a property of the
#       MATH (two non-negative vectors of the same dimension are
#       biased toward high cosine similarity) rather than a property
#       of the models or the images at all. This does not need a GPU
#       or a trained model -- it is a pure statistics check.
# INPUT: none (synthetic).
# OUTPUT: outputs/null_control_random_heatmaps/cosine_floor.json
# =====================================================================

def run_random_heatmap_null_control(n_pairs=2000, shape=(7, 7)):
    """shape=(7,7) matches ResNet-50's layer4 spatial resolution for
    224x224 input (the CAM's native resolution before upsampling) --
    using the native resolution, not the upsampled 224x224 map, keeps
    this test fast; the cosine-similarity-inflation argument applies
    at either resolution since it is about non-negativity, not size."""
    out_path = os.path.join(OUT_ROOT, "null_control_random_heatmaps",
                             "cosine_floor.json")
    if os.path.exists(out_path):
        log("[null-random] already computed, skipping")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rng = np.random.RandomState(42)
    sims_random_uniform = []
    sims_random_relu = []
    for _ in range(n_pairs):
        a = rng.uniform(0, 1, size=shape).flatten()
        b = rng.uniform(0, 1, size=shape).flatten()
        sims_random_uniform.append(cosine_sim(a, b))

        # a second, more realistic control: ReLU applied to standard-
        # normal noise then min-max normalised, matching what
        # Grad-CAM's own ReLU(sum(alpha*A)) step actually produces
        # before normalisation, rather than pure uniform noise
        a2 = np.maximum(rng.standard_normal(shape), 0).flatten()
        b2 = np.maximum(rng.standard_normal(shape), 0).flatten()
        if a2.max() > a2.min():
            a2 = (a2 - a2.min()) / (a2.max() - a2.min())
        if b2.max() > b2.min():
            b2 = (b2 - b2.min()) / (b2.max() - b2.min())
        sims_random_relu.append(cosine_sim(a2, b2))

    result = {
        "n_pairs": n_pairs,
        "shape": list(shape),
        "uniform_random_pairs": {
            "mean_cosine_similarity": float(np.mean(sims_random_uniform)),
            "std_cosine_similarity": float(np.std(sims_random_uniform)),
        },
        "relu_gaussian_pairs": {
            "mean_cosine_similarity": float(np.mean(sims_random_relu)),
            "std_cosine_similarity": float(np.std(sims_random_relu)),
        },
        "note": ("Cosine similarity between INDEPENDENT RANDOM heatmap pairs "
                 "with no model, no image, and no learning involved. If this "
                 "floor is already close to the paper's reported ~0.94 "
                 "cross-replicate consistency, the metric has little content."),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"[null-random] uniform-random floor = "
        f"{result['uniform_random_pairs']['mean_cosine_similarity']:.4f}; "
        f"ReLU-Gaussian floor = "
        f"{result['relu_gaussian_pairs']['mean_cosine_similarity']:.4f}")


# =====================================================================
# SECTION 7: Same-data cross-replicate consistency (the Review-3 fix)
# WHAT: computes Grad-CAM cosine consistency between the primary model
#       and each of the 3 SAME-DATA-different-seed replicates trained
#       in Section 3, on a stratified sample of test images.
# WHY:  this is the actual reproducibility number the paper needs --
#       comparable in spirit to the old Table V/VI, but now measuring
#       ONLY seed/initialisation sensitivity, since all 4 models saw
#       identical training data.
# INPUT: the 4 trained checkpoints (primary + 3 replicates), test set.
# OUTPUT: outputs/consistency_same_data/consistency.json
# =====================================================================

def run_same_data_consistency(primary_ckpt, replicate_ckpts, test_samples,
                               n_per_class=8):
    out_path = os.path.join(OUT_ROOT, "consistency_same_data", "consistency.json")
    if os.path.exists(out_path):
        log("[consistency-same-data] already computed, skipping")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    primary = build_model().to(DEVICE)
    primary.load_state_dict(torch.load(primary_ckpt, map_location=DEVICE, weights_only=True))
    primary.eval()

    replicates = []
    for ckpt in replicate_ckpts:
        m = build_model().to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        m.eval()
        replicates.append(m)

    by_class = {}
    for path, label in test_samples:
        by_class.setdefault(label, []).append(path)
    rng = random.Random(42)
    stratified = []
    for label, paths in by_class.items():
        chosen = rng.sample(paths, min(n_per_class, len(paths)))
        stratified += [(p, label) for p in chosen]

    transform = get_transforms(train=False)
    per_replicate_scores = [[] for _ in replicates]

    for path, label in stratified:
        image = Image.open(path).convert("RGB")
        x = transform(image).unsqueeze(0).to(DEVICE)

        primary_map = gradcam_heatmap(primary, get_target_layer(primary), x, label)
        for i, rep in enumerate(replicates):
            rep_map = gradcam_heatmap(rep, get_target_layer(rep), x, label)
            per_replicate_scores[i].append(cosine_sim(primary_map, rep_map))

    per_replicate_mean = [float(np.mean(s)) for s in per_replicate_scores]
    all_scores_flat = [s for rep_scores in per_replicate_scores for s in rep_scores]

    result = {
        "n_images": len(stratified),
        "per_replicate_mean": per_replicate_mean,
        "per_replicate_mean_std_of_means": float(np.std(per_replicate_mean)),
        "pooled_all_scores_mean": float(np.mean(all_scores_flat)),
        "pooled_all_scores_std": float(np.std(all_scores_flat)),
        "note": ("per_replicate_mean_std_of_means is std across the 3 "
                 "replicate-level MEANS (what the paper's table should "
                 "report as the 'std' column); pooled_all_scores_std is "
                 "std across every individual image-level score pooled "
                 "together (what the old script actually computed and "
                 "mislabelled -- kept here for direct comparison)."),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"[consistency-same-data] per-replicate means: {per_replicate_mean}, "
        f"std-of-means={result['per_replicate_mean_std_of_means']:.4f}")


# =====================================================================
# SECTION 8: Orchestration
# WHAT: runs every section above in the order your workflow specifies:
#       inspect -> preprocess (already done, corrected data attached)
#       -> split -> load model -> train -> monitor -> evaluate on test
#       -> confusion/metrics -> null controls -> consistency.
# =====================================================================

def main():
    log("=== uidatagb_full_retrain.py started ===")
    log(f"Device: {DEVICE}")
    if not os.path.isdir(DATA_ROOT):
        log(f"FATAL: DATA_ROOT {DATA_ROOT} does not exist. "
            f"Check the attached Kaggle dataset's mount path under "
            f"/kaggle/input/ and update DATA_ROOT at the top of this file.")
        sys.exit(1)

    train_samples, val_samples, test_samples = make_70_20_10_split()

    # --- Section 3+4: primary model ---
    primary_ckpt, _ = train_one_model("primary_seed42", PRIMARY_SEED,
                                       train_samples, val_samples)
    evaluate_on_test("primary_seed42", primary_ckpt, test_samples)

    # --- Section 3+4: same-data, different-seed replicates ---
    replicate_ckpts = []
    for seed in REPLICATE_SEEDS:
        tag = f"replicate_seed{seed}"
        ckpt, _ = train_one_model(tag, seed, train_samples, val_samples)
        evaluate_on_test(tag, ckpt, test_samples)
        replicate_ckpts.append(ckpt)

    # --- Section 5+6: null controls (cheap, run regardless of above) ---
    run_untrained_null_control(test_samples)
    run_random_heatmap_null_control()

    # --- Section 7: same-data consistency (the actual fixed metric) ---
    run_same_data_consistency(primary_ckpt, replicate_ckpts, test_samples)

    log("=== ALL DONE ===")
    log(f"Outputs written under: {OUT_ROOT}")

    # --- Section 8: auto-zip immediately, so a single downloadable file
    #     exists the instant training finishes, regardless of whether
    #     this draft session survives long enough for you to click
    #     "Save Version". A script running INSIDE the kernel cannot
    #     force Kaggle itself to persist /kaggle/working across a
    #     session restart -- only Kaggle's own "Save Version" button
    #     does that -- and IPython's FileLink does NOT work for
    #     downloading on Kaggle (it 404s; Kaggle does not serve
    #     /kaggle/working over the classic Jupyter file-serving route
    #     FileLink assumes). The reliable download path on Kaggle is the
    #     notebook's own "Output" panel (right-hand sidebar while the
    #     session is alive) or the committed version's "Output" tab
    #     (after Save & Run All) -- both list /kaggle/working/*
    #     directly with a working download button per file/folder.
    try:
        import shutil
        zip_path = shutil.make_archive(OUT_ROOT, "zip", OUT_ROOT)
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        log(f"[auto-zip] wrote {zip_path} ({size_mb:.1f} MB)")
    except Exception as e:
        log(f"[auto-zip] WARNING: zipping failed ({e}); "
            f"outputs are still present, unzipped, under {OUT_ROOT}")

    log("TO DOWNLOAD: do NOT rely on any link printed above this line -- "
        "it will 404 on Kaggle. Instead open the 'Output' panel in the "
        "right-hand sidebar of this notebook, find "
        "'uidatagb_retrain_outputs.zip' (or the uidatagb_retrain_outputs/ "
        "folder), and click its own download icon there.")
    log("IMPORTANT: click 'Save Version -> Save & Run All' now (or as "
        "soon as possible). Kaggle does not guarantee /kaggle/working "
        "survives a draft-session restart, and the zip above only helps "
        "if you download it before that happens.")


if __name__ == "__main__":
    main()
