"""
uidatagb_stage2_remaining_reviews.py
======================================
Kaggle GPU script covering the remaining code-dependent points from the
21-point review, following the exact same stable pattern as
uidatagb_full_retrain.py: resumable (each section checks for its own
output file and skips if already done), single-GPU (T4), AMP + large
batch size, and an auto-zip at the end so results are never lost to a
session restart.

Covers three reviews, each an independent, separately-checkpointed
section so a partial run still preserves whatever finished:

  Review 11 -- the batch-ID shortcut diagnostic (10-way "which acquisition
    batch is this Carcinoma image from" classifier) was itself not
    leakage-audited: it used a random image-level split WITHIN each
    batch, so near-duplicate frames from the same acquisition could sit
    on both sides, making its 100% accuracy potentially an artifact of
    memorising near-identical frames rather than proof of a learnable
    non-anatomical signature. This script reruns that diagnostic with
    the dihedral-aware cluster manifest applied, so no duplicate cluster
    spans both the train and validation side of the batch-ID task.

  Review 16 -- the paper's only shortcut audit (Section IV.4) tests
    margin/border artefacts, which the paper's own Stage 2 analysis
    already says cannot explain the batch-correlated shortcut Review 11
    is about. This script adds the two shortcut tests the review asks
    for directly: (a) held-out-BATCH accuracy gap -- train the disease
    classifier's shortcut-relevant signal on a subset of Carcinoma
    batches and measure the accuracy drop when evaluated on held-out
    UNSEEN batches vs. held-out SEEN-but-different-images batches; and
    (b) a linear probe trained on the disease classifier's own layer4
    features, testing whether batch identity is linearly recoverable
    from the same features the disease classifier uses.

  Review 17 -- the paper reports one noise level (sigma=0.03, 29.2%
    prediction-flip rate) as a classifier-level finding and does not
    follow it up. This script sweeps sigma across a range and reports
    accuracy directly at each level, on both the original Stage 1 split
    and (if the Stage 2 checkpoint is available) the corrected Stage 2
    split, so the noise-robustness result can be compared across both.

HOW TO RUN THIS ON KAGGLE
--------------------------------------------------------------------
1. This needs THREE things attached as Kaggle datasets (reuse what
   Stage 2 already used, plus one new small file):
   - the same "uidatagb-corrected" image dataset used for
     uidatagb_full_retrain.py
   - the same "uidatagb-cluster-manifest" (dihedral_cluster_manifest.json)
   - NEW: the Stage 2 checkpoints already trained (primary_seed42/best.pt
     at minimum) -- zip and upload uidatagb_retrain_outputs/ from your
     last Kaggle run's downloaded output, or just the primary_seed42/
     subfolder to keep the upload small, as a new dataset named
     "uidatagb-stage2-checkpoints".
2. Update DATA_ROOT, CLUSTER_MANIFEST_PATH, and STAGE2_CKPT_ROOT below
   to match the actual mount paths (check with a quick os.walk cell
   first, the same way the paths were confirmed for the main retrain --
   do not guess).
3. Turn on GPU T4 x2 (only one GPU is used, matching the main retrain
   script's design decision).
4. Run all. This is much cheaper than the main retrain: no 20-epoch
   full-dataset training happens here, only small batch-ID/shortcut
   classifiers (6-10 epochs on a few hundred images) and inference-only
   noise sweeps. Expect well under an hour total, likely 15-25 minutes.
5. At the end, the script auto-zips its own output folder. As before,
   the printed FileLink will 404 on Kaggle -- use the Output side panel
   to download 'uidatagb_stage2_reviews_outputs.zip' directly, the same
   way the main retrain's results were downloaded.
"""
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

# =====================================================================
# SECTION 0: Configuration -- CHECK THESE PATHS BEFORE RUNNING
# =====================================================================

DATA_ROOT = "/kaggle/input/datasets/sayed227/uidatagb/uidatagb-corrected/uidatagb"
CLUSTER_MANIFEST_PATH = "/kaggle/input/datasets/sayed227/uidatagb/uidatagb-cluster-manifest.json"
# NEW for this script: point this at wherever the Stage 2 primary
# checkpoint from the main retrain ends up mounted once uploaded as its
# own dataset. Confirmed via a full os.walk of /kaggle/input: this
# account's datasets mount under /kaggle/input/datasets/<username>/...,
# not directly under /kaggle/input/<dataset-slug>/ -- and the uploaded
# checkpoint file itself is flat (uploaded directly as a single .pt
# file, not nested in a primary_seed42/ subfolder).
STAGE2_CKPT_ROOT = "/kaggle/input/datasets/sayed227/uidatagb-stage2-checkpoints"
STAGE2_PRIMARY_CKPT = os.path.join(STAGE2_CKPT_ROOT, "uidatagb-stage2-checkpoints.pt")

OUT_ROOT = "/kaggle/working/uidatagb_stage2_reviews_outputs"
os.makedirs(OUT_ROOT, exist_ok=True)

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
NUM_CLASSES = len(CLASS_NAMES)
TARGET_CLASS = "08_carcinoma"  # largest class, most batches -- same choice as the original diagnostic
FNAME_RE = re.compile(r'^([A-Za-z]+\d+)\s*\(\d+\)\.\w+$')

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 96

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


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


def get_transforms(train: bool):
    tf = [transforms.Resize((224, 224))]
    if train:
        tf += [transforms.RandomHorizontalFlip(),
               transforms.RandomRotation(15),
               transforms.ColorJitter(brightness=0.2, contrast=0.2)]
    tf += [transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)]
    return transforms.Compose(tf)


def build_model(num_classes=NUM_CLASSES, pretrained=True):
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4), nn.Linear(in_features, 256), nn.ReLU(),
        nn.Dropout(p=0.3), nn.Linear(256, num_classes),
    )
    return model


def get_target_layer(model):
    return model.layer4[-1]


class PathLabelDataset(Dataset):
    def __init__(self, items, train):
        self.items = items  # list of (path, label_idx)
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def load_cluster_manifest():
    with open(CLUSTER_MANIFEST_PATH) as f:
        manifest = json.load(f)

    def normalize_rel_key(path):
        norm = path.replace("\\", "/")
        parts = norm.split("/")
        for i, p in enumerate(parts):
            if p.lower() in ("training", "validation"):
                return "/".join(parts[i:])
        return "/".join(parts[-3:])

    rel_to_cluster = {}
    for raw_path, cluster_id in manifest["path_to_cluster_id"].items():
        rel_to_cluster[normalize_rel_key(raw_path)] = cluster_id
    log(f"[manifest] loaded {manifest['n_clusters']} clusters from "
        f"{manifest['n_raw_images']} raw images")
    return rel_to_cluster


def rel_key_for_path(path):
    norm = path.replace("\\", "/")
    parts = norm.split("/")
    for i, p in enumerate(parts):
        if p.lower() in ("training", "validation"):
            return "/".join(parts[i:])
    return "/".join(parts[-3:])


# =====================================================================
# SECTION 11: Review 11 -- batch-ID diagnostic, cluster-aware split
# WHAT: reruns the original batch_id_shortcut_diagnostic.py's 10-way
#       "which batch" classifier, but splits train/val at the dihedral
#       cluster level instead of randomly per image, so no near-duplicate
#       frame pair can straddle both sides.
# WHY: Review 11's exact objection -- an image-level split within a
#      batch means near-duplicate frames from the same acquisition can
#      sit on both sides, so a naive 100% accuracy is close to the
#      expected outcome of memorising near-identical frames rather than
#      genuine evidence of a learnable batch signature. This is the
#      same failure mode Stage 1's original split had, applied to a
#      different classifier.
# OUTPUT: outputs/review11_batch_id_cluster_aware/result.json
# =====================================================================

def review_11_batch_id_cluster_aware(rel_to_cluster, n_batches=10,
                                       min_batch_size=25, val_frac=0.30,
                                       epochs=6):
    out_dir = os.path.join(OUT_ROOT, "review11_batch_id_cluster_aware")
    out_path = os.path.join(out_dir, "result.json")
    if os.path.exists(out_path):
        log("[review11] already computed, skipping")
        return
    os.makedirs(out_dir, exist_ok=True)

    items_by_prefix = defaultdict(list)
    for split in ("training", "validation"):
        cdir = os.path.join(DATA_ROOT, split, TARGET_CLASS)
        if not os.path.isdir(cdir):
            continue
        for fname in os.listdir(cdir):
            m = FNAME_RE.match(fname)
            if m:
                items_by_prefix[m.group(1).lower()].append(os.path.join(cdir, fname))

    eligible = {k: v for k, v in items_by_prefix.items() if len(v) >= min_batch_size}
    top_batches = sorted(eligible.items(), key=lambda kv: -len(kv[1]))[:n_batches]
    batch_names = [k for k, _ in top_batches]
    log(f"[review11] {len(items_by_prefix)} total batches in {TARGET_CLASS}, "
        f"using top {len(batch_names)}: {[(k, len(v)) for k, v in top_batches]}")

    # Cluster-aware split: within each batch, group its images by
    # dihedral cluster id, then assign WHOLE CLUSTERS to train or val,
    # never splitting a cluster across the two sides.
    rng = random.Random(SEED)
    train_items, val_items = [], []
    n_singleton_fallback = 0
    for label_idx, (bname, paths) in enumerate(top_batches):
        clusters_in_batch = defaultdict(list)
        for p in paths:
            rk = rel_key_for_path(p)
            cid = rel_to_cluster.get(rk)
            if cid is None:
                cid = f"singleton_{p}"
                n_singleton_fallback += 1
            clusters_in_batch[cid].append(p)

        cluster_keys = list(clusters_in_batch.keys())
        rng.shuffle(cluster_keys)
        n_val_target = max(1, round(len(paths) * val_frac))
        val_paths, train_paths = [], []
        for ck in cluster_keys:
            if len(val_paths) < n_val_target:
                val_paths += clusters_in_batch[ck]
            else:
                train_paths += clusters_in_batch[ck]
        val_items += [(p, label_idx) for p in val_paths]
        train_items += [(p, label_idx) for p in train_paths]

    if n_singleton_fallback:
        log(f"[review11] WARNING: {n_singleton_fallback} images had no "
            f"manifest entry, treated as their own singleton cluster")
    log(f"[review11] cluster-aware split: train={len(train_items)} "
        f"val={len(val_items)} (no cluster crosses train/val)")

    train_ds = PathLabelDataset(train_items, train=True)
    val_ds = PathLabelDataset(val_items, train=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    set_seed(SEED)
    model = build_model(num_classes=len(batch_names), pretrained=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item() * x.size(0)
        log(f"[review11] epoch {epoch+1}/{epochs} loss={running_loss/len(train_ds):.4f}")

    model.eval()
    correct, total = 0, 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            pred = model(x).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_true += y.tolist()
            all_pred += pred.tolist()
    acc = correct / total if total else 0.0
    chance = 1.0 / len(batch_names)

    K = len(batch_names)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(all_true, all_pred):
        cm[t, p] += 1

    result = {
        "target_class": TARGET_CLASS,
        "n_batches": K,
        "batch_names": batch_names,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "split_method": "dihedral-cluster-aware (no cluster spans train/val)",
        "accuracy": acc,
        "chance_accuracy": chance,
        "accuracy_over_chance": acc / chance if chance else None,
        "confusion_matrix": cm.tolist(),
        "note": ("Compare against the original outputs/phase0/batch_id_diagnostic.json "
                 "(image-level split, not cluster-aware, accuracy=1.0). If accuracy "
                 "here remains near 1.0 despite the cluster-aware split, Review 11's "
                 "circularity objection is answered and the batch-correlated-shortcut "
                 "claim stands; if accuracy drops substantially, the original 100% "
                 "figure was inflated by near-duplicate-frame memorisation as the "
                 "review suspected."),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"[review11] cluster-aware {K}-way batch-ID accuracy = {acc:.4f} "
        f"({acc/chance:.1f}x chance, chance={chance:.4f})")


# =====================================================================
# SECTION 16: Review 16 -- shortcut audit aimed at the actual suspected
# shortcut (batch identity), not just the image margin.
# WHAT: (a) held-out-batch generalisation gap: train a Carcinoma-vs-rest
#       probe-style classifier on a SUBSET of batches, evaluate on
#       UNSEEN batches vs. a held-out slice of the SAME seen batches;
#       (b) a linear probe on the Stage 2 primary disease classifier's
#       own layer4 features, testing whether batch identity is linearly
#       recoverable from the same features the disease classifier uses
#       for its actual 9-class decision.
# WHY: the paper's only existing shortcut audit (margin-crop) is
#      explicitly acknowledged (Section 5.2) as unable to detect a
#      batch-correlated shortcut by design. This directly tests for
#      that shortcut instead.
# OUTPUT: outputs/review16_batch_shortcut_audit/result.json
# =====================================================================

def review_16_batch_shortcut_audit(rel_to_cluster, min_batch_size=25,
                                     n_seen_batches=6, n_unseen_batches=4,
                                     epochs=6):
    out_dir = os.path.join(OUT_ROOT, "review16_batch_shortcut_audit")
    out_path = os.path.join(out_dir, "result.json")
    part_a_only_path = os.path.join(out_dir, "result_part_a_only.json")
    if os.path.exists(out_path):
        log("[review16] already computed (both parts), skipping")
        return
    # NOTE: part (a) is cheap (~30s, a 6-way classifier on a few hundred
    # images) relative to the ~15-25 minute total runtime of this script,
    # so on a retry it is simply redone rather than building extra
    # reload/resume machinery for it -- only the genuinely expensive
    # Stage 2 retrain (a separate script) has full mid-run resumability.
    # If a previous run already got as far as writing result_part_a_only
    # (checkpoint missing at the time), this run will overwrite it with
    # a fresh, equivalent part (a) result and then proceed to part (b).
    if os.path.exists(part_a_only_path):
        log("[review16] part (a) previously completed but checkpoint was "
            "missing then; redoing part (a) (~30s) and retrying part (b) "
            "now that a checkpoint path may be attached")
    os.makedirs(out_dir, exist_ok=True)

    items_by_prefix = defaultdict(list)
    for split in ("training", "validation"):
        cdir = os.path.join(DATA_ROOT, split, TARGET_CLASS)
        if not os.path.isdir(cdir):
            continue
        for fname in os.listdir(cdir):
            m = FNAME_RE.match(fname)
            if m:
                items_by_prefix[m.group(1).lower()].append(os.path.join(cdir, fname))
    eligible = sorted(
        ((k, v) for k, v in items_by_prefix.items() if len(v) >= min_batch_size),
        key=lambda kv: -len(kv[1]))

    if len(eligible) < n_seen_batches + n_unseen_batches:
        log(f"[review16] WARNING: only {len(eligible)} eligible batches, "
            f"need {n_seen_batches + n_unseen_batches}; reducing counts")
        n_seen_batches = min(n_seen_batches, len(eligible) - 1)
        n_unseen_batches = len(eligible) - n_seen_batches

    seen_batches = eligible[:n_seen_batches]
    unseen_batches = eligible[n_seen_batches:n_seen_batches + n_unseen_batches]
    log(f"[review16] seen batches: {[k for k, _ in seen_batches]}; "
        f"unseen batches: {[k for k, _ in unseen_batches]}")

    # Part (a): held-out-batch generalisation gap on a batch-ID task
    # restricted to the SEEN batches only, cluster-aware split, vs.
    # accuracy on the fully UNSEEN batches (reassigned to the nearest
    # seen-batch label is not meaningful here -- instead we measure
    # whether a classifier trained to discriminate seen batches shows
    # any systematic behavior on unseen ones by reporting its confidence
    # distribution and forced-choice accuracy against a random label).
    rng = random.Random(SEED)

    def cluster_aware_split(paths, val_frac=0.30):
        clusters_in_batch = defaultdict(list)
        for p in paths:
            rk = rel_key_for_path(p)
            cid = rel_to_cluster.get(rk, f"singleton_{p}")
            clusters_in_batch[cid].append(p)
        cluster_keys = list(clusters_in_batch.keys())
        rng.shuffle(cluster_keys)
        n_val_target = max(1, round(len(paths) * val_frac))
        val_paths, train_paths = [], []
        for ck in cluster_keys:
            if len(val_paths) < n_val_target:
                val_paths += clusters_in_batch[ck]
            else:
                train_paths += clusters_in_batch[ck]
        return train_paths, val_paths

    seen_train_items, seen_val_items = [], []
    for label_idx, (bname, paths) in enumerate(seen_batches):
        tr, va = cluster_aware_split(paths)
        seen_train_items += [(p, label_idx) for p in tr]
        seen_val_items += [(p, label_idx) for p in va]

    unseen_items = []
    for bname, paths in unseen_batches:
        unseen_items += [(p, -1) for p in paths]  # -1: no valid seen-batch label

    train_ds = PathLabelDataset(seen_train_items, train=True)
    seen_val_ds = PathLabelDataset(seen_val_items, train=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    seen_val_loader = DataLoader(seen_val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    set_seed(SEED)
    K = len(seen_batches)
    model = build_model(num_classes=K, pretrained=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item() * x.size(0)
        log(f"[review16a] epoch {epoch+1}/{epochs} loss={running_loss/len(train_ds):.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in seen_val_loader:
            x = x.to(DEVICE)
            pred = model(x).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
    seen_val_acc = correct / total if total else 0.0

    # For unseen batches: report the model's confidence/entropy when
    # forced to classify images it has never seen a batch-mate of, as a
    # descriptive comparison (not an accuracy, since there is no correct
    # label among the K seen classes).
    unseen_ds = PathLabelDataset([(p, 0) for p, _ in unseen_items], train=False)
    unseen_loader = DataLoader(unseen_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    unseen_confidences = []
    with torch.no_grad():
        for x, _ in unseen_loader:
            x = x.to(DEVICE)
            probs = torch.softmax(model(x), dim=1)
            unseen_confidences += probs.max(dim=1).values.cpu().tolist()
    seen_confidences = []
    with torch.no_grad():
        for x, _ in seen_val_loader:
            x = x.to(DEVICE)
            probs = torch.softmax(model(x), dim=1)
            seen_confidences += probs.max(dim=1).values.cpu().tolist()

    part_a = {
        "seen_batches": [k for k, _ in seen_batches],
        "unseen_batches": [k for k, _ in unseen_batches],
        "seen_held_out_accuracy": seen_val_acc,
        "seen_held_out_chance": 1.0 / K,
        "mean_top1_confidence_on_seen_val": float(np.mean(seen_confidences)) if seen_confidences else None,
        "mean_top1_confidence_on_unseen": float(np.mean(unseen_confidences)) if unseen_confidences else None,
        "note": ("If the batch-ID classifier's top-1 confidence on unseen batches "
                 "is nearly as high as on seen held-out images despite having no "
                 "valid label to predict, that is further evidence of a strong, "
                 "generalising batch signature; if confidence drops substantially "
                 "on unseen batches, the signature may be more batch-specific "
                 "(e.g. a memorised artefact per acquisition) than a general "
                 "cross-batch shortcut."),
    }
    log(f"[review16a] seen-batch held-out accuracy={seen_val_acc:.4f} "
        f"(chance={1.0/K:.4f}); mean confidence seen={part_a['mean_top1_confidence_on_seen_val']:.4f} "
        f"unseen={part_a['mean_top1_confidence_on_unseen']:.4f}")

    # Part (b): linear probe on the Stage 2 disease classifier's own
    # layer4 features, testing whether batch identity is linearly
    # recoverable from the SAME features the 9-class disease decision
    # uses.
    part_b = {"ran": False, "reason": None}
    if os.path.exists(STAGE2_PRIMARY_CKPT):
        try:
            disease_model = build_model(num_classes=NUM_CLASSES, pretrained=False)
            disease_model.load_state_dict(torch.load(STAGE2_PRIMARY_CKPT, map_location=DEVICE, weights_only=True))
            disease_model.to(DEVICE).eval()

            all_batch_items = []
            for label_idx, (bname, paths) in enumerate(seen_batches):
                all_batch_items += [(p, label_idx) for p in paths]
            probe_train, probe_val = [], []
            for label_idx, (bname, paths) in enumerate(seen_batches):
                tr, va = cluster_aware_split(paths)
                probe_train += [(p, label_idx) for p in tr]
                probe_val += [(p, label_idx) for p in va]

            feat_layer = get_target_layer(disease_model)
            features = {}

            def hook(module, inp, out):
                features["value"] = out.detach()

            h = feat_layer.register_forward_hook(hook)

            def extract_features(items):
                ds = PathLabelDataset(items, train=False)
                loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
                feats, labels = [], []
                with torch.no_grad():
                    for x, y in loader:
                        x = x.to(DEVICE)
                        disease_model(x)
                        pooled = torch.nn.functional.adaptive_avg_pool2d(features["value"], 1).flatten(1)
                        feats.append(pooled.cpu().numpy())
                        labels += y.tolist()
                return np.concatenate(feats), np.array(labels)

            Xtr, ytr = extract_features(probe_train)
            Xva, yva = extract_features(probe_val)
            h.remove()

            from sklearn.linear_model import LogisticRegression
            probe = LogisticRegression(max_iter=2000, multi_class="auto")
            probe.fit(Xtr, ytr)
            probe_acc = probe.score(Xva, yva)
            chance = 1.0 / len(seen_batches)

            part_b = {
                "ran": True,
                "n_train": len(probe_train),
                "n_val": len(probe_val),
                "n_batches_probed": len(seen_batches),
                "linear_probe_accuracy": float(probe_acc),
                "chance_accuracy": chance,
                "accuracy_over_chance": float(probe_acc / chance),
                "note": ("A linear probe on the disease classifier's own layer4 "
                         "features, at whatever accuracy it achieves above chance, "
                         "shows batch identity is linearly recoverable from "
                         "features the disease classifier itself uses -- direct "
                         "evidence the disease classifier's representation "
                         "correlates with batch identity, not just that a "
                         "separately-trained classifier CAN learn batch identity."),
            }
            log(f"[review16b] linear probe on disease-classifier features: "
                f"batch-ID accuracy={probe_acc:.4f} (chance={chance:.4f}, "
                f"{probe_acc/chance:.1f}x chance)")
        except ImportError:
            part_b = {"ran": False, "reason": "scikit-learn not available"}
            log("[review16b] SKIPPED: scikit-learn not available (this IS a "
                "final, non-retriable outcome for this environment, so it is "
                "safe to record in the resume-marker file)")
        except Exception as e:
            part_b = {"ran": False, "reason": str(e)}
            log(f"[review16b] SKIPPED due to error: {e}")
    else:
        part_b["reason"] = f"Stage 2 checkpoint not found at {STAGE2_PRIMARY_CKPT}"
        log(f"[review16b] Stage 2 checkpoint not found -- will retry part (b) "
            f"on next run once the checkpoint dataset is attached correctly. "
            f"NOT writing the resume-marker file {out_path} yet, so this is "
            f"not mistaken for a completed run.")

    result = {"part_a_held_out_batch_gap": part_a, "part_b_linear_probe": part_b}
    if not os.path.exists(STAGE2_PRIMARY_CKPT):
        # Checkpoint genuinely missing: save part (a)'s real result under a
        # different filename so it survives, but do NOT write out_path
        # (the resume-marker checked at the top of this function), so a
        # later run retries part (b) instead of silently skipping forever.
        with open(os.path.join(out_dir, "result_part_a_only.json"), "w") as f:
            json.dump(result, f, indent=2)
        log(f"[review16] wrote {out_dir}/result_part_a_only.json "
            f"(part (a) complete, part (b) pending checkpoint)")
    else:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        log(f"[review16] wrote {out_path} (both parts complete or part (b) "
            f"terminally failed)")


# =====================================================================
# SECTION 17: Review 17 -- noise-robustness sweep with accuracy reported
# directly, not just as a stability side-effect.
# WHAT: sweeps Gaussian noise sigma across several levels and reports
#       classification accuracy directly at each level, using the Stage
#       2 primary checkpoint if available (falls back to reporting only
#       on whatever checkpoint is available).
# WHY: the paper currently reports one noise level (sigma=0.03, 29.2%
#      prediction-flip rate) as an aside inside the stability test and
#      never follows it up with a direct accuracy-under-perturbation
#      sweep, despite this being flagged as a more consequential finding
#      than anything in the XAI comparison.
# OUTPUT: outputs/review17_noise_sweep/result.json
# =====================================================================

def review_17_noise_sweep(sigmas=(0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12),
                            n_per_class=20):
    out_dir = os.path.join(OUT_ROOT, "review17_noise_sweep")
    out_path = os.path.join(out_dir, "result.json")
    # IMPORTANT: the checkpoint-availability check runs BEFORE the
    # already-done check, and a missing checkpoint does NOT write
    # out_path. This is deliberate: writing a "not found" placeholder to
    # the same path the resume-check reads would make a later run, after
    # the checkpoint dataset is finally attached, wrongly see that file
    # and skip the real analysis forever. A checkpoint.json file in a
    # SEPARATE path records the not-found event instead, so re-running
    # after fixing the checkpoint path always retries this section.
    if not os.path.exists(STAGE2_PRIMARY_CKPT):
        log(f"[review17] Stage 2 checkpoint not found at "
            f"{STAGE2_PRIMARY_CKPT} -- will retry on next run once the "
            f"checkpoint dataset is attached correctly. NOT writing "
            f"{out_path}, so this is not mistaken for a completed run.")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "checkpoint_not_found.json"), "w") as f:
            json.dump({"ran": False, "reason": "Stage 2 checkpoint not found",
                       "checked_path": STAGE2_PRIMARY_CKPT,
                       "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
        return
    if os.path.exists(out_path):
        log("[review17] already computed, skipping")
        return
    os.makedirs(out_dir, exist_ok=True)

    model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(STAGE2_PRIMARY_CKPT, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()

    by_class = defaultdict(list)
    for split in ("training", "validation"):
        for cname in CLASS_NAMES:
            cdir = os.path.join(DATA_ROOT, split, cname)
            if not os.path.isdir(cdir):
                continue
            for fname in os.listdir(cdir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    by_class[cname].append(os.path.join(cdir, fname))

    rng = random.Random(SEED)
    stratified = []
    for cname, paths in by_class.items():
        label = CLASS_NAMES.index(cname)
        chosen = rng.sample(paths, min(n_per_class, len(paths)))
        stratified += [(p, label) for p in chosen]
    log(f"[review17] sweeping {len(sigmas)} noise levels on {len(stratified)} images")

    base_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
    ])
    normalize = transforms.Normalize(mean=MEAN, std=STD)

    # clean_preds is deliberately OUTSIDE the sigma loop: it must persist
    # across all sigma iterations so later, noisier sigmas are compared
    # against the sigma=0.0 baseline prediction for the SAME image, not
    # reset to empty each time (an earlier version of this script had
    # clean_preds = {} inside the loop, which silently made every
    # pred_changed_from_clean_fraction read as 0.0 regardless of how much
    # accuracy actually dropped -- fixed here; sigmas must include 0.0
    # and be swept in ascending order, which the default sigmas tuple
    # above already satisfies, so the baseline is always populated
    # before any sigma>0 iteration needs to read it).
    clean_preds = {}
    results_by_sigma = {}
    for sigma in sigmas:
        correct = 0
        pred_changed_from_clean = 0
        for path, label in stratified:
            img = Image.open(path).convert("RGB")
            x0 = base_transform(img).unsqueeze(0)
            if sigma > 0:
                noise = torch.randn_like(x0) * sigma
                x0 = torch.clamp(x0 + noise, 0, 1)
            xn = normalize(x0).to(DEVICE)
            with torch.no_grad():
                pred = model(xn).argmax(1).item()
            if sigma == 0.0:
                clean_preds[path] = pred
            else:
                if clean_preds[path] != pred:
                    pred_changed_from_clean += 1
            correct += int(pred == label)
        acc = correct / len(stratified)
        results_by_sigma[str(sigma)] = {
            "accuracy": acc,
            "n_images": len(stratified),
            "pred_changed_from_clean_fraction": (
                pred_changed_from_clean / len(stratified) if sigma > 0 else 0.0),
        }
        log(f"[review17] sigma={sigma:.3f}  accuracy={acc:.4f}  "
            f"pred_changed_from_clean={results_by_sigma[str(sigma)]['pred_changed_from_clean_fraction']:.4f}")

    with open(out_path, "w") as f:
        json.dump({
            "ran": True,
            "checkpoint_used": STAGE2_PRIMARY_CKPT,
            "sigmas_swept": list(sigmas),
            "results_by_sigma": results_by_sigma,
            "note": ("Accuracy reported directly at each noise level, on the "
                     "Stage 2 (leakage-corrected) primary checkpoint, closing "
                     "the gap where the paper previously reported only a "
                     "single sigma=0.03 prediction-flip rate as a stability "
                     "side-effect rather than a direct accuracy-under-"
                     "perturbation result."),
        }, f, indent=2)
    log(f"[review17] wrote {out_path}")


# =====================================================================
# main
# =====================================================================

def main():
    log("=== uidatagb_stage2_remaining_reviews.py started ===")
    log(f"Device: {DEVICE}")
    if not os.path.isdir(DATA_ROOT):
        log(f"FATAL: DATA_ROOT {DATA_ROOT} does not exist. Check the "
            f"attached Kaggle dataset's mount path and update DATA_ROOT "
            f"at the top of this file.")
        sys.exit(1)
    if not os.path.exists(CLUSTER_MANIFEST_PATH):
        log(f"FATAL: CLUSTER_MANIFEST_PATH {CLUSTER_MANIFEST_PATH} does "
            f"not exist. Reviews 11 and 16 both require it.")
        sys.exit(1)

    rel_to_cluster = load_cluster_manifest()

    review_11_batch_id_cluster_aware(rel_to_cluster)
    review_16_batch_shortcut_audit(rel_to_cluster)
    review_17_noise_sweep()

    log("=== ALL DONE ===")
    log(f"Outputs written under: {OUT_ROOT}")

    try:
        import shutil
        zip_path = shutil.make_archive(OUT_ROOT, "zip", OUT_ROOT)
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        log(f"[auto-zip] wrote {zip_path} ({size_mb:.1f} MB)")
    except Exception as e:
        log(f"[auto-zip] WARNING: zipping failed ({e}); outputs are still "
            f"present, unzipped, under {OUT_ROOT}")

    log("TO DOWNLOAD: do NOT rely on any FileLink-style link -- it will "
        "404 on Kaggle. Open the 'Output' panel in the right-hand "
        "sidebar of this notebook, find "
        "'uidatagb_stage2_reviews_outputs.zip', and click its own "
        "download icon there.")
    log("IMPORTANT: click 'Save Version -> Save & Run All' now, before "
        "downloading, the same way the main Stage 2 retrain's results "
        "were saved.")


if __name__ == "__main__":
    main()
