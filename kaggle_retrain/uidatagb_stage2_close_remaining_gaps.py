"""
uidatagb_stage2_close_remaining_gaps.py
=========================================
Closes the three remaining COMPUTABLE gaps identified after the full
Stage 2 XAI battery re-run (uidatagb_stage2_full_battery.py). Two other
gaps -- clinician-annotated localization and multi-site/scanner
validation -- are NOT computable from this dataset release and are not
attempted here; they stay honestly reported as open in the Limitations
section.

This script closes:

  Gap A -- pHash threshold sensitivity, accuracy half. The existing
    phash_threshold_sensitivity.py already swept cluster count and
    mixed-class count across 5 thresholds (4,6,8,10,12 bits), but
    explicitly did NOT retrain a classifier at each threshold to report
    real Stage 2 accuracy per threshold. This section does that: for
    each of the 5 thresholds, it re-clusters (reusing the cached
    dihedral hashes, no re-hashing needed), re-splits 70/20/10 at the
    cluster level, trains one primary-seed model, and reports its
    held-out test accuracy. This is 5 real training runs.

  Gap B -- batch-shortcut linear probe, full-coverage half. The
    existing linear probe (Review 16b, in
    uidatagb_stage2_remaining_reviews.py) covered only ONE disease
    class (Carcinoma) and 6 of its batches. This section repeats the
    same linear-probe design across ALL NINE disease classes and ALL
    eligible batches within each (not just the single largest class),
    so the batch-shortcut finding is no longer scoped to one class.

  Gap C -- failure-case analysis on Stage 2. The paper's grounded
    failure-case analysis (Section IV.2) was run on the Stage 1
    checkpoint/split and never re-run on Stage 2's genuinely held-out
    test set. This section re-runs the EXISTING, unmodified
    analyze_failure_cases_uidatagb.py script (fedgb/) against the
    Stage 2 primary checkpoint and materialized test split, via the
    same P0_CKPT/P0_VAL_DIR/P0_FIG_DIR/P0_FAILURE_JSON env-var pattern
    already used for the other battery scripts -- no code changes to
    that script are needed.

Each gap is its own independently-checkpointed section (same
resumability pattern as uidatagb_stage2_remaining_reviews.py and
uidatagb_full_retrain.py): if the session restarts partway through, a
re-run skips whatever already finished and resumes the rest.

HOW TO RUN THIS ON KAGGLE
--------------------------------------------------------------------
1. Attach the same three datasets as uidatagb_stage2_full_battery.py:
   "uidatagb", "uidatagb-retrain-outputs" (contains the Stage 2
   checkpoints + split_70_20_10.json), and "fedgb-code".
2. Also needs outputs/phase0/dihedral_hashes_cache.npy and
   outputs/phase0/dihedral_hashes_cache_items.json (the cached hashes
   phash_threshold_sensitivity.py already used) -- these should already
   be inside the fedgb-code dataset if it was zipped from the fedgb/
   directory with its outputs/ folder included; if not, they will be
   recomputed once here (slower, CPU-only, but only needs to happen
   once and is itself cached for the 5-threshold sweep).
3. Copy fedgb-code into /kaggle/working/fedgb, same as before:
     !cp -r /kaggle/input/datasets/sayed227/fedgb-code/fedgb /kaggle/working/fedgb
4. Turn on GPU T4 x2.
5. Run all. Gap A is the expensive part: 5 training runs, each a
   lighter budget than the main Stage 2 retrain (10 epochs, no
   replicate models, single seed) since this sweep only needs to show
   the accuracy TREND across thresholds, not another fully-tuned
   model. Expect 45-90 minutes total, most of it Gap A. Gaps B and C
   are cheap (a few minutes each) and run after Gap A finishes.
6. At the end, auto-zips outputs. Use the Output side panel to
   download, never a printed FileLink (404s on Kaggle).
"""
import json
import os
import random
import re
import shutil
import subprocess
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
# Same real, verified mount paths as uidatagb_stage2_full_battery.py.
# =====================================================================

DATA_ROOT = "/kaggle/input/datasets/sayed227/uidatagb/uidatagb-corrected/uidatagb"
CLUSTER_MANIFEST_PATH = "/kaggle/input/datasets/sayed227/uidatagb/uidatagb-cluster-manifest.json"

CKPT_ROOT = "/kaggle/input/datasets/sayed227/uidatagb-retrain-outputs"
SPLIT_JSON_PATH = os.path.join(CKPT_ROOT, "split_70_20_10.json")
PRIMARY_CKPT = os.path.join(CKPT_ROOT, "primary_seed42", "best.pt")

FEDGB_DIR = "/kaggle/working/fedgb"
HASH_CACHE_PATH = os.path.join(FEDGB_DIR, "outputs", "phase0", "dihedral_hashes_cache.npy")
HASH_CACHE_ITEMS_PATH = os.path.join(FEDGB_DIR, "outputs", "phase0", "dihedral_hashes_cache_items.json")

MATERIALIZED_TEST_DIR = "/kaggle/working/stage2_test_materialized"
OUT_ROOT = "/kaggle/working/uidatagb_stage2_close_remaining_gaps_outputs"
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
FNAME_RE = re.compile(r'^([A-Za-z]+\d+)\s*\(\d+\)\.\w+$')

SEED = 42
# Two explicit, fixed device handles -- never mutated after this point.
# Gap A (training-heavy) runs pinned to DEVICE_0; Gaps B and C
# (inference-only) run pinned to DEVICE_1. Every function below takes
# its device as an explicit argument rather than reading a shared
# global, so the two threads launched in main() cannot race on which
# GPU a given tensor lands on.
DEVICE_0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE_1 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else DEVICE_0)
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
        self.items = items
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def list_all_images():
    items = []
    for split in ("training", "validation"):
        for cname in CLASS_NAMES:
            cdir = os.path.join(DATA_ROOT, split, cname)
            if not os.path.isdir(cdir):
                continue
            for fname in os.listdir(cdir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    items.append((os.path.join(cdir, fname), cname, split))
    return items


def rel_key_for_path(path):
    norm = path.replace("\\", "/")
    parts = norm.split("/")
    for i, p in enumerate(parts):
        if p.lower() in ("training", "validation"):
            return "/".join(parts[i:])
    return "/".join(parts[-3:])


def popcount64_array(x):
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0f0f0f0f0f0f0f0f)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


# =====================================================================
# GAP A: pHash threshold sensitivity, accuracy half.
# For each of 5 thresholds: recluster (cached hashes), resplit
# 70/20/10 at the cluster level, train ONE primary-seed model
# (lighter budget than the main retrain -- 10 fixed epochs, no
# replicates, since this sweep only needs the accuracy TREND across
# thresholds, not another fully-tuned model), report test accuracy.
# =====================================================================

def dihedral_union_find_clusters(hashes, thresh, batch=300):
    n = hashes.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for start in range(0, n, batch):
        end = min(start + batch, n)
        chunk = hashes[start:end]
        rest = hashes[start:]
        xor = chunk[:, None, :, None] ^ rest[None, :, None, :]
        dist = popcount64_array(xor)
        min_dist = dist.min(axis=(2, 3))
        close_i, close_j = np.where(min_dist <= thresh)
        for ii, jj in zip(close_i, close_j):
            gi, gj = start + ii, start + jj
            if gi != gj:
                union(gi, gj)

    roots = [find(i) for i in range(n)]
    clusters = {}
    for i, r in enumerate(roots):
        clusters.setdefault(r, []).append(i)
    return list(clusters.values())


def apply_chaining_backstop(raw_clusters, items, max_classes=2, max_size=30):
    clusters, vetoed = [], []
    for c in raw_clusters:
        classes_in_c = set(items[i][1] for i in c)
        if len(classes_in_c) > max_classes or len(c) > max_size:
            vetoed.append(c)
            for i in c:
                clusters.append([i])
        else:
            clusters.append(c)
    return clusters, vetoed


def split_clusters_70_20_10(clusters, items, seed=SEED):
    rng = random.Random(seed)
    # majority class per cluster, for stratified-ish assignment
    cluster_info = []
    for c in clusters:
        classes = [items[i][1] for i in c]
        maj = max(set(classes), key=classes.count)
        cluster_info.append((c, maj))
    rng.shuffle(cluster_info)

    by_class = defaultdict(list)
    for c, maj in cluster_info:
        by_class[maj].append(c)

    train_idx, val_idx, test_idx = [], [], []
    for cname, clist in by_class.items():
        n = len(clist)
        n_test = max(1, round(n * 0.10))
        n_val = max(1, round(n * 0.20))
        test_c = clist[:n_test]
        val_c = clist[n_test:n_test + n_val]
        train_c = clist[n_test + n_val:]
        for c in test_c:
            test_idx += c
        for c in val_c:
            val_idx += c
        for c in train_c:
            train_idx += c
    return train_idx, val_idx, test_idx


def train_one_model(train_items, val_items, device, epochs=10):
    train_ds = PathLabelDataset(train_items, train=True)
    val_ds = PathLabelDataset(val_items, train=False)
    train_loader = DataLoader(train_ds, batch_size=96, shuffle=True,
                               num_workers=2, pin_memory=True,
                               persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=96, shuffle=False,
                             num_workers=2, pin_memory=True)

    set_seed(SEED)
    model = build_model(pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item() * x.size(0)
        log(f"  epoch {epoch+1}/{epochs} loss={running_loss/len(train_ds):.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0


def gap_a_phash_threshold_accuracy(device, thresholds=(4, 6, 8, 10, 12), epochs=10):
    out_dir = os.path.join(OUT_ROOT, "gap_a_phash_threshold_accuracy")
    out_path = os.path.join(out_dir, "result.json")
    if os.path.exists(out_path):
        log("[gap_a] already computed, skipping")
        with open(out_path) as f:
            return json.load(f)
    os.makedirs(out_dir, exist_ok=True)

    # per-threshold progress file lets a restart resume mid-sweep
    progress_path = os.path.join(out_dir, "progress.json")
    done = {}
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            done = json.load(f)
        log(f"[gap_a] resuming: {len(done)}/{len(thresholds)} thresholds already done")

    if not (os.path.exists(HASH_CACHE_PATH) and os.path.exists(HASH_CACHE_ITEMS_PATH)):
        log(f"[gap_a] FATAL: hash cache not found at {HASH_CACHE_PATH}. "
            f"This must ship inside the fedgb-code dataset's outputs/phase0/ "
            f"folder, or be generated by generate_dihedral_cluster_manifest.py "
            f"first. Skipping Gap A entirely (Gaps B and C will still run).")
        return None

    hashes = np.load(HASH_CACHE_PATH)
    with open(HASH_CACHE_ITEMS_PATH) as f:
        cached_paths = json.load(f)
    items_on_disk = list_all_images()
    if cached_paths != [it[0] for it in items_on_disk]:
        log("[gap_a] WARNING: cached item order does not match current disk "
            "listing (the cache was generated on a different machine, so "
            "its paths are stale); reconstructing items from the cache's "
            "own path list, REBASED onto this session's real DATA_ROOT, "
            "so hash indices stay aligned but the resulting paths are "
            "actually openable here. Any cached entry that cannot be "
            "rebased to a real file is dropped from BOTH items and the "
            "hashes array together (by index), so the two stay aligned -- "
            "dropping only from items while leaving hashes at its "
            "original length would silently desync hashes[i] from "
            "items[i] for every i past the first drop.")
        items = []
        keep_mask = []
        n_unmatched = 0
        for p in cached_paths:
            norm = p.replace("\\", "/")
            parts = norm.split("/")
            rel_parts = None
            for i, part in enumerate(parts):
                if part.lower() in ("training", "validation"):
                    rel_parts = parts[i:]
                    break
            if rel_parts is None:
                n_unmatched += 1
                keep_mask.append(False)
                continue
            real_path = os.path.join(DATA_ROOT, *rel_parts)
            if not os.path.exists(real_path):
                keep_mask.append(False)
                continue
            cname = next((c for c in CLASS_NAMES if c in rel_parts), "UNKNOWN")
            items.append((real_path, cname, rel_parts[0]))
            keep_mask.append(True)
        n_dropped = len(keep_mask) - sum(keep_mask)
        if n_dropped:
            log(f"[gap_a] WARNING: {n_dropped}/{len(keep_mask)} cached "
                f"entries dropped ({n_unmatched} had no training/validation "
                f"segment, {n_dropped - n_unmatched} rebased to a path "
                f"that does not exist on disk); filtering hashes to match "
                f"by the same mask so hashes[i] still lines up with items[i]")
            hashes = hashes[np.array(keep_mask)]
    else:
        items = items_on_disk

    log(f"[gap_a] {len(items)} images, sweeping thresholds {list(thresholds)}, "
        f"{epochs} epochs/threshold")

    for thresh in thresholds:
        if str(thresh) in done:
            log(f"[gap_a] threshold={thresh} already done "
                f"(acc={done[str(thresh)]['test_accuracy']:.4f}), skipping")
            continue
        t0 = time.time()
        raw_clusters = dihedral_union_find_clusters(hashes, thresh)
        clusters, vetoed = apply_chaining_backstop(raw_clusters, items)
        train_idx, val_idx, test_idx = split_clusters_70_20_10(clusters, items)

        train_items = [(items[i][0], CLASS_NAMES.index(items[i][1])) for i in train_idx]
        val_items = [(items[i][0], CLASS_NAMES.index(items[i][1])) for i in val_idx]
        test_items = [(items[i][0], CLASS_NAMES.index(items[i][1])) for i in test_idx]

        log(f"[gap_a] threshold={thresh}: {len(clusters)} clusters, "
            f"train={len(train_items)} val={len(val_items)} test={len(test_items)}")

        test_acc = train_one_model(train_items, val_items=test_items, device=device, epochs=epochs)
        dt = time.time() - t0
        done[str(thresh)] = {
            "threshold": thresh,
            "n_clusters": len(clusters),
            "n_mixed_class_clusters": sum(
                1 for c in clusters if len(set(items[i][1] for i in c)) > 1),
            "n_train": len(train_items),
            "n_val_unused_here": len(val_items),
            "n_test": len(test_items),
            "test_accuracy": test_acc,
            "epochs": epochs,
            "wall_time_seconds": round(dt, 1),
        }
        with open(progress_path, "w") as f:
            json.dump(done, f, indent=2)
        log(f"[gap_a] threshold={thresh}: test_accuracy={test_acc:.4f} ({dt:.0f}s)")

    result = {
        "thresholds_swept": list(thresholds),
        "epochs_per_run": epochs,
        "by_threshold": done,
        "note": ("Real Stage 2 test accuracy at each pHash clustering "
                 "threshold, closing the half of the original threshold-"
                 "sensitivity sweep (fedgb/phash_threshold_sensitivity.py) "
                 "that explicitly did not retrain a classifier per "
                 "threshold. Each run uses a single seed and a fixed "
                 f"{epochs}-epoch budget (lighter than the main Stage 2 "
                 "retrain's early-stopped, up-to-20-epoch, 4-seed design), "
                 "since this sweep's purpose is to show the accuracy TREND "
                 "across thresholds, not to produce another publication-"
                 "grade checkpoint."),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"[gap_a] wrote {out_path}")
    return result


# =====================================================================
# GAP B: batch-shortcut linear probe, extended to ALL nine classes and
# all eligible batches within each (not just Carcinoma / 6 batches).
# =====================================================================

def gap_b_full_coverage_linear_probe(device, min_batch_size=15, val_frac=0.30):
    out_dir = os.path.join(OUT_ROOT, "gap_b_full_coverage_linear_probe")
    out_path = os.path.join(out_dir, "result.json")
    if os.path.exists(out_path):
        log("[gap_b] already computed, skipping")
        return
    if not os.path.exists(PRIMARY_CKPT):
        log(f"[gap_b] Stage 2 checkpoint not found at {PRIMARY_CKPT} -- "
            f"will retry on next run. NOT writing {out_path}.")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "checkpoint_not_found.json"), "w") as f:
            json.dump({"ran": False, "reason": "checkpoint not found",
                       "checked_path": PRIMARY_CKPT}, f, indent=2)
        return
    os.makedirs(out_dir, exist_ok=True)

    disease_model = build_model(num_classes=NUM_CLASSES, pretrained=False)
    disease_model.load_state_dict(torch.load(PRIMARY_CKPT, map_location=device, weights_only=True))
    disease_model.to(device).eval()

    feat_layer = get_target_layer(disease_model)
    features = {}

    def hook(module, inp, out):
        features["value"] = out.detach()

    h = feat_layer.register_forward_hook(hook)

    def extract_features(items):
        ds = PathLabelDataset(items, train=False)
        loader = DataLoader(ds, batch_size=96, shuffle=False, num_workers=2)
        feats, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                disease_model(x)
                pooled = torch.nn.functional.adaptive_avg_pool2d(features["value"], 1).flatten(1)
                feats.append(pooled.cpu().numpy())
                labels += y.tolist()
        return np.concatenate(feats), np.array(labels)

    from sklearn.linear_model import LogisticRegression

    rng = random.Random(SEED)
    per_class_results = {}

    for cname in CLASS_NAMES:
        items_by_prefix = defaultdict(list)
        for split in ("training", "validation"):
            cdir = os.path.join(DATA_ROOT, split, cname)
            if not os.path.isdir(cdir):
                continue
            for fname in os.listdir(cdir):
                m = FNAME_RE.match(fname)
                if m:
                    items_by_prefix[m.group(1).lower()].append(os.path.join(cdir, fname))

        eligible = sorted(
            ((k, v) for k, v in items_by_prefix.items() if len(v) >= min_batch_size),
            key=lambda kv: -len(kv[1]))

        if len(eligible) < 2:
            log(f"[gap_b] {cname}: only {len(eligible)} eligible batch(es) "
                f"(>= {min_batch_size} images), skipping (need >= 2 to probe)")
            per_class_results[cname] = {"ran": False, "reason": "fewer than 2 eligible batches"}
            continue

        probe_train, probe_val = [], []
        for label_idx, (bname, paths) in enumerate(eligible):
            paths = paths[:]
            rng.shuffle(paths)
            n_val = max(1, round(len(paths) * val_frac))
            probe_val += [(p, label_idx) for p in paths[:n_val]]
            probe_train += [(p, label_idx) for p in paths[n_val:]]

        Xtr, ytr = extract_features(probe_train)
        Xva, yva = extract_features(probe_val)

        probe = LogisticRegression(max_iter=2000)
        probe.fit(Xtr, ytr)
        probe_acc = probe.score(Xva, yva)
        chance = 1.0 / len(eligible)

        per_class_results[cname] = {
            "ran": True,
            "n_batches_probed": len(eligible),
            "batch_sizes": [len(v) for _, v in eligible],
            "n_train": len(probe_train),
            "n_val": len(probe_val),
            "linear_probe_accuracy": float(probe_acc),
            "chance_accuracy": chance,
            "accuracy_over_chance": float(probe_acc / chance),
        }
        log(f"[gap_b] {cname}: {len(eligible)}-way batch-ID probe accuracy="
            f"{probe_acc:.4f} (chance={chance:.4f}, {probe_acc/chance:.1f}x)")

    h.remove()

    ran_results = [v for v in per_class_results.values() if v.get("ran")]
    macro_mean_over_chance = (
        float(np.mean([v["accuracy_over_chance"] for v in ran_results]))
        if ran_results else None)

    result = {
        "per_class": per_class_results,
        "n_classes_probed": len(ran_results),
        "n_classes_total": len(CLASS_NAMES),
        "macro_mean_accuracy_over_chance": macro_mean_over_chance,
        "note": ("Extends Review 16b's linear probe (originally Carcinoma "
                 "only, 6 batches) to all nine disease classes and all "
                 "eligible batches within each, on the same Stage 2 "
                 "primary disease classifier's own layer4 features. A "
                 "class is skipped if it has fewer than 2 batches with "
                 f">= {min_batch_size} images, since a probe needs at "
                 "least 2 classes to be meaningful."),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log(f"[gap_b] wrote {out_path} (macro mean accuracy-over-chance="
        f"{macro_mean_over_chance:.2f}x across {len(ran_results)} classes)"
        if macro_mean_over_chance else f"[gap_b] wrote {out_path} (no classes had enough batches)")


# =====================================================================
# GAP C: failure-case analysis re-run on Stage 2, via the existing,
# unmodified fedgb/analyze_failure_cases_uidatagb.py script.
# =====================================================================

def materialize_test_split():
    if os.path.isdir(MATERIALIZED_TEST_DIR) and any(os.scandir(MATERIALIZED_TEST_DIR)):
        log(f"[materialize] {MATERIALIZED_TEST_DIR} already populated, skipping")
        return
    if not os.path.exists(SPLIT_JSON_PATH):
        log(f"[materialize] FATAL: {SPLIT_JSON_PATH} not found; Gap C cannot run.")
        return
    with open(SPLIT_JSON_PATH) as f:
        split = json.load(f)
    test_items = split["test"]
    for cname in CLASS_NAMES:
        os.makedirs(os.path.join(MATERIALIZED_TEST_DIR, cname), exist_ok=True)

    def rebase(stored_path):
        norm = stored_path.replace("\\", "/")
        parts = norm.split("/")
        for i, p in enumerate(parts):
            if p.lower() in ("training", "validation"):
                return os.path.join(DATA_ROOT, *parts[i:])
        return None

    n_copied, n_missing = 0, 0
    for stored_path, label_idx in test_items:
        real_path = rebase(stored_path)
        cname = CLASS_NAMES[label_idx]
        if real_path is None or not os.path.exists(real_path):
            n_missing += 1
            continue
        dest = os.path.join(MATERIALIZED_TEST_DIR, cname, os.path.basename(real_path))
        if not os.path.exists(dest):
            shutil.copy2(real_path, dest)
        n_copied += 1
    log(f"[materialize] copied {n_copied} images, {n_missing} missing")


def gap_c_stage2_failure_cases():
    out_dir = os.path.join(OUT_ROOT, "gap_c_stage2_failure_cases")
    out_path = os.path.join(out_dir, "phase0", "failure_cases.json")
    if os.path.exists(out_path):
        log("[gap_c] already computed, skipping")
        return
    if not os.path.exists(PRIMARY_CKPT):
        log(f"[gap_c] Stage 2 checkpoint not found at {PRIMARY_CKPT} -- skipping for now")
        return
    if not os.path.isdir(FEDGB_DIR):
        log(f"[gap_c] FATAL: {FEDGB_DIR} not found; cannot run the existing "
            f"analyze_failure_cases_uidatagb.py script.")
        return

    materialize_test_split()
    if not (os.path.isdir(MATERIALIZED_TEST_DIR) and any(os.scandir(MATERIALIZED_TEST_DIR))):
        log("[gap_c] materialized test split is empty; skipping")
        return

    os.makedirs(os.path.join(out_dir, "phase0"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(DEVICE_1.index) if DEVICE_1.type == "cuda" else ""
    env["P0_CKPT"] = PRIMARY_CKPT
    env["P0_VAL_DIR"] = MATERIALIZED_TEST_DIR
    env["P0_FIG_DIR"] = os.path.join(out_dir, "figures")
    env["P0_FAILURE_JSON"] = out_path

    log("[gap_c] running fedgb/analyze_failure_cases_uidatagb.py against "
        "the Stage 2 primary checkpoint and materialized test split")
    proc = subprocess.run(
        [sys.executable, "analyze_failure_cases_uidatagb.py"],
        cwd=FEDGB_DIR, env=env, capture_output=True, text=True,
    )
    log(f"[gap_c] exit={proc.returncode}")
    if proc.stdout:
        log("[gap_c] stdout tail:\n" + "\n".join(proc.stdout.splitlines()[-20:]))
    if proc.returncode != 0:
        log("[gap_c] stderr tail:\n" + "\n".join(proc.stderr.splitlines()[-30:]))


# =====================================================================
# main
# =====================================================================

def main():
    log(f"=== uidatagb_stage2_close_remaining_gaps.py started ===")
    log(f"DEVICE_0: {DEVICE_0}, DEVICE_1: {DEVICE_1}, "
        f"GPU count: {torch.cuda.device_count()}")

    if not os.path.isdir(DATA_ROOT):
        log(f"FATAL: DATA_ROOT {DATA_ROOT} does not exist.")
        sys.exit(1)

    # Gap A (expensive, training-heavy) runs pinned to DEVICE_0; Gaps B
    # and C (both inference-only) run sequentially on DEVICE_1. Each
    # function takes its device as an explicit argument (no shared
    # mutable global), so the two threads cannot race on GPU placement.
    import concurrent.futures

    def run_gaps_b_and_c():
        gap_b_full_coverage_linear_probe(device=DEVICE_1)
        gap_c_stage2_failure_cases()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(gap_a_phash_threshold_accuracy, device=DEVICE_0)
        future_bc = executor.submit(run_gaps_b_and_c)
        future_a.result()
        future_bc.result()

    log("=== ALL DONE ===")
    log(f"Outputs written under: {OUT_ROOT}")

    try:
        zip_path = shutil.make_archive(OUT_ROOT, "zip", OUT_ROOT)
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        log(f"[auto-zip] wrote {zip_path} ({size_mb:.1f} MB)")
    except Exception as e:
        log(f"[auto-zip] WARNING: zipping failed ({e})")

    log("TO DOWNLOAD: use the Output side panel, not any printed link "
        "(FileLink-style links 404 on Kaggle).")
    log("IMPORTANT: click 'Save Version -> Save & Run All' before downloading.")


if __name__ == "__main__":
    main()
