"""
session_leakage_check.py
-------------------------
Quick robustness check: does the paper's existing near-duplicate (pHash)
leakage correction actually close patient/session-level leakage, or does it
miss a coarser, stronger leakage channel?

Every UIdataGB filename follows the pattern "<prefix><n> (<k>).jpg", e.g.
"a1 (7).jpg". The numeric suffix k is perfectly contiguous within each
prefix group (verified separately), and no prefix group spans more than one
disease class -- both are the signature of one continuous video/scan
acquisition, i.e. the prefix is a strong proxy for patient/session identity
that the pHash-based correction does not use.

This script:
  1. Lists the CURRENT (already pHash-corrected) train/validation split.
  2. Groups images by filename-prefix ("session").
  3. Reports how many sessions / images still cross the train/validation
     boundary even after the pHash correction.
  4. Rebuilds a NEW split, stratified at the session level (never splitting
     one session across train/val), and copies files into a separate
     directory (data/uidatagb_sessioncorrected/) -- the existing corrected
     split used by the paper is left untouched.
  5. Trains a single ResNet-50 (same hyperparameters as Table 2, seed=42)
     on the new split and reports accuracy / macro-F1 / macro-AUC, for
     direct comparison against the paper's reported 93.05% / 0.9303 / 0.9965
     (pHash-corrected) and 98.78% (original contaminated) figures.

Run from fedgb/:  python session_leakage_check.py
"""
import json
import os
import random
import re
import shutil
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, get_transforms, GBCUDataset
from model import build_model
from evaluate import run_inference, compute_metrics

DATA_DIR = "./data/uidatagb"
NEW_DATA_DIR = "./data/uidatagb_sessioncorrected"
OUT_DIR = "./outputs/phase0"
TRAIN_FRAC = 0.70
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FNAME_RE = re.compile(r'^([A-Za-z]+\d+)\s*\(\d+\)\.\w+$')


def list_images():
    items = []
    for split in ("training", "validation"):
        for cname in CLASS_NAMES:
            cdir = os.path.join(DATA_DIR, split, cname)
            if not os.path.isdir(cdir):
                continue
            for fname in os.listdir(cdir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    items.append((os.path.join(cdir, fname), fname, cname, split))
    return items


def session_key(fname):
    m = FNAME_RE.match(fname)
    return m.group(1).lower() if m else None


def main():
    t0 = time.time()
    items = list_images()
    print(f"[session-check] total images (current pHash-corrected split): {len(items)}")

    sess_splits = defaultdict(set)
    sess_class = defaultdict(set)
    sess_items = defaultdict(list)
    unmatched = 0
    for path, fname, cname, split in items:
        key = session_key(fname)
        if key is None:
            unmatched += 1
            continue
        sess_splits[key].add(split)
        sess_class[key].add(cname)
        sess_items[key].append((path, fname, cname, split))

    print(f"[session-check] unmatched filenames (no session key): {unmatched}")
    print(f"[session-check] total session groups: {len(sess_items)}")

    cross = {k: v for k, v in sess_splits.items() if len(v) > 1}
    cross_images = sum(len(sess_items[k]) for k in cross)
    print(f"[session-check] sessions still crossing train/val after pHash correction: "
          f"{len(cross)}/{len(sess_items)}")
    print(f"[session-check] images belonging to a cross-split session: "
          f"{cross_images}/{len(items)} ({100*cross_images/len(items):.1f}%)")

    mixed_class_sessions = sum(1 for v in sess_class.values() if len(v) > 1)
    print(f"[session-check] sessions spanning >1 class label: {mixed_class_sessions}")

    # ---- rebuild split at session level, stratified by (majority) class ----
    sess_majority_class = {}
    for k, its in sess_items.items():
        classes = [c for _, _, c, _ in its]
        sess_majority_class[k] = max(set(classes), key=classes.count)

    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for k, cname in sess_majority_class.items():
        by_class[cname].append(k)

    train_sessions, val_sessions = set(), set()
    for cname, keys in by_class.items():
        rng.shuffle(keys)
        n_train = round(len(keys) * TRAIN_FRAC)
        train_sessions.update(keys[:n_train])
        val_sessions.update(keys[n_train:])

    # materialize new split directory (non-destructive: copies, doesn't move)
    if os.path.isdir(NEW_DATA_DIR):
        shutil.rmtree(NEW_DATA_DIR)
    final_counts = {"training": {c: 0 for c in CLASS_NAMES},
                     "validation": {c: 0 for c in CLASS_NAMES}}
    for k, its in sess_items.items():
        target = "training" if k in train_sessions else "validation"
        for path, fname, cname, _ in its:
            out_dir = os.path.join(NEW_DATA_DIR, target, cname)
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(path, os.path.join(out_dir, fname))
            final_counts[target][cname] += 1

    n_train = sum(final_counts["training"].values())
    n_val = sum(final_counts["validation"].values())
    print(f"\n[session-check] new session-level split: train={n_train} val={n_val} "
          f"({100*n_train/(n_train+n_val):.1f}% / {100*n_val/(n_train+n_val):.1f}%)")

    report = {
        "n_images_checked": len(items),
        "n_sessions": len(sess_items),
        "n_cross_split_sessions": len(cross),
        "n_cross_split_images": cross_images,
        "pct_cross_split_images": round(100 * cross_images / len(items), 2),
        "n_mixed_class_sessions": mixed_class_sessions,
        "new_split_train_total": n_train,
        "new_split_val_total": n_val,
        "new_split_counts": final_counts,
    }
    with open(os.path.join(OUT_DIR, "session_leakage_check_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[session-check] wrote {OUT_DIR}/session_leakage_check_report.json "
          f"({time.time()-t0:.0f}s elapsed)")

    # ---- train ResNet-50 on the session-corrected split (Table 2 config) ----
    print("\n[session-check] training ResNet-50 on session-corrected split "
          "(Adam, lr=1e-4, wd=1e-4, batch=32, 10 epochs, seed=42)...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_ds = GBCUDataset(os.path.join(NEW_DATA_DIR, "training"), transform=get_transforms(train=True))
    val_ds = GBCUDataset(os.path.join(NEW_DATA_DIR, "validation"), transform=get_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = build_model(pretrained=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
        print(f"[session-check] epoch {epoch+1}/10  loss={running_loss/len(train_ds):.4f}  "
              f"({time.time()-t0:.0f}s elapsed)")

    model.eval()
    y_true, y_pred, y_probs = run_inference(model, val_loader)
    metrics = compute_metrics(y_true, y_pred, y_probs)

    print("\n[session-check] === RESULTS ON SESSION-CORRECTED SPLIT ===")
    print(f"  accuracy   = {metrics['accuracy']:.4f}")
    print(f"  macro-F1   = {metrics['f1_macro']:.4f}")
    print(f"  macro-AUC  = {metrics['auc_macro']:.4f}")
    print("\n  For comparison, paper's reported figures:")
    print("    original contaminated split : accuracy 0.9878")
    print("    pHash-corrected split        : accuracy 0.9305, F1 0.9303, AUC 0.9965")

    report["session_corrected_accuracy"] = metrics["accuracy"]
    report["session_corrected_f1_macro"] = metrics["f1_macro"]
    report["session_corrected_auc_macro"] = metrics["auc_macro"]
    with open(os.path.join(OUT_DIR, "session_leakage_check_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[session-check] updated report written. Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
