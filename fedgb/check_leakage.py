"""
check_leakage.py
-----------------
Investigates whether the suspiciously high accuracy (98.78%) on UIdataGB is
real or driven by near-duplicate images (e.g. adjacent video frames from the
same ultrasound exam) split across train and validation by the purely
image-level random split in split_uidatagb.py.

Method: perceptual hash (pHash, 64-bit) every image in training/ and
validation/, then for every validation image find its nearest training
image by Hamming distance. Images from the same short video clip look
near-identical under pHash even with small resize/frame differences, so a
small Hamming distance (<=THRESH) between a val image and a train image is
strong evidence of leakage, not coincidence (random pHash pairs are ~32 bits
apart on average out of 64).

Also re-evaluates the trained model's accuracy on a "clean" validation
subset (val images with no close training-set match) to see whether the
reported 98.78% survives once likely-leaked images are excluded.

Run from fedgb/:  python check_leakage.py
Outputs: outputs/phase0/leakage_check.json, outputs/figures/leakage_hist.png
"""
import json
import os

import imagehash
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import CLASS_NAMES, GBCUDataset, get_transforms, CLASS_TO_IDX
from model import build_model

DATA_DIR = "./data/uidatagb"
CKPT = "./outputs/seeds/seed_42/centralized.pt"
OUT_DIR = "./outputs/phase0"
FIG_DIR = "./outputs/figures"
THRESH = 8  # Hamming distance <=8/64 bits -> near-duplicate (standard pHash rule of thumb)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def list_images(split_dir):
    items = []  # (path, class_name)
    for cname in CLASS_NAMES:
        cdir = os.path.join(split_dir, cname)
        if not os.path.isdir(cdir):
            continue
        for fname in os.listdir(cdir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.join(cdir, fname), cname))
    return items


def compute_hashes(items):
    hashes = np.empty(len(items), dtype=np.uint64)
    for i, (path, _) in enumerate(items):
        h = imagehash.phash(Image.open(path).convert("RGB"))
        hashes[i] = int(str(h), 16)
    return hashes


def hamming_matrix_min(val_hashes, train_hashes):
    """For each val hash, return (min Hamming distance, index of nearest train hash)."""
    # popcount via byte lookup table, vectorized
    popcount_table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def popcount64(x):
        b = x.view(np.uint8).reshape(-1, 8)
        return popcount_table[b].sum(axis=1)

    min_dist = np.empty(len(val_hashes), dtype=np.int32)
    nearest_idx = np.empty(len(val_hashes), dtype=np.int64)
    BATCH = 500
    for start in range(0, len(val_hashes), BATCH):
        chunk = val_hashes[start:start + BATCH]
        xor = chunk[:, None] ^ train_hashes[None, :]  # (batch, n_train)
        dist = popcount64(xor.reshape(-1)).reshape(xor.shape)
        idx = dist.argmin(axis=1)
        min_dist[start:start + BATCH] = dist[np.arange(len(chunk)), idx]
        nearest_idx[start:start + BATCH] = idx
        print(f"[leakage] {min(start+BATCH, len(val_hashes))}/{len(val_hashes)} val images scanned")
    return min_dist, nearest_idx


def main():
    train_items = list_images(os.path.join(DATA_DIR, "training"))
    val_items = list_images(os.path.join(DATA_DIR, "validation"))
    print(f"[leakage] train={len(train_items)} val={len(val_items)}")

    print("[leakage] hashing training images...")
    train_hashes = compute_hashes(train_items)
    print("[leakage] hashing validation images...")
    val_hashes = compute_hashes(val_items)

    min_dist, nearest_idx = hamming_matrix_min(val_hashes, train_hashes)

    leaked_mask = min_dist <= THRESH
    n_leaked = int(leaked_mask.sum())
    print(f"\n[leakage] {n_leaked}/{len(val_items)} validation images "
          f"({100*n_leaked/len(val_items):.1f}%) have a near-duplicate "
          f"(Hamming <= {THRESH}) in the training set")

    # per-class breakdown
    per_class = {}
    for cname in CLASS_NAMES:
        idxs = [i for i, (_, c) in enumerate(val_items) if c == cname]
        if not idxs:
            continue
        leaked = sum(1 for i in idxs if leaked_mask[i])
        per_class[cname] = {"n_val": len(idxs), "n_leaked": leaked,
                            "pct_leaked": round(100 * leaked / len(idxs), 1)}
        print(f"  {cname:55s} {leaked:4d}/{len(idxs):4d} leaked "
              f"({per_class[cname]['pct_leaked']}%)")

    # same-class check: is the leak same-class (expected if true near-dup) or cross-class (hash collision)?
    same_class_leaks = sum(
        1 for i in range(len(val_items))
        if leaked_mask[i] and val_items[i][1] == train_items[int(nearest_idx[i])][1]
    )
    print(f"\n[leakage] of the {n_leaked} leaked images, {same_class_leaks} "
          f"({100*same_class_leaks/max(1,n_leaked):.1f}%) match a same-class "
          f"training image (expected for genuine leakage, not hash collision)")

    # ---- re-evaluate model on clean-only validation subset ----
    print("\n[leakage] re-evaluating primary model on clean (non-leaked) validation subset...")
    model = build_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    full_val = GBCUDataset(os.path.join(DATA_DIR, "validation"),
                           transform=get_transforms(train=False))
    # full_val.samples order must match val_items order (both built by os.listdir
    # per class in CLASS_NAMES order) -- verify alignment defensively
    assert len(full_val.samples) == len(val_items)

    correct_all = correct_clean = 0
    n_clean = 0
    with torch.no_grad():
        for i in range(len(val_items)):
            x, y = full_val[i]
            pred = model(x.unsqueeze(0).to(DEVICE)).argmax(1).item()
            hit = int(pred == y)
            correct_all += hit
            if not leaked_mask[i]:
                correct_clean += hit
                n_clean += 1

    acc_all = correct_all / len(val_items)
    acc_clean = correct_clean / n_clean if n_clean else float("nan")
    print(f"[leakage] accuracy on FULL validation set:  {acc_all:.4f} (n={len(val_items)})")
    print(f"[leakage] accuracy on CLEAN subset only:     {acc_clean:.4f} (n={n_clean})")

    result = {
        "threshold_hamming_bits": THRESH,
        "n_train": len(train_items), "n_val": len(val_items),
        "n_leaked": n_leaked, "pct_leaked": round(100*n_leaked/len(val_items), 2),
        "n_same_class_leaks": same_class_leaks,
        "per_class": per_class,
        "accuracy_full_val": round(acc_all, 4),
        "accuracy_clean_val_only": round(acc_clean, 4),
        "n_clean_val": n_clean,
    }
    with open(os.path.join(OUT_DIR, "leakage_check.json"), "w") as f:
        json.dump(result, f, indent=2)

    # ---- figure: histogram of nearest-train Hamming distances ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(min_dist, bins=range(0, 65, 2), color="#1768c4", edgecolor="white")
    ax.axvline(THRESH, color="#d9601a", linestyle="--",
              label=f"leakage threshold ({THRESH} bits)")
    ax.set_xlabel("Hamming distance to nearest training image (pHash, 64-bit)")
    ax.set_ylabel("Number of validation images")
    ax.set_title("Train/validation near-duplicate check (UIdataGB)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "leakage_hist.png"), dpi=170)
    print(f"\n[leakage] wrote {OUT_DIR}/leakage_check.json and {FIG_DIR}/leakage_hist.png")


if __name__ == "__main__":
    main()
