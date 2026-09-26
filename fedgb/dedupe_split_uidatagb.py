"""
dedupe_split_uidatagb.py
-------------------------
Fixes the leakage found by check_leakage.py: 93.2% of validation images had
a near-duplicate (pHash Hamming <= 8) in training, almost certainly from
video-frame extraction with no patient/exam ID to split on properly.

This clusters ALL images (train+val combined) into near-duplicate groups via
union-find on the same Hamming-distance graph, then does a fresh 70/30 split
at the CLUSTER level (never splitting one cluster across train/val) --
the connected-components clustering stands in for the missing exam/patient
ID: images that are near-duplicates of each other are, for splitting
purposes, treated as "the same exam" regardless of which class folder they
originally landed in.

Physically moves files between training/ and validation/ to match the new
assignment (starting from the current, already-split state).

Run from fedgb/:  python dedupe_split_uidatagb.py
Outputs: outputs/phase0/dedupe_split_report.json
"""
import json
import os
import random

import imagehash
import numpy as np
from PIL import Image

from dataset import CLASS_NAMES

DATA_DIR = "./data/uidatagb"
OUT_DIR = "./outputs/phase0"
THRESH = 8
TRAIN_FRAC = 0.70
SEED = 42
os.makedirs(OUT_DIR, exist_ok=True)


def list_images():
    """Returns [(path, class_name, current_split)] for every image currently
    in training/ or validation/."""
    items = []
    for split in ("training", "validation"):
        for cname in CLASS_NAMES:
            cdir = os.path.join(DATA_DIR, split, cname)
            if not os.path.isdir(cdir):
                continue
            for fname in os.listdir(cdir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    items.append((os.path.join(cdir, fname), cname, split))
    return items


def compute_hashes(items):
    hashes = np.empty(len(items), dtype=np.uint64)
    for i, (path, _, _) in enumerate(items):
        h = imagehash.phash(Image.open(path).convert("RGB"))
        hashes[i] = int(str(h), 16)
        if (i + 1) % 1000 == 0:
            print(f"[dedupe] hashed {i+1}/{len(items)}")
    return hashes


def union_find_clusters(hashes, thresh):
    n = len(hashes)
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

    popcount_table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def popcount64(x):
        b = x.view(np.uint8).reshape(-1, 8)
        return popcount_table[b].sum(axis=1)

    BATCH = 500
    for start in range(0, n, BATCH):
        chunk = hashes[start:start + BATCH]
        # compare this chunk against all hashes from `start` onward (upper triangle)
        rest = hashes[start:]
        xor = chunk[:, None] ^ rest[None, :]
        dist = popcount64(xor.reshape(-1)).reshape(xor.shape)
        close_i, close_j = np.where(dist <= thresh)
        for i, j in zip(close_i, close_j):
            gi, gj = start + i, start + j
            if gi != gj:
                union(gi, gj)
        print(f"[dedupe] clustering {min(start+BATCH, n)}/{n}")

    roots = [find(i) for i in range(n)]
    clusters = {}
    for i, r in enumerate(roots):
        clusters.setdefault(r, []).append(i)
    return list(clusters.values())  # list of lists of item-indices


def main():
    items = list_images()
    print(f"[dedupe] total images (train+val): {len(items)}")

    hashes = compute_hashes(items)
    clusters = union_find_clusters(hashes, THRESH)
    print(f"\n[dedupe] {len(items)} raw images -> {len(clusters)} near-duplicate "
          f"clusters (Hamming<={THRESH}) -- this is the TRUE effective sample size")

    cluster_sizes = sorted((len(c) for c in clusters), reverse=True)
    print(f"[dedupe] cluster size distribution: max={cluster_sizes[0]}, "
          f"median={cluster_sizes[len(cluster_sizes)//2]}, "
          f"singletons={sum(1 for s in cluster_sizes if s==1)}")

    # majority class per cluster (should be near-unanimous given same-class leak rate)
    cluster_class = []
    mixed_clusters = 0
    for c in clusters:
        classes = [items[i][1] for i in c]
        maj = max(set(classes), key=classes.count)
        if len(set(classes)) > 1:
            mixed_clusters += 1
        cluster_class.append(maj)
    print(f"[dedupe] {mixed_clusters} clusters contain more than one class label "
          f"(ambiguous; assigned to majority class)")

    # stratified cluster-level 70/30 split
    rng = random.Random(SEED)
    by_class = {}
    for idx, cname in enumerate(cluster_class):
        by_class.setdefault(cname, []).append(idx)

    train_clusters, val_clusters = set(), set()
    for cname, cidxs in by_class.items():
        rng.shuffle(cidxs)
        n_train = round(len(cidxs) * TRAIN_FRAC)
        train_clusters.update(cidxs[:n_train])
        val_clusters.update(cidxs[n_train:])

    # assign every image to its cluster's split
    new_split = [None] * len(items)
    for cidx, member_idxs in enumerate(clusters):
        target = "training" if cidx in train_clusters else "validation"
        for i in member_idxs:
            new_split[i] = target

    # move files that changed split
    moved = 0
    final_counts = {"training": {c: 0 for c in CLASS_NAMES},
                    "validation": {c: 0 for c in CLASS_NAMES}}
    for (path, cname, cur_split), target in zip(items, new_split):
        final_counts[target][cname] += 1
        if target != cur_split:
            new_dir = os.path.join(DATA_DIR, target, cname)
            os.makedirs(new_dir, exist_ok=True)
            new_path = os.path.join(new_dir, os.path.basename(path))
            if os.path.exists(new_path):
                base, ext = os.path.splitext(os.path.basename(path))
                new_path = os.path.join(new_dir, f"{base}_dup{ext}")
            os.replace(path, new_path)
            moved += 1

    print(f"\n[dedupe] moved {moved} files to their new cluster-consistent split")
    n_train_final = sum(final_counts["training"].values())
    n_val_final = sum(final_counts["validation"].values())
    print(f"[dedupe] final split: train={n_train_final} val={n_val_final} "
          f"({100*n_train_final/(n_train_final+n_val_final):.1f}% / "
          f"{100*n_val_final/(n_train_final+n_val_final):.1f}%)")
    for cname in CLASS_NAMES:
        print(f"  {cname:55s} train={final_counts['training'][cname]:5d}  "
              f"val={final_counts['validation'][cname]:5d}")

    report = {
        "threshold_hamming_bits": THRESH,
        "n_raw_images": len(items),
        "n_clusters_true_effective_n": len(clusters),
        "n_mixed_class_clusters": mixed_clusters,
        "cluster_size_max": cluster_sizes[0],
        "cluster_size_median": cluster_sizes[len(cluster_sizes)//2],
        "n_singleton_clusters": sum(1 for s in cluster_sizes if s == 1),
        "files_moved": moved,
        "final_train_total": n_train_final,
        "final_val_total": n_val_final,
        "final_counts": final_counts,
    }
    with open(os.path.join(OUT_DIR, "dedupe_split_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[dedupe] wrote {OUT_DIR}/dedupe_split_report.json")


if __name__ == "__main__":
    main()
