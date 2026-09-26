"""
phash_threshold_sensitivity.py
================================
Review 9's ask: "Show how the cluster count, the residual leakage rate,
and the Stage 1 accuracy move as the [pHash] threshold varies over, say,
4, 6, 8, 10, and 12 bits."

WHAT: re-runs the dihedral-aware union-find clustering (same algorithm as
generate_dihedral_cluster_manifest.py) at each of several Hamming-distance
thresholds, using the ALREADY-COMPUTED hash cache
(outputs/phase0/dihedral_hashes_cache.npy) so this is CPU-only and does
not require recomputing any image hashes or retraining any model. It
reports, per threshold: number of clusters (the "effective sample size"),
number of vetoed (suspected chaining) raw components, number of
post-veto mixed-class clusters, and cluster-size distribution.

WHY this design: the expensive step (hashing all 10,692 images at 8
orientations each) was already done once and cached to disk. Re-running
the union-find step at 5 different thresholds against that same cache
takes seconds to low minutes total, not hours, so this does not need a
Kaggle GPU session -- it is a CPU-bound reclustering sweep. It is
included in the Kaggle retrain project only so it can be run in the same
session as the other Kaggle-side reviews (11, 16, 17) for convenience,
not because it needs Kaggle's GPU.

WHAT THIS SCRIPT DOES NOT DO: it does not retrain a classifier at each
threshold and report Stage 1/2 accuracy per threshold, since that would
require up to 5x the full training pipeline. That remains a genuine gap
even after this script runs -- see the printed summary's closing note.
This script closes the "cluster count and false-positive/negative rate"
half of Review 9's ask; the "Stage 1 accuracy per threshold" half is
flagged as still open.

Run from fedgb/ (locally, or on Kaggle with the cache files uploaded):
    python phash_threshold_sensitivity.py
Outputs: outputs/phase0/phash_threshold_sensitivity.json
"""
import json
import os
import time

import numpy as np

from dataset import CLASS_NAMES
from augmentation_aware_leakage_check import popcount64_array

OUT_DIR = "./outputs/phase0"
HASH_CACHE_PATH = os.path.join(OUT_DIR, "dihedral_hashes_cache.npy")
HASH_CACHE_ITEMS_PATH = os.path.join(OUT_DIR, "dihedral_hashes_cache_items.json")

THRESHOLDS_TO_SWEEP = [4, 6, 8, 10, 12]
MAX_CLASSES_PER_CLUSTER = 2
MAX_CLUSTER_SIZE = 30

DATA_DIR = "./data/uidatagb"


def list_images():
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


def apply_chaining_backstop(raw_clusters, items):
    clusters = []
    vetoed = []
    for c in raw_clusters:
        classes_in_c = set(items[i][1] for i in c)
        if len(classes_in_c) > MAX_CLASSES_PER_CLUSTER or len(c) > MAX_CLUSTER_SIZE:
            vetoed.append(c)
            for i in c:
                clusters.append([i])
        else:
            clusters.append(c)
    return clusters, vetoed


def main():
    if not (os.path.exists(HASH_CACHE_PATH) and os.path.exists(HASH_CACHE_ITEMS_PATH)):
        print(f"FATAL: {HASH_CACHE_PATH} not found. This script requires "
              f"the hash cache generate_dihedral_cluster_manifest.py "
              f"already produced. Run that script first (it computes "
              f"hashes once, the expensive step), or upload the cache "
              f"files alongside this script if running on Kaggle.")
        return

    print(f"[sweep] loading cached hashes from {HASH_CACHE_PATH}")
    hashes = np.load(HASH_CACHE_PATH)
    with open(HASH_CACHE_ITEMS_PATH) as f:
        cached_paths = json.load(f)
    items = list_images()
    if cached_paths != [it[0] for it in items]:
        print("[sweep] WARNING: cached item list does not match current "
              "directory listing. Results below describe the CACHED "
              "image set, not necessarily what's on disk right now.")
        # Reconstruct a minimal items list matching the cache's own order,
        # inferring class from each cached path so the sweep can still run.
        items = []
        for p in cached_paths:
            cname = next((c for c in CLASS_NAMES if f"{os.sep}{c}{os.sep}" in p
                          or f"/{c}/" in p), "UNKNOWN")
            items.append((p, cname, None))

    n = len(items)
    print(f"[sweep] {n} images, sweeping thresholds {THRESHOLDS_TO_SWEEP}")

    results = {}
    for thresh in THRESHOLDS_TO_SWEEP:
        t0 = time.time()
        raw_clusters = dihedral_union_find_clusters(hashes, thresh)
        clusters, vetoed = apply_chaining_backstop(raw_clusters, items)

        cluster_class = []
        mixed_count = 0
        for c in clusters:
            classes = [items[i][1] for i in c]
            maj = max(set(classes), key=classes.count)
            cluster_class.append(maj)
            if len(set(classes)) > 1:
                mixed_count += 1

        sizes = sorted((len(c) for c in clusters), reverse=True)
        dt = time.time() - t0
        results[str(thresh)] = {
            "n_raw_connected_components": len(raw_clusters),
            "n_vetoed_raw_components": len(vetoed),
            "n_vetoed_member_images": sum(len(c) for c in vetoed),
            "n_final_clusters": len(clusters),
            "n_mixed_class_clusters_post_veto": mixed_count,
            "max_cluster_size": sizes[0] if sizes else 0,
            "median_cluster_size": sizes[len(sizes) // 2] if sizes else 0,
            "n_singleton_clusters": sum(1 for s in sizes if s == 1),
            "wall_time_seconds": round(dt, 1),
        }
        print(f"[sweep] threshold={thresh:2d}  "
              f"raw_components={len(raw_clusters):5d}  "
              f"vetoed={len(vetoed):3d}  "
              f"final_clusters={len(clusters):5d}  "
              f"mixed_class={mixed_count:3d}  "
              f"max_size={sizes[0] if sizes else 0:3d}  "
              f"({dt:.1f}s)")

    out_path = os.path.join(OUT_DIR, "phash_threshold_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_images": n,
            "thresholds_swept": THRESHOLDS_TO_SWEEP,
            "max_classes_per_cluster_backstop": MAX_CLASSES_PER_CLUSTER,
            "max_cluster_size_backstop": MAX_CLUSTER_SIZE,
            "results_by_threshold": results,
            "note": ("This sweep answers the cluster-count and mixed-class "
                     "(proxy for false-positive) half of Review 9's ask. "
                     "It does NOT retrain a classifier at each threshold "
                     "to report Stage 1/2 accuracy per threshold -- that "
                     "remains a separate, more expensive follow-up "
                     "requiring up to 5 full training runs, one per "
                     "threshold, and is not attempted here."),
        }, f, indent=2)
    print(f"\n[sweep] wrote {out_path}")
    print("[sweep] NOTE: accuracy-per-threshold was NOT computed here; "
          "only cluster-count / mixed-class sensitivity. See the JSON "
          "note field for why.")


if __name__ == "__main__":
    main()
