"""
generate_dihedral_cluster_manifest.py
----------------------------------------
Corrects the gap found by augmentation_aware_leakage_check.py: the existing
dedupe_split_uidatagb.py clusters images using a SINGLE-ORIENTATION pHash,
which misses flip/rotation-augmented duplicates of the same base image
(validated finding: 46% of the corpus -- 4,918 of 10,692 images -- has at
least one cross-split dihedral-orientation duplicate that the old pipeline
could not detect, confirmed against a proper null distribution and a
manually-inspected real example showing a clean group-theoretic match
pattern, not coincidence).

WHAT this script does: re-clusters the FULL corpus (train+val combined)
using dihedral-group-aware pHash matching (8 orientations per image,
Hamming<=8 across all 64 orientation-pair combinations -- same validated
method as the investigative script), then writes a CLUSTER MANIFEST
mapping every image path to a cluster ID.

WHY a manifest file, not just moving files between folders: the Kaggle
retraining script (uidatagb_full_retrain.py) does its OWN fresh 70/20/10
split when it runs. If this script only fixed the local training/
validation folder assignment, that Kaggle-side re-split would ignore the
correction and could scatter a duplicate cluster across train/val/test
again by pure chance. The manifest is uploaded alongside the image data so
the Kaggle script can split at the CLUSTER level instead of the raw
per-image level, which is the only way the correction actually survives
into the retrain.

Only Pass 1 (dihedral pHash) results are used here, not Pass 2 (CNN
embedding). Pass 1 was validated against a proper null distribution
(0/300 false positives at threshold<=8) and a manually-confirmed real
example. Pass 2's threshold (cosine>=0.95) was calibrated more loosely
(known limitation: rotations beyond ~15-20 degrees are not reliably
separable from genuinely different same-class images with a generic
ImageNet backbone), so using it to physically merge clusters risks
over-merging genuinely different images and silently shrinking the
effective dataset for the wrong reason. Pass 2's cross-class matches
(255 pairs) are written out separately for manual review, not
auto-applied.

Run from fedgb/:  python generate_dihedral_cluster_manifest.py
Outputs: outputs/phase0/dihedral_cluster_manifest.json
           {"clusters": {cluster_id: [image_paths...]}, ...}
         outputs/phase0/dihedral_recluster_report.json (summary stats,
           comparable in format to dedupe_split_report.json)
"""
import json
import os
import time

import imagehash
import numpy as np
from PIL import Image

from dataset import CLASS_NAMES
from augmentation_aware_leakage_check import DIHEDRAL_TRANSFORMS, popcount64_array

DATA_DIR = "./data/uidatagb"
OUT_DIR = "./outputs/phase0"
HASH_CACHE_PATH = os.path.join(OUT_DIR, "dihedral_hashes_cache.npy")
HASH_CACHE_ITEMS_PATH = os.path.join(OUT_DIR, "dihedral_hashes_cache_items.json")

# THRESH was tightened from 8 (the original single-orientation pipeline's
# threshold) after a first run at 8 produced a single-linkage CHAINING
# artifact: a 498-image cluster spanning all 9 disease classes, with a
# directly-measured distance of 20 between its most disparate members --
# i.e. those two images are NOT duplicates (20 is within the null/random
# range, mean=23), they were only connected transitively through a long
# chain of intermediate near-threshold matches. The null distribution
# (300 genuinely random pairs) never went below 16, so THRESH=4 keeps a
# wide margin against false positives while making long bridging chains
# much less likely to form, since each additional link in a chain now
# requires a much rarer coincidence.
THRESH = 4

# Even with a tighter threshold, chaining can still theoretically occur.
# As a hard backstop, any resulting cluster that either (a) spans more
# than MAX_CLASSES_PER_CLUSTER distinct class labels, or (b) exceeds
# MAX_CLUSTER_SIZE members, is NOT trusted as one genuine duplicate
# group -- its members are instead each treated as their own singleton
# "cluster" (the conservative direction to err in: this can only ever
# under-merge, i.e. fail to catch some genuine duplicates within that
# flagged group, never falsely force unrelated images together). This
# is a deliberate, disclosed limitation, not a claim of a fully general
# fix to single-linkage chaining -- see the report's
# "vetoed_suspicious_clusters" field for exactly what was excluded and
# why, so it can be manually reviewed later.
MAX_CLASSES_PER_CLUSTER = 2   # real confirmed duplicates were same-class;
                               # allow up to 2 to tolerate the occasional
                               # genuine cross-label near-duplicate found
                               # during spot-checking, without allowing a
                               # 9-class blob through
MAX_CLUSTER_SIZE = 30          # generous vs. the real spot-checked example
                               # cluster sizes (~6-17 members)

os.makedirs(OUT_DIR, exist_ok=True)


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


def compute_dihedral_hashes(items):
    n = len(items)
    hashes = np.empty((n, 8), dtype=np.uint64)
    t0 = time.time()
    for i, (path, _, _) in enumerate(items):
        img = Image.open(path).convert("RGB")
        for k, transform in enumerate(DIHEDRAL_TRANSFORMS):
            h = imagehash.phash(transform(img))
            hashes[i, k] = int(str(h), 16)
        if (i + 1) % 1000 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            print(f"[hash] {i+1}/{n} ({rate:.1f} img/s, "
                  f"ETA {(n-i-1)/rate/60:.1f} min)", flush=True)
    return hashes


def dihedral_union_find_clusters(hashes, thresh, batch=300):
    """Same union-find pattern as the existing dedupe_split_uidatagb.py,
    extended to use the min-over-64-orientation-combos distance instead
    of a single direct XOR, so two images are merged into one cluster if
    ANY of their 8x8 orientation pairs matches within `thresh`."""
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

    t0 = time.time()
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
        dt = time.time() - t0
        print(f"[cluster] scanned {end}/{n} rows ({dt:.0f}s elapsed)", flush=True)

    roots = [find(i) for i in range(n)]
    clusters = {}
    for i, r in enumerate(roots):
        clusters.setdefault(r, []).append(i)
    return list(clusters.values())


def main():
    items = list_images()
    n = len(items)
    print(f"[recluster] {n} images total (train+val combined)")

    if os.path.exists(HASH_CACHE_PATH) and os.path.exists(HASH_CACHE_ITEMS_PATH):
        print(f"[recluster] loading cached hashes from {HASH_CACHE_PATH} "
              f"(delete this file to force recomputation, e.g. if the "
              f"dataset itself changed)")
        hashes = np.load(HASH_CACHE_PATH)
        with open(HASH_CACHE_ITEMS_PATH) as f:
            cached_paths = json.load(f)
        if cached_paths != [it[0] for it in items]:
            print("[recluster] WARNING: cached item list does not match "
                  "current directory listing -- recomputing hashes fresh.")
            hashes = compute_dihedral_hashes(items)
            np.save(HASH_CACHE_PATH, hashes)
            with open(HASH_CACHE_ITEMS_PATH, "w") as f:
                json.dump([it[0] for it in items], f)
    else:
        hashes = compute_dihedral_hashes(items)
        np.save(HASH_CACHE_PATH, hashes)
        with open(HASH_CACHE_ITEMS_PATH, "w") as f:
            json.dump([it[0] for it in items], f)
        print(f"[recluster] cached hashes to {HASH_CACHE_PATH} for reuse "
              f"if the threshold needs retuning again")

    raw_clusters = dihedral_union_find_clusters(hashes, THRESH)
    print(f"\n[recluster] {n} raw images -> {len(raw_clusters)} raw "
          f"connected components at THRESH={THRESH}")

    # --- chaining backstop: veto any cluster spanning too many classes or
    # too many members, splitting its members into singletons instead of
    # trusting a possibly-chained blob. See MAX_CLASSES_PER_CLUSTER /
    # MAX_CLUSTER_SIZE definitions above for the rationale. ---
    clusters = []
    vetoed = []
    for c in raw_clusters:
        classes_in_c = set(items[i][1] for i in c)
        if len(classes_in_c) > MAX_CLASSES_PER_CLUSTER or len(c) > MAX_CLUSTER_SIZE:
            vetoed.append({
                "n_members": len(c),
                "n_distinct_classes": len(classes_in_c),
                "classes": sorted(classes_in_c),
                "member_paths": [items[i][0] for i in c],
                "reason": ("exceeds MAX_CLASSES_PER_CLUSTER"
                           if len(classes_in_c) > MAX_CLASSES_PER_CLUSTER
                           else "exceeds MAX_CLUSTER_SIZE"),
            })
            for i in c:
                clusters.append([i])  # each member becomes its own singleton
        else:
            clusters.append(c)

    if vetoed:
        print(f"[recluster] VETOED {len(vetoed)} suspicious cluster(s) as likely "
              f"single-linkage chaining artifacts (spanning too many classes "
              f"or too large) -- their {sum(v['n_members'] for v in vetoed)} "
              f"member images were split back into singletons rather than "
              f"trusted as one duplicate group. See "
              f"dihedral_vetoed_clusters.json for full detail.")
        with open(os.path.join(OUT_DIR, "dihedral_vetoed_clusters.json"), "w") as f:
            json.dump(vetoed, f, indent=2)

    print(f"[recluster] {n} raw images -> {len(clusters)} dihedral-aware "
          f"near-duplicate clusters after the chaining backstop "
          f"(TRUE effective sample size)")

    cluster_sizes = sorted((len(c) for c in clusters), reverse=True)
    print(f"[recluster] cluster size distribution: max={cluster_sizes[0]}, "
          f"median={cluster_sizes[len(cluster_sizes)//2]}, "
          f"singletons={sum(1 for s in cluster_sizes if s == 1)}")

    # majority class per cluster + mixed-class flag (same convention as
    # the original dedupe_split_uidatagb.py, so results are comparable).
    # Note: by construction, no cluster here can span more than
    # MAX_CLASSES_PER_CLUSTER distinct labels -- anything worse was
    # already vetoed above.
    cluster_class = []
    mixed_clusters = []
    for cidx, c in enumerate(clusters):
        classes = [items[i][1] for i in c]
        maj = max(set(classes), key=classes.count)
        cluster_class.append(maj)
        if len(set(classes)) > 1:
            mixed_clusters.append({
                "cluster_id": cidx,
                "majority_class": maj,
                "member_classes": classes,
                "member_paths": [items[i][0] for i in c],
            })
    print(f"[recluster] {len(mixed_clusters)} (post-veto) clusters still "
          f"contain more than one class label (written to report for "
          f"manual review)")

    # cluster manifest: path -> cluster_id, and cluster_id -> [paths]
    path_to_cluster = {}
    cluster_to_paths = {}
    for cidx, member_idxs in enumerate(clusters):
        paths = [items[i][0] for i in member_idxs]
        cluster_to_paths[str(cidx)] = paths
        for p in paths:
            path_to_cluster[p] = cidx

    manifest = {
        "threshold_hamming_bits": THRESH,
        "max_classes_per_cluster": MAX_CLASSES_PER_CLUSTER,
        "max_cluster_size": MAX_CLUSTER_SIZE,
        "method": "dihedral-group-aware pHash (8 orientations x 8 orientations, "
                   "min Hamming distance across all 64 combinations), with a "
                   "chaining backstop that vetoes and un-merges any raw "
                   "connected component spanning too many classes or too "
                   "many members",
        "n_raw_images": n,
        "n_clusters": len(clusters),
        "n_vetoed_raw_clusters": len(vetoed),
        "path_to_cluster_id": path_to_cluster,
        "cluster_id_to_paths": cluster_to_paths,
        "cluster_id_to_majority_class": {str(i): c for i, c in enumerate(cluster_class)},
    }
    with open(os.path.join(OUT_DIR, "dihedral_cluster_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    report = {
        "threshold_hamming_bits": THRESH,
        "n_raw_images": n,
        "n_raw_connected_components_before_veto": len(raw_clusters),
        "n_vetoed_suspicious_clusters": len(vetoed),
        "n_clusters_true_effective_n": len(clusters),
        "n_mixed_class_clusters_post_veto": len(mixed_clusters),
        "cluster_size_max": cluster_sizes[0],
        "cluster_size_median": cluster_sizes[len(cluster_sizes)//2],
        "n_singleton_clusters": sum(1 for s in cluster_sizes if s == 1),
        "comparison_to_old_single_orientation_clustering": (
            "The old dedupe_split_uidatagb.py found 2,993 clusters from the "
            "same 10,692 raw images using single-orientation pHash. This "
            "script's dihedral-aware clustering should find FEWER, LARGER "
            "clusters than that (since it additionally merges flip/rotation "
            "duplicates the old method could not see), but NOT as extreme "
            "as the first (THRESH=8, no veto) attempt, which produced a "
            "498-image cluster spanning all 9 classes -- confirmed via "
            "direct measurement to be a single-linkage chaining artifact, "
            "not real duplicates (directly-measured distance 20 between "
            "its two most disparate members, within the null/random range) "
            "-- compare n_clusters_true_effective_n above to the old figure."
        ),
    }
    with open(os.path.join(OUT_DIR, "dihedral_recluster_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(OUT_DIR, "dihedral_mixed_class_clusters.json"), "w") as f:
        json.dump(mixed_clusters, f, indent=2)

    print(f"\n[recluster] wrote:")
    print(f"  {OUT_DIR}/dihedral_cluster_manifest.json  (upload this to Kaggle "
          f"alongside the image data)")
    print(f"  {OUT_DIR}/dihedral_recluster_report.json")
    print(f"  {OUT_DIR}/dihedral_mixed_class_clusters.json  (manual review needed)")
    print("[recluster] THIS SCRIPT DID NOT MOVE ANY FILES -- the manifest is "
          "consumed by uidatagb_full_retrain.py's splitting logic instead of "
          "physically reorganising the local training/validation folders.")


if __name__ == "__main__":
    main()
