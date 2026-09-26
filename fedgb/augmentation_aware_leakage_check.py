"""
augmentation_aware_leakage_check.py
-------------------------------------
Investigates a gap in dedupe_split_uidatagb.py: that script computes a
SINGLE-ORIENTATION perceptual hash (imagehash.phash) per image and clusters
on Hamming distance <=8. Standard DCT-based pHash is NOT rotation- or
flip-invariant, so if the dataset's own stated augmentation (Turki et al.,
Data in Brief 2024: "random rotations, flips, and translations, as well as
changes in brightness, contrast, and saturation") produced a flipped or
rotated copy of the same base image, the existing pipeline would NOT flag
it as a near-duplicate -- meaning it could still be split across train/
validation undetected, even after the existing Stage 1 correction.

This is an INVESTIGATIVE script only (matches the project's existing
check_leakage.py -> dedupe_split_uidatagb.py two-step discipline: measure
first, then decide whether/how to correct). It does not move any files.

Two independent detection passes, since they catch different transform
types:

  Pass 1 -- Dihedral-group-aware pHash (catches EXACT 90/180/270-degree
  rotations and horizontal flips). For each image, compute pHash at all
  8 dihedral-group orientations (identity, hflip, rot90, rot90+hflip,
  rot180, rot180+hflip, rot270, rot270+hflip). Two images A and B are
  flagged as a dihedral duplicate if ANY of A's 8 hashes is within
  Hamming<=8 of ANY of B's 8 hashes. (Group-theory justification: if
  B = f(A) for some dihedral transform f, then since the dihedral group
  is closed under composition, {g(B) : g in group} = {g(f(A)) : g in
  group} = {h(A) : h in group} -- so the two images' full 8-hash sets
  should overlap almost exactly, not just their single default-
  orientation hash.)

  Pass 2 -- CNN-embedding cosine-similarity (catches arbitrary-angle
  rotations, translations, and brightness/contrast/saturation changes
  that Pass 1's fixed 8 orientations cannot). Uses ImageNet-pretrained
  ResNet-50 average-pooled features (2048-d), which are empirically far
  more robust to these transforms than raw-pixel perceptual hashing,
  since convolutional features already have some built-in local
  translation/rotation tolerance from pooling. Flags pairs above a
  cosine-similarity threshold as candidate augmentation-duplicates.

Run from fedgb/:  python augmentation_aware_leakage_check.py
Outputs: outputs/phase0/augmentation_aware_leakage_report.json
         outputs/phase0/augmentation_aware_candidate_pairs.json (raw pairs,
         for manual spot-checking before any correction is applied)
"""
import json
import os
import time

import imagehash
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from dataset import CLASS_NAMES

DATA_DIR = "./data/uidatagb"
OUT_DIR = "./outputs/phase0"
os.makedirs(OUT_DIR, exist_ok=True)

PHASH_THRESH = 8            # same threshold as the existing single-orientation check

# COSINE_SIM_THRESH was empirically calibrated (not guessed) against this
# corpus before the full run: on a real sample image, hflip=0.975,
# 5-deg rotation=0.966, 12-deg rotation=0.934, 25-deg rotation=0.894,
# brightness/contrast jitter=0.993, 10px translate=0.959 -- versus
# DIFFERENT real images of the SAME class at 0.872-0.908 (negative
# control). 0.95 sits clearly above the negative-control ceiling
# (0.908) while still catching flip/small-rotation/translation/
# brightness-jitter duplicates. KNOWN LIMITATION: rotations beyond
# roughly 15-20 degrees are NOT reliably separable from genuinely
# different same-class images using a generic ImageNet-pretrained
# backbone -- this pass will under-catch large-angle rotated
# duplicates, and that gap should be stated explicitly wherever these
# results are reported, not silently accepted as complete coverage.
COSINE_SIM_THRESH = 0.95
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ---------------------------------------------------------------------------
# Pass 1: dihedral-group-aware pHash
# ---------------------------------------------------------------------------

DIHEDRAL_TRANSFORMS = [
    lambda im: im,
    lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    lambda im: im.transpose(Image.ROTATE_90),
    lambda im: im.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT),
    lambda im: im.transpose(Image.ROTATE_180),
    lambda im: im.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT),
    lambda im: im.transpose(Image.ROTATE_270),
    lambda im: im.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT),
]


def compute_dihedral_hashes(items):
    """Returns an (n_items, 8) uint64 array: each row is one image's 8
    dihedral-orientation pHashes."""
    n = len(items)
    hashes = np.empty((n, 8), dtype=np.uint64)
    t0 = time.time()
    for i, (path, _, _) in enumerate(items):
        img = Image.open(path).convert("RGB")
        for k, transform in enumerate(DIHEDRAL_TRANSFORMS):
            h = imagehash.phash(transform(img))
            hashes[i, k] = int(str(h), 16)
        if (i + 1) % 500 == 0:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (n - i - 1) / rate
            print(f"[dihedral-hash] {i+1}/{n} ({rate:.1f} img/s, "
                  f"ETA {eta/60:.1f} min)", flush=True)
    return hashes


def popcount64_array(x):
    """Vectorised popcount for an array of uint64 values."""
    popcount_table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    b = x.view(np.uint8).reshape(x.shape + (8,))
    return popcount_table[b].sum(axis=-1)


def find_dihedral_duplicate_pairs(hashes, thresh, batch=300):
    """hashes: (n, 8) array. Flags pair (i, j), i<j, if min over all 8x8=64
    orientation combinations of Hamming distance is <= thresh."""
    n = hashes.shape[0]
    pairs = []
    t0 = time.time()
    for start in range(0, n, batch):
        end = min(start + batch, n)
        chunk = hashes[start:end]                       # (b, 8)
        rest = hashes[start:]                            # (m, 8) upper triangle only
        # xor every (chunk_orientation, rest_orientation) combo:
        # chunk[:, None, :, None] (b,1,8,1) ^ rest[None, :, None, :] (1,m,1,8)
        # -> (b, m, 8, 8)
        xor = chunk[:, None, :, None] ^ rest[None, :, None, :]
        dist = popcount64_array(xor)                      # (b, m, 8, 8)
        min_dist = dist.min(axis=(2, 3))                   # (b, m) best orientation match
        close_i, close_j = np.where(min_dist <= thresh)
        for ii, jj in zip(close_i, close_j):
            gi, gj = start + ii, start + jj
            if gi != gj:
                pairs.append((int(gi), int(gj), int(min_dist[ii, jj])))
        dt = time.time() - t0
        print(f"[dihedral-match] scanned {end}/{n} rows ({dt:.0f}s elapsed)", flush=True)
    # dedupe (i,j)/(j,i) duplicates from the upper-triangle scan
    seen = set()
    unique_pairs = []
    for i, j, d in pairs:
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            unique_pairs.append((key[0], key[1], d))
    return unique_pairs


# ---------------------------------------------------------------------------
# Pass 2: CNN-embedding cosine similarity
# ---------------------------------------------------------------------------

def build_embedding_extractor():
    """ImageNet-pretrained ResNet-50 with the classification head removed,
    used purely as a fixed feature extractor -- NOT the paper's trained
    disease classifier, so this check is independent of anything learned
    from (potentially leaky) UIdataGB training."""
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    backbone = models.resnet50(weights=weights)
    backbone.fc = nn.Identity()  # output the 2048-d pooled feature directly
    backbone.eval().to(DEVICE)
    return backbone


EMBED_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_embeddings(items, model, batch_size=64):
    n = len(items)
    embeddings = np.empty((n, 2048), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_imgs = []
            for path, _, _ in items[start:end]:
                img = Image.open(path).convert("RGB")
                batch_imgs.append(EMBED_TRANSFORM(img))
            x = torch.stack(batch_imgs).to(DEVICE)
            feat = model(x).cpu().numpy()
            embeddings[start:end] = feat
            if (end % 1000) < batch_size:
                dt = time.time() - t0
                rate = end / dt
                eta = (n - end) / max(rate, 1e-6)
                print(f"[embed] {end}/{n} ({rate:.1f} img/s, ETA {eta/60:.1f} min)",
                      flush=True)
    # L2-normalise so dot product == cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def find_embedding_duplicate_pairs(norm_embeddings, thresh, batch=500):
    n = norm_embeddings.shape[0]
    pairs = []
    emb_t = torch.from_numpy(norm_embeddings).to(DEVICE)
    t0 = time.time()
    for start in range(0, n, batch):
        end = min(start + batch, n)
        chunk = emb_t[start:end]                # (b, 2048)
        rest = emb_t[start:]                     # (m, 2048) upper triangle only
        sims = (chunk @ rest.T).cpu().numpy()     # (b, m)
        close_i, close_j = np.where(sims >= thresh)
        for ii, jj in zip(close_i, close_j):
            gi, gj = start + ii, start + jj
            if gi != gj:
                pairs.append((int(gi), int(gj), float(sims[ii, jj])))
        dt = time.time() - t0
        print(f"[embed-match] scanned {end}/{n} rows ({dt:.0f}s elapsed)", flush=True)
    seen = set()
    unique_pairs = []
    for i, j, s in pairs:
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            unique_pairs.append((key[0], key[1], s))
    return unique_pairs


# ---------------------------------------------------------------------------
# Main: run both passes, report how many NEW cross-split pairs each finds
# beyond what the existing single-orientation pHash already caught.
# ---------------------------------------------------------------------------

def main():
    print(f"[augmentation-aware-check] device: {DEVICE}")
    items = list_images()
    n = len(items)
    print(f"[augmentation-aware-check] {n} images total")

    # --- Pass 1: dihedral pHash ---
    print("\n=== PASS 1: dihedral-group-aware pHash ===")
    dihedral_hashes = compute_dihedral_hashes(items)
    dihedral_pairs = find_dihedral_duplicate_pairs(dihedral_hashes, PHASH_THRESH)
    print(f"[pass 1] {len(dihedral_pairs)} candidate pairs found")

    dihedral_cross_split = [
        (i, j, d) for i, j, d in dihedral_pairs
        if items[i][2] != items[j][2]
    ]
    dihedral_cross_class = [
        (i, j, d) for i, j, d in dihedral_pairs
        if items[i][1] != items[j][1]
    ]
    print(f"[pass 1] {len(dihedral_cross_split)} of those CROSS the train/val split "
          f"(potential undetected leakage)")
    print(f"[pass 1] {len(dihedral_cross_class)} of those span DIFFERENT class labels "
          f"(worth manual spot-check -- could be hashing false positives OR "
          f"genuinely mislabeled/ambiguous cases)")

    # --- Pass 2: CNN embedding ---
    print("\n=== PASS 2: CNN-embedding cosine similarity ===")
    model = build_embedding_extractor()
    embeddings = compute_embeddings(items, model)
    embedding_pairs = find_embedding_duplicate_pairs(embeddings, COSINE_SIM_THRESH)
    print(f"[pass 2] {len(embedding_pairs)} candidate pairs found "
          f"(cosine >= {COSINE_SIM_THRESH})")

    embedding_cross_split = [
        (i, j, s) for i, j, s in embedding_pairs
        if items[i][2] != items[j][2]
    ]
    embedding_cross_class = [
        (i, j, s) for i, j, s in embedding_pairs
        if items[i][1] != items[j][1]
    ]
    print(f"[pass 2] {len(embedding_cross_split)} of those CROSS the train/val split")
    print(f"[pass 2] {len(embedding_cross_class)} of those span DIFFERENT class labels")

    # --- union: images cross-split-flagged by EITHER pass, not already caught ---
    already_flagged_path = os.path.join(OUT_DIR, "dedupe_split_report.json")
    prior_note = ("Prior single-orientation pHash pipeline already clustered the "
                   "full corpus at Hamming<=8 and re-split at the cluster level "
                   "(see dedupe_split_report.json), so ANY cross-split pair found "
                   "here that single-orientation pHash also would have caught is "
                   "not actually 'new' leakage -- it is evidence the CURRENT data "
                   "already reflects that correction. What matters is whether "
                   "dihedral/embedding-only matches (pairs NOT within Hamming<=8 "
                   "at single orientation) still cross the split.")

    report = {
        "n_images": n,
        "note_on_relationship_to_prior_correction": prior_note,
        "pass_1_dihedral_phash": {
            "threshold_hamming_bits": PHASH_THRESH,
            "n_candidate_pairs": len(dihedral_pairs),
            "n_cross_split_pairs": len(dihedral_cross_split),
            "n_cross_class_pairs": len(dihedral_cross_class),
        },
        "pass_2_cnn_embedding": {
            "cosine_similarity_threshold": COSINE_SIM_THRESH,
            "n_candidate_pairs": len(embedding_pairs),
            "n_cross_split_pairs": len(embedding_cross_split),
            "n_cross_class_pairs": len(embedding_cross_class),
        },
    }
    with open(os.path.join(OUT_DIR, "augmentation_aware_leakage_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # save raw pairs (with file paths, not just indices) for manual inspection
    def pairs_to_named(pairs, items):
        return [
            {
                "image_a": items[i][0], "class_a": items[i][1], "split_a": items[i][2],
                "image_b": items[j][0], "class_b": items[j][1], "split_b": items[j][2],
                "score": score,
            }
            for i, j, score in pairs
        ]

    with open(os.path.join(OUT_DIR, "augmentation_aware_candidate_pairs.json"), "w") as f:
        json.dump({
            "dihedral_cross_split_pairs": pairs_to_named(dihedral_cross_split, items),
            "embedding_cross_split_pairs": pairs_to_named(embedding_cross_split, items),
        }, f, indent=2)

    print(f"\n[augmentation-aware-check] wrote "
          f"{OUT_DIR}/augmentation_aware_leakage_report.json and "
          f"{OUT_DIR}/augmentation_aware_candidate_pairs.json")
    print("[augmentation-aware-check] THIS SCRIPT DID NOT MOVE ANY FILES. "
          "Review the candidate pairs before deciding whether/how to correct.")


if __name__ == "__main__":
    main()
