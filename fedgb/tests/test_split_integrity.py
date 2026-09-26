"""
tests/test_split_integrity.py
-------------------------------
Phase 11 regression tests: verifies the on-disk Stage 1 and Stage 2 splits
still have the properties the manuscript reports, so a future data change
(re-running a script, editing a file by hand) can't silently reintroduce
leakage without a test failing.

Checks:
  1. Stage 1 split (data/uidatagb/): no pHash near-duplicate (Hamming<=8)
     crosses training/validation. This re-derives the check from
     check_leakage.py's own logic rather than trusting its cached JSON.
  2. Stage 2 split (data/uidatagb_sessioncorrected/): no filename-prefix
     group crosses training/validation.

Run from fedgb/:  python -m pytest tests/ -v
(or: python tests/test_split_integrity.py, for a standalone run without pytest)
"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dataset import CLASS_NAMES  # noqa: E402

STAGE1_DIR = os.path.join(ROOT, "data", "uidatagb")
STAGE2_DIR = os.path.join(ROOT, "data", "uidatagb_sessioncorrected")
FNAME_RE = re.compile(r'^([A-Za-z]+\d+)\s*\(\d+\)\.\w+$')
PHASH_THRESH = 8


def list_images(data_dir):
    items = []
    for split in ("training", "validation"):
        for cname in CLASS_NAMES:
            cdir = os.path.join(data_dir, split, cname)
            if not os.path.isdir(cdir):
                continue
            for fname in os.listdir(cdir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    items.append((fname, split))
    return items


def session_key(fname):
    m = FNAME_RE.match(fname)
    return m.group(1).lower() if m else None


def test_stage2_no_group_crosses_split():
    if not os.path.isdir(STAGE2_DIR):
        print(f"[SKIP] {STAGE2_DIR} not present")
        return
    items = list_images(STAGE2_DIR)
    assert items, f"no images found under {STAGE2_DIR}"

    splits_by_group = {}
    for fname, split in items:
        key = session_key(fname)
        if key is None:
            continue  # unmatched filenames are not part of this invariant
        splits_by_group.setdefault(key, set()).add(split)

    crossing = {k: v for k, v in splits_by_group.items() if len(v) > 1}
    assert not crossing, (
        f"{len(crossing)} filename-prefix groups cross train/val in "
        f"{STAGE2_DIR} (Stage 2 split integrity violated): "
        f"{list(crossing)[:10]}..."
    )
    print(f"[PASS] Stage 2: {len(splits_by_group)} groups, "
          f"0 crossing train/val ({len(items)} images checked)")


def test_stage1_no_near_duplicate_crosses_split():
    if not os.path.isdir(STAGE1_DIR):
        print(f"[SKIP] {STAGE1_DIR} not present")
        return
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        print("[SKIP] imagehash/PIL not installed")
        return

    paths_splits = []
    for split in ("training", "validation"):
        for cname in CLASS_NAMES:
            cdir = os.path.join(STAGE1_DIR, split, cname)
            if not os.path.isdir(cdir):
                continue
            for fname in os.listdir(cdir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths_splits.append((os.path.join(cdir, fname), split))

    assert paths_splits, f"no images found under {STAGE1_DIR}"

    hashes = np.empty(len(paths_splits), dtype=np.uint64)
    for i, (path, _) in enumerate(paths_splits):
        h = imagehash.phash(Image.open(path).convert("RGB"))
        hashes[i] = int(str(h), 16)

    popcount_table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def popcount64(x):
        b = x.view(np.uint8).reshape(-1, 8)
        return popcount_table[b].sum(axis=1)

    train_idx = [i for i, (_, s) in enumerate(paths_splits) if s == "training"]
    val_idx = [i for i, (_, s) in enumerate(paths_splits) if s == "validation"]
    train_hashes = hashes[train_idx]
    val_hashes = hashes[val_idx]

    n_crossing = 0
    BATCH = 500
    for start in range(0, len(val_hashes), BATCH):
        chunk = val_hashes[start:start + BATCH]
        xor = chunk[:, None] ^ train_hashes[None, :]
        dist = popcount64(xor.reshape(-1)).reshape(xor.shape)
        n_crossing += int((dist.min(axis=1) <= PHASH_THRESH).sum())

    assert n_crossing == 0, (
        f"{n_crossing}/{len(val_hashes)} validation images in {STAGE1_DIR} "
        f"have a near-duplicate (Hamming<={PHASH_THRESH}) still in training "
        f"(Stage 1 split integrity violated)"
    )
    print(f"[PASS] Stage 1: 0/{len(val_hashes)} validation images have a "
          f"near-duplicate in training (threshold={PHASH_THRESH})")


if __name__ == "__main__":
    test_stage2_no_group_crosses_split()
    test_stage1_no_near_duplicate_crosses_split()
    print("\nAll split-integrity checks passed.")
