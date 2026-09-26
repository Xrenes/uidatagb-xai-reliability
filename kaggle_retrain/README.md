# Full retrain on Kaggle GPU — setup steps

This folder fixes several problems found across two rounds of review
(missing test set, the mislabeled "replicate" models, no training curves,
no null control for the consistency metric) plus a newly-discovered
dataset-leakage gap: the dataset's own source paper (Turki et al., *Data
in Brief* 2024) confirms images were augmented via random rotations,
flips, translations, and brightness/contrast/saturation jitter, but the
existing de-duplication pipeline only hashed each image at its single
stored orientation. A dihedral-aware re-check
(`augmentation_aware_leakage_check.py`) found that **46% of the
corpus (4,918 of 10,692 images)** has at least one cross-split duplicate
the old pipeline could not detect — confirmed against a proper null
distribution (0/300 false positives on genuinely random pairs) and a
manually-inspected real example showing a clean group-theoretic match
pattern, not coincidence. See the top of `uidatagb_full_retrain.py` and
`fedgb/generate_dihedral_cluster_manifest.py` for the full rationale.

## What changed vs. the old local pipeline

| Old | New |
|---|---|
| 70/30 train/validation, no held-out test | 70/20/10 train/validation/test, test touched exactly once at the end |
| 3 "replicates" = 3 disjoint data subsets + different seeds | 3 replicates = SAME full training set, different seed only |
| Only final-epoch metrics saved | Full per-epoch train/val loss + accuracy saved to `history.json` |
| No null control for the consistency metric | Untrained-model control + random-heatmap control, both included |
| 10 fixed epochs, no early stopping | Up to 20 epochs, early stopping on validation macro-F1 |
| Duplicate clustering by single-orientation pHash only | Duplicate clustering by dihedral-aware pHash (8 orientations), split at the cluster level via an uploaded manifest, not per-image |

## Step 1 — Zip the corrected dataset

The corrected (pHash-deduplicated) data already lives at
`fedgb/data/uidatagb/` locally. Zip just that folder — do not re-run
deduplication, this data is already correct:

```
cd "fedgb/data"
zip -r uidatagb_corrected.zip uidatagb/
```

(or right-click → "Send to → Compressed folder" in Windows Explorer on
the `uidatagb` folder). This produces roughly a 1.5 GB zip.

## Step 2 — Generate the cluster manifest (do this before uploading)

Run, from `fedgb/`:

```
python generate_dihedral_cluster_manifest.py
```

This produces `fedgb/outputs/phase0/dihedral_cluster_manifest.json` — a
mapping from every image path to a duplicate-cluster ID. It does **not**
move any files; it only records which images are duplicates of each
other so the Kaggle-side split can keep every duplicate cluster together
on one side of train/val/test. Skipping this step means the Kaggle
script falls back to a per-image split with no duplicate protection
(it will warn loudly in the log if this happens).

## Step 3 — Upload as two Kaggle Datasets

**Dataset 1 — the images** (same as before):
1. Go to kaggle.com → **Datasets** → **New Dataset**.
2. Upload `uidatagb_corrected.zip`.
3. Name it `uidatagb-corrected`, set to **Private**, **Create**.

**Dataset 2 — the cluster manifest** (new):
1. **New Dataset** again.
2. Upload just `dihedral_cluster_manifest.json` from
   `fedgb/outputs/phase0/`.
3. Name it `uidatagb-cluster-manifest`, set to **Private**, **Create**.

(Using two datasets, rather than bundling the manifest into the image
zip, means you can regenerate/re-upload the manifest alone later
without re-uploading 1.5 GB of images.)

## Step 4 — Create the notebook

1. **New Notebook** on Kaggle.
2. **Add Input** → attach BOTH `uidatagb-corrected` and
   `uidatagb-cluster-manifest`.
3. **Settings → Accelerator → GPU T4 x2** (this is the dialog you saw
   earlier — click "Turn on GPU T4 x2").
4. Check both datasets' actual mount paths once attached — they'll be
   `/kaggle/input/uidatagb-corrected/` and
   `/kaggle/input/uidatagb-cluster-manifest/` if named exactly as above.
   Open the notebook's file browser (left panel) to confirm, since
   Kaggle sometimes appends a version suffix.
5. Update `DATA_ROOT` and `CLUSTER_MANIFEST_PATH` at the top of
   `uidatagb_full_retrain.py` if either mount path differs from the
   defaults.

## Step 5 — Run

Copy the contents of `uidatagb_full_retrain.py` into one notebook cell
(or upload the `.py` file via **Add Data → Upload** and `%run` it), then
**Run All**.

Expect several hours of GPU time: 4 models × up to 20 epochs each,
plus two cheap null controls. Kaggle's free tier gives 30 GPU-hours per
week — if you're close to that limit, you can split this into two
sessions:
- Session 1: comment out everything in `main()` except the primary
  model + evaluate_on_test call.
- Session 2: run the rest — the script skips anything already
  completed (checks for output files first), so it resumes safely.

## Step 6 — Download results

Kaggle does **not** persist `/kaggle/working/` across sessions unless
you explicitly save it. Before your session ends:
- Click **Save Version** (top right) with **Save & Run All** — this
  commits `/kaggle/working/uidatagb_retrain_outputs/` as the notebook's
  output, downloadable afterward from the notebook's **Output** tab.

Bring the downloaded `uidatagb_retrain_outputs/` folder back here and
I'll fold the real numbers into the paper's Stage 2 section and
reconcile them against the review's points.

## What to look at first once it finishes

1. `null_control_untrained/consistency.json` and
   `null_control_random_heatmaps/cosine_floor.json` — read these
   BEFORE trusting any other consistency number. If either floor is
   close to ~0.9, the earlier paper's 0.94 consistency claim has much
   less content than presented.
2. `consistency_same_data/consistency.json` — the corrected,
   same-data-different-seed consistency score, with `per_replicate_mean_std_of_means`
   as the number that should go in the paper's std column (not the old
   pooled-image std, which was the Table V/VI bug).
3. `primary_seed42/history.json` — plot `train_acc` vs `val_acc` and
   `train_loss` vs `val_loss` over epochs to see overfitting/underfitting
   directly, per the workflow checklist.
4. `primary_seed42/test_metrics.json` — the one, final, unbiased test
   number.
