# Training paused — status as of this pause

**Paused by explicit user request.** All completed training data is saved
and checksummed (`outputs/frozen_results/checksums.txt`, 63 files, SHA-256).
Nothing that had already reached a save point was lost.

## What's complete and saved

- **Stage 1 (near-duplicate-corrected split), ResNet-50, 3 seeds** (7, 42,
  123): `outputs/seeds/seed_{7,42,123}/centralized.pt` + metrics.
- **Stage 2 (batch-corrected split), ResNet-50, 5 seeds** (7, 42, 123, 2026,
  2027): `outputs/session_corrected/seed_*/primary.pt` + metrics +
  `multiseed_summary.json` (accuracy 0.3497±0.0279). This is complete and
  was the headline retraining result added to both manuscripts.
- **Stage 2, DenseNet-121 second backbone, seed 7 only**:
  `outputs/session_corrected_densenet121/seed_7/primary.pt` (accuracy
  0.3395, consistent with the ResNet-50 collapse).
- pHash threshold sensitivity, group-aware bootstrap statistics, and all
  Phase 0 diagnostic JSONs — all complete, unaffected by this pause.

## What was lost (acceptable, by design)

- **DenseNet-121 seed 42**: was at epoch 8/10 when stopped. The training
  script (`session_densenet_seeds.py`) only writes a checkpoint after all
  10 epochs of a seed complete, so this in-progress run was not saved and
  will restart from epoch 1, not resume from epoch 8, next time.
- **DenseNet-121 seeds 123, 2026, 2027**: never started.

## To resume

Run from `fedgb/`:
```
python session_densenet_seeds.py
```
The script already checks for existing `primary_metrics.json` per seed and
skips seeds that are done (seed 7 will be skipped automatically), so
resuming will only retrain seeds 42, 123, 2026, and 2027 — no need to
re-run seed 7. Estimated remaining time: ~4 seeds × ~3.25-3.6 hours each
≈ 13-14 hours on this machine's GPU (GeForce GTX 1050).

## What still depends on the remaining DenseNet-121 seeds

- `paper/claim_evidence_matrix.md` row 11 (second-backbone check) is
  currently bounded to "early corroborating evidence from one seed," not a
  full 5-seed mean — update once training resumes and completes.
- The manuscript sections (`UIdataGB_Paper.html`,
  `UIdataGB_Paper_Short.html`, and `uidatagb-paper/sections/*.tex`) already
  correctly describe this as "in progress... one seed complete" — no
  overclaim exists, so nothing needs to be walked back for the pause itself.
