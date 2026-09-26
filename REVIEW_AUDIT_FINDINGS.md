# Audit of the 21-point review against actual pipeline state

Compiled 2026-09-22 by checking every cited number/claim directly against
`fedgb/outputs/**/*.json` and the generating scripts, not by trusting either
the review or the manuscript at face value.

## Root cause: two divergent paper versions

`uidatagb-paper/README.md` names `../UIdataGB_Paper.html` as content
source-of-truth and warns to re-sync `sections/*.tex` from it after any HTML
revision. That re-sync never happened for the Stage 2 (session/batch-level
leakage correction) work. The `.tex` files being edited/reviewed contain only
Stage 1 (pHash near-duplicate correction, 93.05% accuracy) with **no** Stage 2
section, no batch-ID diagnostic, no GBCU cross-dataset audit. All of Reviews
1, 2, 3, 6 (Stage-2-column claims), 9-16's Stage-2-dependent parts, 20's
29.59-vs-31.00 contradiction, and 21 are about content that exists in the
`fedgb/` pipeline outputs but was never written into the paper being reviewed
here. This is not the reviewer inventing numbers -- it's a real, more advanced
analysis that the paper drafting fell behind on.

## Verified against source files (real, confirmed)

| # | Claim | Verified against | Status |
|---|---|---|---|
| 1/20 | Stage 2 single-run accuracy 29.59% | `outputs/phase0/session_leakage_check_report.json`: `session_corrected_accuracy: 0.29591...` | **CONFIRMED**, exact match |
| 1/20 | Stage 2 3-seed mean 0.3341 (matrix) vs. current 5-seed mean 0.3497±0.0279 | `outputs/session_corrected/multiseed_summary.json` (5 seeds: 7,42,123,2026,2027) | **CONFIRMED as a THIRD, more recent number** -- pipeline was rerun with 5 seeds since both the review and the claim-evidence matrix were written. Current canonical: **34.97% ± 2.79%** (n=5), individual seed 42 = 30.99% |
| 3 | Replicates are disjoint subsets, not same-data-different-seed | `session_replicate_train.py` docstring line 83-84: "three replicates: disjoint random thirds of the training set (matches the original replicate-subset methodology...)" | **CONFIRMED**, admitted in the script's own comment |
| 4 | ReLU + [0,1] normalization inflates cosine similarity; no null baseline exists | `gradcam.py` lines 94/176/183/402 (`F.relu` on all CAM outputs); no `random`/`untrained`/`null` baseline script found anywhere in `fedgb/` | **CONFIRMED**, no mitigating script exists |
| 6 | Table V std column doesn't match the 3 replicate means shown | `compute_consistency.py` line 129: `per_replicate_mean` computed separately from line 128 `macro_std = np.std(all_scores)` (pooled per-image, not per-replicate-mean) | **CONFIRMED real bug** -- the std column is pooled-image std, not std-of-3-means, and the table/caption present it as if it were the latter |
| 9 | 37 mixed-class pHash components at threshold=8, unreviewed | `claim_evidence_matrix.md` row 2 and row 13 | **CONFIRMED**, matrix itself flags these as unreviewed |
| 10 | 96.73% of images cross the Stage 1 split at session/filename-prefix level | `session_leakage_check_report.json`: `pct_cross_split_images: 96.73`, `n_mixed_class_sessions: 0` | **CONFIRMED** -- this is real and is the actual justification for Stage 2 existing at all |
| 11 | Batch-ID diagnostic (100% accuracy) not pHash-de-duplicated within itself | `outputs/phase0/batch_id_diagnostic.json`: perfect block-diagonal confusion matrix, image-level split, 629 train / 271 val on 10 batches, no dedup step visible in the diagnostic's own methodology | **CONFIRMED**, matches Review 11's technical objection exactly |
| 21 | No code repository URL, DOI "upon acceptance" | `sections/08_declarations.tex` (current .tex) | **CONFIRMED**, current text still says this |

## Not yet verified / needs more checking

- Review 5 (Eigen-CAM class-agnostic scoring artefact) -- plausible given the
  paper's own text says Eigen-CAM is class-agnostic twice, but I have not
  re-derived whether the 0.073/0.085 collapse is *purely* a metric artefact
  or partly real signal. Needs the null-control result (Review 2/4) to
  actually disentangle.
- Review 7 (no CIs on XAI tables) -- true by inspection of `compute_consistency.py`
  (no bootstrap call found), not yet cross-checked against `compute_faithfulness.py`
  and `compute_stability.py`.
- Review 12 (undertrained Stage 2 baseline) -- **partially answered already**:
  `outputs/session_corrected/ablation/summary.json` shows two ablations
  (per-image normalization: 32.88% vs baseline 30.99%; stronger augmentation:
  35.15% vs baseline 30.99%) already ran and neither closes the gap to
  Stage-1-like performance. This is evidence the collapse is not simply an
  undertrained-baseline artifact, though a full scheduler/early-stopping/
  stronger-backbone sweep per the review's exact ask has not been done.
- Review 13, 15, 17, 18, 19 -- not yet checked against source; likely valid
  given the hit rate on everything else, but unverified.

## What genuinely needs new work (not just prose fixes)

1. **Null controls (Reviews 2, 4)** -- do not exist, must be written and run.
   Cheapest, highest-value fix per the reviewer's own framing.
2. **Same-data-different-seed replicates (Review 3)** -- requires 3 new full
   training runs on identical data, different seed only. Expensive but
   changes what every consistency number in the paper actually means.
3. **Table V std-column fix (Review 6)** -- either relabel what's being
   reported or add the correct per-replicate-mean std; cheap, no retraining.
4. **Batch-ID diagnostic re-run with pHash dedup applied within-batch
   (Review 11)** -- moderate effort, reuses existing pHash tooling.
5. **Reconcile which Stage 2 number is canonical** (29.59% single-run vs.
   33.41% 3-seed vs. 34.97% 5-seed) before writing any of it into the paper.
