# Stage 2 status against the 21-point review

Updated 2026-09-26, after the Kaggle T4 retrain (`kaggle_retrain/uidatagb_full_retrain.py`)
completed and its outputs were verified and folded into `uidatagb-paper/sections/04_results.tex`
(new Section IV.5) and `06_limitations.tex` (2b, 3c updated).

## Resolved by the Stage 2 run

| # | Review point | Resolution |
|---|---|---|
| 2 | No null control for the consistency metric | Untrained-model floor (0.333±0.231) and synthetic-heatmap floors (uniform 0.755, ReLU-Gaussian 0.325) now computed. Trained same-data consistency (~0.84) clears both realistic floors by ~2.5x. **Claim survives**, but only checked for Grad-CAM, not all five methods. |
| 3 | Replicates were disjoint-data, not same-data/different-seed | Fixed: primary + 3 replicates now trained on the identical 7,470-image training set, varying only seed. |
| 4 | Cosine similarity structurally inflated by ReLU+normalisation | Same null controls as #2 answer this directly. |
| 6 | Std column in Table V was pooled-image std, not std-of-replicate-means | Recomputed correctly: std-of-means = 0.0196 vs. the old pooled-std bug (~0.19, an order of magnitude larger). Written up in new Section IV.5's closing paragraph. |
| 12 | 10 fixed epochs, no early stopping, undertrained-Stage-2 concern | Stage 2 uses up to 20 epochs with early stopping on val macro-F1 (patience 5); all 4 models converged within that budget. Doesn't fully satisfy the review's ask (no scheduler, no stronger-backbone sweep), but addresses the core "not a fair attempt" objection. |
| 13 | No held-out test set separate from validation | Fixed: proper 70/20/10 split, 1,084-image test set touched exactly once, after every design decision was frozen. |
| 1 / 20 (partially) | Conflicting Stage 2 accuracy numbers (29.59% vs 31.00% vs 33.41% vs 34.97%) | A new, more carefully constructed number now exists: **72.56% ± 2.35%** mean test accuracy across 4 seeds, on a genuinely leakage-controlled split. This doesn't erase the older numbers' existence, but gives the paper one defensible headline to report going forward rather than picking among four flawed ones. See "still open" below for why this isn't a full reconciliation. |

## Still open — nothing below required GPU time; these are writing/analysis fixes

| # | Review point | What's needed |
|---|---|---|
| 1 (fully) | Old Stage 1 93.05% headline and Stage 2 72.56% number coexist in the same paper without the paper's Abstract/title framing being reconciled | Abstract, title emphasis, and Conclusion still describe the paper as if 93.05% is the operative number. Needs a decision: lead with Stage 2 as the honest number and reframe Stage 1 as "what leakage looks like," or restructure the paper around the audit itself per the reviewer's own suggestion. |
| 5 | Eigen-CAM scored on a class-conditional metric it structurally can't satisfy | Not touched by this run. Needs either an explicit caveat in Table V/VI's caption stating Eigen-CAM's low $O_r^c$ is a metric-mismatch artefact, not a reliability finding, or a class-agnostic consistency variant computed separately for it. |
| 7 | No confidence intervals on any XAI table | Not addressed. Needs bootstrap CIs added to Tables V, VI, VII, VIII (consistency, faithfulness, stability) — same bootstrap machinery already used for classification accuracy in Section IV.1 can likely be reused. |
| 9 | pHash/dihedral-hash threshold (Hamming ≤ 8) never sensitivity-tested | Not addressed. Needs a sweep at 4/6/8/10/12 bits reporting cluster count and Stage 2 accuracy at each, plus a manual-inspection false-positive/negative check on a sample. |
| 10 | Filename→"session"/batch inference never validated against real acquisition metadata | Not addressed. Needs either contacting the dataset providers (Turki et al.) for the actual export convention, or quantitative evidence (shared image dimensions, JPEG quantisation tables, speckle statistics) that within-prefix images share acquisition characteristics that across-prefix images don't. |
| 11 | Batch-ID diagnostic (100% accuracy) not itself dedup'd, so may be circular | Not addressed by this run (this run fixed the *disease classifier's* split, not the separate batch-ID diagnostic). Needs the batch-ID diagnostic rerun with the dihedral cluster manifest applied so no cluster spans its own train/test sides. |
| 14 | Table IX (prior-work comparison) includes non-gallbladder method papers as straw-man rows | Writing-only fix. Needs the table rebuilt restricted to actual gallbladder-ultrasound classification studies. |
| 15 | GBCU cross-dataset audit (Section 5.3) draws a generalisability conclusion from an overlap count with no measured accuracy impact | Writing-only fix. Either measure the accuracy impact on GBCU, or cut the subsection to one sentence in Limitations as the review suggests. |
| 16 | Shortcut audit only tests margin/border shortcuts, not the batch-correlated shortcut the paper's own Stage 2 analysis implies exists | Not addressed. Needs a new test: accuracy gap between held-out images from *seen* batches vs. *unseen* batches, or a linear probe on layer4 features testing whether batch identity is recoverable. |
| 17 | Noise-robustness finding (29.2% of predictions flip under barely-visible Gaussian noise) is flagged but never followed up | Not addressed. Needs accuracy-under-perturbation reported directly, swept across a range of sigma values, ideally re-run on the Stage 2 split too. |
| 18 | Anatomical-plausibility claims (Section 5.1) are unblinded author judgement presented as a finding | Writing-only fix. Needs either explicit hypothesis-generating framing throughout, or an actual radiologist review before resubmission. |
| 19 | DeLong's test misapplied (compares correlated curves on the same samples; used here across different seeds), no multiplicity correction, n=3 seeds insufficient to call seed 123 "genuinely weaker" | Not addressed. Needs the DeLong comparison either dropped in favour of McNemar's (already run correctly two sentences later) or properly justified; multiplicity correction added; the seed-123 claim softened or backed by more seeds. |
| 21 | No code repository URL; DOI "upon acceptance" | Documentation-only fix — publish the `fedgb/` and `kaggle_retrain/` code to a public repo now that Stage 2 results exist to accompany it. |

## Net count

- **6 of 21** resolved outright by the Kaggle run (2, 3, 4, 6, 12, 13)
- **1 of 21** substantially improved but not fully closed (1/20 — a good new number exists, but the paper's framing hasn't caught up to it yet)
- **14 of 21** untouched, and every one of them is a writing, analysis, or small-script fix rather than a GPU-time problem — none require retraining.

## Suggested next priorities, cheapest first

1. **Review 21** — publish the code. Zero technical risk, purely administrative, and unblocks reviewers verifying everything else.
2. **Review 14 + 15 + 18** — pure prose fixes to the comparison table, GBCU subsection, and anatomical-plausibility framing. No new computation needed.
3. **Review 9** — the pHash/dihedral threshold sweep. Reuses tooling already built (`generate_dihedral_cluster_manifest.py`), just needs to loop over a few threshold values.
4. **Review 11** — rerun the batch-ID diagnostic with the dihedral manifest applied. Same reused tooling, moderate effort.
5. **Review 7** — add bootstrap CIs to the XAI tables. Statistical add-on, no retraining.
6. **Review 17** — sweep noise sigma and report accuracy directly. Cheap, reuses the existing stability-test harness.
7. **Review 16** — the actual batch-correlated shortcut test. Needs new code but no fresh full training run (a layer4 linear probe on the already-trained Stage 2 checkpoints would work).
8. **Review 5, 19** — metric/statistics fixes, cheap once someone sits down to do them.
9. **Review 10** — needs either provider contact (out of your control) or a same-effort quantitative acquisition-metadata check as a fallback.
10. **Review 1 (fully)** — the big structural decision: reframe the paper's Abstract/title/Conclusion around Stage 2 as the honest headline. This is a judgment call for you, not a technical task, and probably the last thing to lock in once everything else above is settled.
