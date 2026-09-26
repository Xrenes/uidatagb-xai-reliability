# Claim-evidence matrix (Phase 12)

Every load-bearing quantitative or interpretive claim in
`UIdataGB_Paper.html` / `UIdataGB_Paper_Short.html`, its evidence file, the
wording the paper is allowed to use, and the boundary that keeps the wording
honest. Built from `fedgb/outputs/frozen_results/results_master.csv` and the
raw `outputs/phase0/*.json` files (Phase 0).

| # | Claim | Evidence file | Allowed wording | Limitation / boundary |
|---|---|---|---|---|
| 1 | Released split is contaminated by near-duplicate images | `outputs/phase0/leakage_check.json` | "93.2% of validation images had a near-duplicate (pHash Hamming≤ ? bits) in training; released-split accuracy 98.78%" | Threshold-dependent; see `outputs/leakage/phash_threshold_sensitivity.csv` for sensitivity to the 8-bit choice |
| 2 | Cluster-level (Stage 1) correction removes measured near-duplicate leakage | `outputs/phase0/dedupe_split_report.json` | "0% residual near-duplicate leakage under the selected pHash rule; still yields 93.05% accuracy" | "0% leakage" is relative to the pHash detector and threshold, not proof of true independence; 37 mixed-class components exist and require manual review (`outputs/leakage/cross_label_component_audit.csv`) |
| 3 | Stage 1 accuracy is reproducible across seeds | `outputs/seeds/{seed_7,seed_42,seed_123}/centralized.json`, `outputs/seeds/aggregate.json` | "three-seed mean 0.9210±0.0131 (accuracy), 0.9207±0.0131 (macro-F1), 0.9948±0.0023 (macro-AUC)" | Same-data seed variance only; does not capture split-composition variance (Phase 2.7 repeated-group evaluation not yet performed) |
| 4 | Filename-prefix groups substantially overlap Stage 1's train/val boundary | `outputs/phase0/session_leakage_check_report.json` | "95.0% of filename-prefix groups (96.7% of images) still cross the Stage 1 split boundary" | Filename-prefix group identity is inferred from the release's naming convention, not confirmed by the dataset provider as patient/session ID (see `outputs/correspondence/dataset_provenance_email_draft.md`, unsent) |
| 5 | Group-level (Stage 2) correction collapses accuracy | `outputs/phase0/session_leakage_check_report.json` (single run); `outputs/session_corrected/multiseed_summary.json` (3-5 seed mean) | "single diagnostic run: 29.59% accuracy (macro-F1 0.2853, macro-AUC 0.7016); independently confirmed by a 3-seed replication, 0.3341±0.0252" | The 29.59% figure and the 0.3341 three-seed mean are DIFFERENT runs of the same protocol, not the same number — do not conflate them (this was the Phase 0 contradiction found and fixed in this pass) |
| 6 | Group-associated information is directly learnable, not just inferred from an accuracy drop | `outputs/phase0/batch_id_diagnostic.json` | "a classifier restricted to Carcinoma alone reaches 100% held-out accuracy (10× chance) distinguishing 10 acquisition batches" | Single disease label only; mechanism (patient vs. site vs. scanner vs. export artefact) is unconfirmed — "non-anatomical shortcut" language must stay about the classification task, not a claim about the physical source |
| 7 | Grad-CAM++ / Score-CAM are the most consistent/faithful XAI methods under this protocol | `outputs/phase0/consistency.json`, `outputs/phase0/faithfulness.json` | "highest agreement/faithfulness under the reported ResNet-50, layer4 target-layer, sample, and metric protocol" | Not a universal ranking claim; Eigen-CAM scored lowest (macro_mean 0.242) but this is protocol-specific |
| 8 | Grad-CAM++ passes the model-randomization sanity check | `outputs/phase0/sanity_checks.json` | "heatmap similarity to the trained model collapses as parameters are randomized fc→conv1 (Spearman 0.948→0.204–0.325), confirming the explanation depends on learned weights" | Tested for Grad-CAM++ only in the current run; not yet repeated for all 5 methods |
| 9 | Temperature scaling improves calibration | `outputs/phase0/calibration.json` | "T=1.468 fit on a held-out calibration half reduces ECE 0.034→0.009 and Brier 0.109→0.106 without changing accuracy" | Fit/evaluated on the Stage 1 split only; Stage 2 calibration not yet computed |
| 10 | Margin-crop ablation shows genuine central-anatomy dependence, not a margin shortcut | `outputs/phase0/shortcut_audit.json` | "cropping the outer 12% margin drops accuracy 0.930→0.712, while only 21.8% of Grad-CAM++ mass falls in that 41.0%-area ring" | Explicitly flagged in the manuscript itself as "the one reliability result... not fully clean" — confounded with genuine anatomy loss; margin-dependence test only, not a full shortcut audit (superseded by the batch-ID diagnostic, claim #6) |
| 11 | Second-backbone (DenseNet-121) check of the Stage 2 collapse | `outputs/session_corrected_densenet121/multiseed_summary.json` (in progress: seed 7 done, accuracy 0.3395/F1 0.3432/AUC 0.7511, consistent with the ResNet-50 collapse; seeds 42/123/2026/2027 still training) | "consistent with the ResNet-50 finding at the seed(s) completed so far" — do not yet claim the full 5-seed DenseNet-121 mean until training finishes | Until all 5 seeds complete, keep "conclusions are limited to the selected ResNet-50 architecture" as the primary bounded claim per Phase 3.3, with the in-progress DenseNet-121 seed cited only as early corroborating evidence |
| 12 | Group-aware (not just image-level) uncertainty on Stage 1 | `outputs/phase0/group_aware_statistics.json` (complete, all 3 seeds) | "Group-aware 95% bootstrap CIs (resampling whole filename-prefix groups, n=209 groups over 3,193 validation images) are 2.7-2.9x wider than naive image-level bootstrap CIs for accuracy at every seed (seed 7: 0.0179 to 0.0513; seed 42: 0.0169 to 0.0456; seed 123: 0.0200 to 0.0537), confirming within-group images are correlated and the plain image-level CI understates true uncertainty. Also adds macro PR-AUC (0.968-0.981) and balanced accuracy (0.909-0.929) per seed." | Computed on the Stage 1 split only (still group-confounded, not the Stage 2 group-separated split); group identity is filename-prefix-derived, same caveat as claim #4 |
| 13 | pHash threshold choice (8 bits) is not arbitrary | `outputs/leakage/phash_threshold_sensitivity.csv`, `outputs/leakage/phash_threshold_sensitivity_summary.json`, `outputs/leakage/cross_label_component_audit.csv` (all complete) | "Mixed-class components stay low and the cross-split match rate stays 0% through threshold=8 (37 mixed-class components, 2,993 total components), but both jump sharply at threshold=10 (118 mixed-class components, 14.78% of validation images gain a spurious cross-split match) and worsen further at threshold=12 (100 mixed-class components at only 1,657 total components, 48.01% cross-split match, one runaway component of 4,272 images) — the selected threshold=8 sits just below where cross-label contamination and threshold over-merging both start compounding." | 37 cross-label components exist even at threshold=8 and are logged UNREVIEWED in `cross_label_component_audit.csv`, pending manual classification (likely-duplicate-inconsistent-label / clinically-overlapping-labels / pHash-false-match / different-crop-same-source / uncertain) — do not claim threshold=8 has zero cross-label risk, only that it minimizes it relative to looser thresholds |

## Explicitly forbidden claims (checked against both HTML files in this pass — none found)

Swept for: `leakage-free` (only used in negated form), `patient-independent`,
`multicentre validation`, `clinically validated`, `correct anatomy`,
`causal explanation`, `ready for deployment`, `universal best method`,
`first-ever study`, `confirmed non-anatomical shortcut`,
`hospital-independent`, `scanner-independent`, `clinically ready`,
`clinical deployment` (only in negated form), `externally validated` (only
in negated form), `deployment readiness`, `radiologist-equivalent`,
`state-of-the-art`. **Result: no unhedged instance of any blocklisted
phrase found in either manuscript file as of this pass.**

## Rows still marked "pending"

Rows 11-13 depend on background jobs (DenseNet-121 training, group-aware
statistics, pHash sensitivity sweep) that were still running when this
matrix was built. Update this file once those complete and the
corresponding manuscript sections are revised.
