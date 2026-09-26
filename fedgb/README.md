# UIdataGB XAI reliability study — code and artifacts

Code and generated artifacts for "Quantitative Evaluation of Explainable AI
Methods for Gallbladder Diseases through Ultrasound Images"
(Sayed Ifti Ahmed, Mst. Khadiza Akter Sammi; Department of Computer Science
and Engineering, Daffodil International University, Dhaka, Bangladesh). The
paper's primary contribution is a five-method quantitative XAI benchmark and
reliability battery (Grad-CAM, Grad-CAM++, Saliency Maps, Eigen-CAM,
Score-CAM); the two-stage data-leakage correction is necessary supporting
methodology so that benchmark is computed on genuinely held-out data, not
the paper's headline finding on its own.

This repository is scoped to the UIdataGB nine-class gallbladder-disease
classification study. Adjacent directories under `fedgb/` from an earlier
federated-learning (5-class) project share some infrastructure (model.py,
gradcam.py) but are not part of this paper.

## Environment

- Python 3.12.10
- PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121 (`requirements-lock.txt` has
  the full pinned environment)
- GPU used for all training/evaluation runs: NVIDIA GeForce GTX 1050 (4GB),
  driver 582.66
- Windows; all scripts run from the `fedgb/` directory

```
pip install -r requirements-lock.txt
```

## Pipeline (matches the manuscript's Section 3.1 and the revision plan's
Phase-0-through-5 checklist)

1. **Released-split leakage audit** — `check_leakage.py` →
   `outputs/phase0/leakage_check.json`, `outputs/figures/leakage_hist.png`
2. **pHash threshold sensitivity** — `phash_threshold_sensitivity.py` →
   `outputs/leakage/phash_threshold_sensitivity.csv`,
   `outputs/leakage/cross_label_component_audit.csv`,
   `figures/phash_threshold_sensitivity.png`
3. **Stage 1 near-duplicate cluster-level split correction** —
   `dedupe_split_uidatagb.py` → `outputs/phase0/dedupe_split_report.json`
   (mutates `data/uidatagb/` in place, moving files between
   `training/`/`validation/` to match the cluster-level split)
4. **Stage 1 multi-seed classification** — `run_seeds.py` (seeds 7, 42, 123)
   → `outputs/seeds/seed_<n>/centralized.json`,
   `outputs/seeds/aggregate.json`
5. **Stage 1 statistical tests** — `stage1_statistical_tests.py`
   (McNemar, DeLong, image-level bootstrap) →
   `outputs/phase0/stage1_statistical_tests.json`
6. **Group-aware statistics** — `group_aware_statistics.py` (group
   bootstrap by filename-prefix, PR-AUC, balanced accuracy) →
   `outputs/phase0/group_aware_statistics.json`
7. **Session/filename-prefix-group leakage audit and Stage 2 split** —
   `session_leakage_check.py` →
   `outputs/phase0/session_leakage_check_report.json` (creates
   `data/uidatagb_sessioncorrected/`)
8. **Stage 2 multi-seed classification** — `session_extra_seeds_5.py`
   (ResNet-50, seeds 7, 42, 123, 2026, 2027) →
   `outputs/session_corrected/seed_<n>/`,
   `outputs/session_corrected/multiseed_summary.json`
9. **Stage 2 second-backbone check** — `session_densenet_seeds.py`
   (DenseNet-121, same 5 seeds) →
   `outputs/session_corrected_densenet121/`
10. **Batch-ID shortcut diagnostic** — `batch_id_shortcut_diagnostic.py` →
    `outputs/phase0/batch_id_diagnostic.json`
11. **XAI reliability battery** (consistency, faithfulness, stability,
    sanity checks, calibration, shortcut/margin audit) —
    `compute_consistency.py`, `compute_faithfulness.py`,
    `compute_stability.py`, `sanity_checks.py`, `compute_calibration.py`,
    `shortcut_audit.py` → corresponding files in `outputs/phase0/`

## Source-of-truth results

`outputs/frozen_results/results_master.csv` is the single traceable table
every number in the manuscript should match (Phase 0). Each row names the
exact JSON file it was computed from. If a manuscript number does not
appear in this file, treat it as unverified until added.

## Claim-evidence matrix

`paper/claim_evidence_matrix.md` maps every quantitative/interpretive claim
in the manuscript to its evidence file, allowed wording, and limitation
boundary (Phase 12).

## Tests

`tests/test_split_integrity.py` re-derives (does not just trust cached
JSON) that the on-disk Stage 1 split has zero cross-split pHash
near-duplicates and the Stage 2 split has zero filename-prefix groups
crossing train/validation — the two invariants the manuscript's leakage
claims depend on. Run with `python tests/test_split_integrity.py` or
`python -m pytest tests/ -v` from `fedgb/`.

## Reproducibility notes

- All training uses fixed seeds via `torch.manual_seed`,
  `np.random.seed`, and `random.seed`; `num_workers=0` in all DataLoaders
  (multi-worker loading was measurably slower and less deterministic on
  this machine — see comments in `session_extra_seeds.py`).
- Checkpoint SHA-256 hashes: run `sha256sum outputs/**/*.pt` (or
  `checksums.py`, see below) before archiving a release.
- Dataset manifests are not redistributed in this repository pending
  confirmation of the UIdataGB release's license terms for derived
  artifacts (see `outputs/correspondence/ethics_and_licence_questions_DIU.md`).

## What is NOT yet resolved (see the revision plan and claim-evidence matrix)

- Filename-prefix group identity (patient/session/scanner/hospital) is
  inferred, not confirmed by the dataset providers
  (`outputs/correspondence/dataset_provenance_email_draft.md`, drafted but
  unsent).
- No clinician/radiologist has yet rated any XAI output or annotated any
  ROI (`outputs/correspondence/radiologist_study_status.md`).
- DIU ethics/licence confirmation is outstanding
  (`outputs/correspondence/ethics_and_licence_questions_DIU.md`).
- A dated, logged database literature search has not been run; current
  novelty language is deliberately hedged pending one
  (`paper/phase8_literature_search_status.md`).
