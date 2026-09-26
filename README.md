# UIdataGB: Quantitative Evaluation of Explainable AI Methods for Gallbladder Diseases through Ultrasound Images

Code, analysis pipeline, and LaTeX paper source for a quantitative XAI-reliability
study on nine-class gallbladder ultrasound classification, using the public
UIdataGB dataset (Turki et al., *Data in Brief*, 2024).

The paper's central contribution is a five-method quantitative XAI benchmark
and reliability battery (Grad-CAM, Grad-CAM++, Saliency Maps, Eigen-CAM,
Score-CAM). Getting a trustworthy number to run that battery on required a
two-stage data-leakage audit and correction, which is reported as supporting
methodology, not the paper's headline finding on its own — an initial split
looked clean but let near-duplicate frames from the same imaging session
cross the train/validation boundary, inflating apparent accuracy from an
honestly-measured ~72.6% to 93.05%.

## Repository layout

```
fedgb/              Main analysis pipeline: dataset handling, leakage
                     correction, all five XAI methods, the full quantitative
                     reliability battery (consistency, faithfulness,
                     stability, sanity checks, calibration, shortcut audit),
                     and figure generation. See fedgb/README.md for the
                     full pipeline order and reproducibility notes.
kaggle_retrain/     Kaggle-GPU scripts for the corrected Stage 2 retrain
                     (cluster-aware 70/20/10 split, same-data replicate
                     seeds, null controls for the consistency metric) and
                     follow-up audits (pHash threshold sensitivity,
                     leakage-controlled batch-ID diagnostic, a linear probe
                     for batch-shortcut evidence in the disease classifier's
                     own features, and a noise-robustness sweep). Designed
                     to run standalone in a Kaggle notebook against the
                     dataset and cluster manifest as attached Kaggle
                     Datasets; each script is resumable and checkpoints its
                     own progress.
uidatagb-paper/     IEEEtran LaTeX source for the manuscript. Build with
                     `latexmk -pdf -outdir=build main.tex` from this
                     directory (see uidatagb-paper/README.md for the local
                     MiKTeX bibtex workaround this project needs).
```

## Status

The dataset images and trained model checkpoints are not included in this
repository (see `.gitignore`) — they are either the public UIdataGB release
itself (linked in the paper's Data Availability statement) or are too large
for version control. Small JSON/log result files that the paper's reported
numbers are computed from are retained under `fedgb/outputs/phase0/` and
`fedgb/outputs/frozen_results/`.

Several items remain open at the time of this release and are stated
plainly in the paper's Limitations section and in `fedgb/README.md`'s "What
is NOT yet resolved" list — most notably, direct confirmation of the
dataset's acquisition/export convention from the original providers (a
draft, unsent inquiry is included at
`fedgb/outputs/correspondence/dataset_provenance_email_draft.md`), and
clinician/radiologist validation of the XAI outputs. This repository
reports what has and has not been established, rather than overstating
either.

## Getting started

For the main analysis pipeline (leakage audit, classifier training, XAI
battery), see `fedgb/README.md`. For the Kaggle-GPU Stage 2 retrain and
follow-up audits, see `kaggle_retrain/README.md`. For building the paper,
see `uidatagb-paper/README.md`.

## Citation

If you use this code, please cite the paper (see `fedgb/CITATION.cff`).

## License

Source code is licensed MIT (see `fedgb/LICENSE`). The UIdataGB dataset
itself is a separate public release with its own terms; this repository
does not redistribute it.
