# Radiologist ROI study — status and what's still needed

**Status: infrastructure exists; awaits an actual clinician (Phase 4.11 / 7).
I cannot recruit or substitute for a radiologist/ultrasound expert myself.**

## What already exists

- `fedgb/build_radiologist_study.py` — builds a stratified panel set
  (10 images/class, spanning confidence range, correct + incorrect cases)
  with Grad-CAM++ overlays, a blank Likert rating CSV, and a manifest.
- `fedgb/outputs/radiologist_study/` — already-generated panels, raw
  images, and a rating template.
- `fedgb/compute_kappa.py` — computes inter-rater agreement once two or
  more radiologists have filled in the template.
- `future/RADIOLOGIST_STUDY.html`, `future/roi_annotation_tool.html` /
  `roi_annotation_tool_template.html` — a browser-based ROI annotation
  tool for bounding-box/region annotation.

## What's missing before this satisfies Phase 4.11 / 7

1. **The existing panels were generated from the Stage 1 model
   (`build_model`/ResNet-50 on the near-duplicate-corrected split), not
   the Stage 2 (session/batch-corrected) model.** Per the plan and per
   Section 5.4 of the manuscript, Stage 1 ratings would be made on a model
   "now known to rely heavily on a shortcut" — any radiologist review
   should be redone on Stage 2 checkpoints once available
   (`outputs/session_corrected/seed_42/primary.pt` after retraining, or
   the DenseNet-121 equivalent) to be reported as this paper's clinician
   validation, not just kept as a Stage 1 pilot.
2. **No radiologist has actually rated anything yet** — `rating_template.csv`
   is blank; `compute_kappa.py` has nothing to compute on.
3. **Needs 1-2 qualified radiologists/ultrasound experts** willing to spend
   time on ~50-100+ image ratings and/or ROI annotation. This is a
   recruitment task for the authors (DIU clinical collaborators, or an
   external radiologist contact), not something I can do.

## Recommended next step (for you, not automatable)

Once Stage 2 checkpoints exist (after the current training run finishes),
regenerate the panel set against the Stage 2 model, then reach out to a
radiologist/ultrasound-specialist collaborator to complete
`rating_template.csv` and/or use the ROI annotation tool. Until that
happens, the manuscript should keep its current posture: no clinician-valid
localization claim, "future validation step" language in Discussion/
Limitations (already present).
