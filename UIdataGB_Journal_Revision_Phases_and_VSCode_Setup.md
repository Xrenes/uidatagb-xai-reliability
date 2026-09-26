# UIdataGB Journal Paper Revision Plan and VS Code Structure

**Working project title:**  
**When Explainable AI Explains a Shortcut: Data Leakage and XAI Reliability in Gallbladder Ultrasound Classification**

**Authors:** Sayed Ifti Ahmed; Mst. Khadiza Akter Sammi  
**Institution:** Department of Computer Science and Engineering, Daffodil International University, Dhaka, Bangladesh  
**Purpose:** This document converts the full paper review into an executable, phase-by-phase revision checklist. It separates:

- changes that can be completed by rewriting;
- changes that require retraining or new analysis;
- changes that require metadata, clinician input, or institutional confirmation;
- the recommended VS Code + LaTeX workflow for producing a professional IEEE-style journal manuscript.

---

## 1. Status Legend

| Status | Meaning |
|---|---|
| **Resolved by writing** | Can be corrected using the existing evidence, without new experiments. |
| **Claim must be narrowed** | Existing experiment does not support the original broad claim. |
| **New analysis required** | Existing outputs can be re-analysed, but new statistical or XAI calculations are needed. |
| **Retraining required** | New model checkpoints must be trained using the corrected protocol. |
| **External evidence required** | Requires dataset-provider metadata, clinician annotation, radiologist review, ethics confirmation, or external validation. |
| **Submission blocker** | The paper should not be submitted with a strong claim until this item is addressed. |

---

# 2. Final Scientific Position of the Paper

The paper should be presented primarily as a **data-leakage, grouping, shortcut-learning, and XAI-reliability study**.

The main scientific sequence is:

```text
Released image-level split
        ↓
Near-duplicate audit with perceptual hashing
        ↓
Near-duplicate component-separated split
        ↓
Residual filename-prefix group overlap
        ↓
Filename-prefix group-separated internal validation
        ↓
Retraining and re-evaluation
        ↓
Classification, calibration, and XAI reliability analysis
```

The paper must not claim:

- patient-independent validation;
- hospital-independent validation;
- scanner-independent validation;
- fully leakage-free evaluation;
- clinician-validated localization;
- clinical deployment readiness;
- universal superiority of one XAI method;
- confirmed physical cause of the batch-associated signal.

---

# 3. Phase Summary

| Phase | Main objective | Required outcome |
|---:|---|---|
| 0 | Freeze results and remove contradictions | One verified source-of-truth results table |
| 1 | Reframe the scientific contribution | Leakage/shortcut learning becomes the primary story |
| 2 | Validate dataset grouping and splitting | Group-aware, non-overlapping development/calibration/test design |
| 3 | Retrain models and repair replication | Stage 2 checkpoints using clean same-data multi-seed training |
| 4 | Strengthen quantitative XAI evaluation | Multiple spatial, faithfulness, stability, and sanity metrics |
| 5 | Repair statistical analysis | Group-aware uncertainty, PR-AUC, balanced accuracy, corrected tests |
| 6 | Reorganize results | Stage 2 results dominate; Stage 1 becomes diagnostic comparison |
| 7 | Correct clinical interpretation | No unsupported anatomical or deployment claims |
| 8 | Strengthen literature review and novelty | Reproducible search and qualified novelty statement |
| 9 | Rewrite every manuscript section | Consistent title-to-conclusion research story |
| 10 | Rebuild IEEE formatting and visuals | Clean IEEEtran two-column LaTeX manuscript |
| 11 | Complete reproducibility, ethics, and availability | Repository, manifests, environment, licence, and declarations |
| 12 | Conduct final claim and submission audit | Every claim linked to evidence and limitation |

---

# Phase 0 — Freeze Results and Remove Contradictions

**Priority:** Submission blocker  
**Status:** Requires verification from real output files

## Problems to correct

- [ ] Stage 2 seed-42 performance appears in more than one form.
- [ ] Preliminary and finalized Stage 2 values are mixed.
- [ ] The three-seed mean was previously detached from its correct sentence.
- [ ] The manuscript previously stated both that multi-seed work was pending and that three seeds were completed.
- [ ] Stage 1 was described inconsistently as corrected, leakage-free, and shortcut-affected.
- [ ] Public-code claims conflicted with a future Zenodo-release statement.
- [ ] Some tables, captions, and narrative paragraphs may use different metric versions.

## Required action

Create a source-of-truth file:

```text
outputs/frozen_results/results_master.csv
```

Recommended columns:

```text
experiment_id
split_name
grouping_rule
seed
checkpoint_path
prediction_file
accuracy
macro_f1
macro_roc_auc
macro_pr_auc
balanced_accuracy
ece
brier_score
date_completed
code_commit
notes
```

## Canonical split names

Use these names everywhere:

1. **Released split**  
   Original dataset split; extensive detected near-duplicate overlap.

2. **Stage 1 near-duplicate component-separated split**  
   Zero detected cross-split pHash matches under the selected threshold, but still filename-prefix-group confounded.

3. **Stage 2 filename-prefix group-separated internal-validation split**  
   No filename-prefix group crosses the defined partitions; this is not verified patient-level independence.

## Required checks

- [ ] Every abstract number matches `results_master.csv`.
- [ ] Every table number matches the source output.
- [ ] Every figure is regenerated from machine-readable results.
- [ ] Every checkpoint has a matching prediction file.
- [ ] Preliminary screening values are clearly labelled or removed.
- [ ] No result is manually typed without a traceable output file.

## Phase 0 completion gate

Do not continue manuscript polishing until all reported values can be traced through:

```text
checkpoint → predictions → metrics file → table/figure script → manuscript
```

---

# Phase 1 — Reframe the Scientific Contribution

**Priority:** Major  
**Status:** Mostly resolved by rewriting

## Core issue

The original paper looked like two separate studies:

1. a comparison of five XAI methods;
2. a leakage and shortcut-learning audit.

The leakage result is more consequential because it changes the interpretation of both classification performance and XAI output.

## Required framing

### Primary contribution

A two-stage audit showing that evaluation results change substantially after:

- near-duplicate separation;
- stricter filename-prefix group separation.

### Secondary contribution

A quantitative comparison of:

- Grad-CAM;
- Grad-CAM++;
- Saliency Maps;
- Eigen-CAM;
- Score-CAM;

under the corrected and bounded evaluation protocol.

## Recommended research questions

**RQ1.** How much does classification performance change when near-duplicate and filename-prefix group overlap are controlled?

**RQ2.** How stable and perturbation-sensitive are the five XAI methods across retrained models under the group-separated protocol?

**RQ3.** Can apparently reproducible explanations coexist with poor generalization to unseen filename-prefix groups?

## Recommended contribution list

Limit the contribution list to four items:

1. A two-stage leakage and grouping audit for the public UIdataGB release.
2. Group-separated multi-seed classification evaluation.
3. A five-method quantitative XAI reliability comparison.
4. Explicit separation between model-behaviour explanation reliability and clinician-valid anatomical localization.

## Title options

### Recommended

**When Explainable AI Explains a Shortcut: Data Leakage and XAI Reliability in Gallbladder Ultrasound Classification**

### More formal

**Data Leakage, Batch-Associated Shortcuts, and Explainable AI Reliability in Gallbladder Ultrasound Classification**

### More method-focused

**A Two-Stage Leakage Audit and Quantitative XAI Evaluation for Gallbladder Ultrasound Classification**

## Phase 1 completion gate

- [ ] One central research story.
- [ ] No more than four contributions.
- [ ] Title, abstract, research questions, results, and conclusion describe the same study.
- [ ] XAI is not presented independently from the leakage problem.

---

# Phase 2 — Dataset Validity and Split Methodology

**Priority:** Critical submission blocker  
**Status:** New analysis and external evidence required

## 2.1 Verify what filename prefixes represent

The filename prefix is currently inferred to represent an acquisition/export group. It is not confirmed as a:

- patient ID;
- study ID;
- video ID;
- scanner ID;
- hospital ID;
- acquisition-session ID.

### Required action

- [ ] Contact the dataset creators.
- [ ] Search the dataset publication and supplementary files.
- [ ] Check any original preprocessing or export scripts.
- [ ] Ask what the prefix, index, and parenthesized number mean.
- [ ] Preserve written confirmation from the provider.

Until verified, use:

> **filename-prefix group**

Avoid presenting it as a confirmed patient or acquisition batch.

## 2.2 Treat groups as the likely statistical units

The raw dataset contains many images, but images within a filename-prefix group are correlated.

### Required analysis

- [ ] Report total number of prefix groups.
- [ ] Report groups per class.
- [ ] Report training, development, calibration, and test groups.
- [ ] Report images per group: minimum, median, mean, maximum.
- [ ] Report whether class labels are constant within groups.
- [ ] Use group-level resampling for confidence intervals.

Do not assume that 10,692 images equal 10,692 independent observations.

## 2.3 Validate the pHash threshold

The selected 64-bit pHash rule uses a Hamming-distance threshold. The threshold needs empirical support.

### Required threshold sweep

Evaluate at:

```text
Hamming distance: 4, 6, 8, 10, 12
```

For each threshold report:

- number of connected components;
- singleton count;
- maximum component size;
- cross-label component count;
- percentage of validation images matched to training;
- manual precision on sampled positive pairs;
- manual inspection of sampled negative pairs near the boundary.

### Required output

```text
outputs/leakage/phash_threshold_sensitivity.csv
outputs/leakage/manual_pair_audit.csv
figures/phash_threshold_sensitivity.pdf
```

## 2.4 Audit connected-component chaining

Graph clustering can join:

```text
A ≈ B
B ≈ C
```

even when `A` and `C` are not directly similar.

### Required analysis

- [ ] Maximum pairwise Hamming distance inside each component.
- [ ] Diameter of large components.
- [ ] Manual inspection of the largest components.
- [ ] Comparison with stricter clustering alternatives.
- [ ] Report whether conclusions change under threshold variation.

## 2.5 Resolve the cross-label components

The analysis identified components containing more than one released label.

### Required action

Manually inspect all cross-label components and classify each as:

- likely duplicate with inconsistent label;
- clinically overlapping labels;
- pHash false match;
- different crop of the same source;
- uncertain.

### Required output

```text
supplementary/cross_label_component_audit.csv
supplementary/cross_label_component_examples.pdf
```

The paper must not automatically call these “inherent ambiguity” without evidence.

## 2.6 Build a clean grouped partition

Preferred design:

```text
Grouped training set
Grouped development set
Grouped calibration set
Locked grouped test set
```

No filename-prefix group may cross any partition.

### Required controls

- [ ] Split by group, not image.
- [ ] Stratify as far as possible by class.
- [ ] Freeze the final test manifest before model selection.
- [ ] Fit temperature scaling only on the grouped calibration set.
- [ ] Use the final test set only after methods and hyperparameters are frozen.
- [ ] Save manifest hashes.

### Recommended files

```text
data/manifests/train_groups.csv
data/manifests/development_groups.csv
data/manifests/calibration_groups.csv
data/manifests/test_groups.csv
data/manifests/manifest_checksums.txt
```

## 2.7 Use repeated grouped evaluation

One fixed group split does not measure split uncertainty.

### Minimum design

- [ ] Five repeated stratified group holdouts, or
- [ ] Stratified Group K-Fold cross-validation.

Separate:

- variability caused by different training seeds;
- variability caused by different grouped partitions.

### Recommended reporting

```text
overall mean
between-split standard deviation
within-split seed standard deviation
group-bootstrap 95% confidence interval
```

## 2.8 Repeat the batch/group diagnostic correctly

The original group-ID diagnostic used an image-level split, which may place highly correlated frames on both sides.

### Corrected diagnostic

- [ ] Restrict to one disease label.
- [ ] Remove or cluster near-duplicates first.
- [ ] Keep duplicate components in one side.
- [ ] Use temporal separation where frame order is available.
- [ ] Repeat across several disease classes.
- [ ] Repeat across several random group selections.
- [ ] Report mean, confidence interval, and chance baseline.

Allowed conclusion:

> Filename-prefix groups contain learnable within-class information under the corrected diagnostic.

Do not claim that the physical mechanism is known.

## Phase 2 completion gate

- [ ] Group meaning verified or carefully bounded.
- [ ] pHash sensitivity completed.
- [ ] Cross-label components audited.
- [ ] Grouped train/development/calibration/test manifests frozen.
- [ ] Repeated grouped evaluation prepared.
- [ ] Corrected group-ID diagnostic completed.

---

# Phase 3 — Model Training and Replication Design

**Priority:** Critical  
**Status:** Retraining required

## 3.1 Retrain the Stage 2 classifier

Do not reuse Stage 1 checkpoints as final evidence.

### Required Stage 2 training

Use only the grouped training partition.

Train at least five same-data seeds, for example:

```text
7, 42, 123, 2026, 2027
```

The exact seeds are less important than using:

- the same training data;
- the same hyperparameters;
- the same model-selection rule;
- independent initialization and data-loader randomness.

## 3.2 Separate two kinds of reproducibility

### A. Same-data seed reproducibility

```text
same grouped training set
same hyperparameters
different seeds
```

This measures run-to-run training variation.

### B. Group-resampling robustness

```text
different grouped partitions or grouped bootstrap samples
comparable sample sizes
preserved class balance where possible
```

This measures sensitivity to group composition.

Do not combine these and call the result pure seed reproducibility.

## 3.3 Add a second backbone or narrow the claim

Recommended minimum comparison:

- ResNet-50;
- DenseNet-121, EfficientNet-B0, or another justified CNN.

The objective is not to maximize accuracy at any cost. The purpose is to test whether the leakage/generalization conclusion survives a reasonable architecture change.

If no second backbone is added, state clearly:

> Conclusions are limited to the selected ResNet-50 architecture and target layer.

## 3.4 Validate the training schedule

The ten-epoch schedule requires justification.

### Required checks

- [ ] Longer maximum epoch budget.
- [ ] Early stopping defined using grouped development performance.
- [ ] Learning-rate scheduling.
- [ ] Frozen-backbone versus full fine-tuning.
- [ ] Learning-rate sensitivity.
- [ ] Weighted cross-entropy.
- [ ] Focal loss or another class-balanced loss.
- [ ] Complete training and validation curves for every seed.
- [ ] Checkpoint-selection rule documented before final testing.

### Required outputs

```text
outputs/training/<experiment_id>/config.yaml
outputs/training/<experiment_id>/history.csv
outputs/training/<experiment_id>/metrics.json
checkpoints/<experiment_id>/best_model.pth
figures/training_curves/<experiment_id>.pdf
```

## 3.5 Ensure deterministic and traceable execution

Record:

- Python version;
- PyTorch version;
- torchvision/timm version;
- CUDA and cuDNN versions;
- GPU/CPU model;
- random seeds;
- deterministic backend settings;
- dataset manifest hash;
- Git commit;
- checkpoint SHA-256 hash.

## Phase 3 completion gate

- [ ] Stage 2 retrained from clean grouped manifests.
- [ ] At least five same-data seeds.
- [ ] Seed and group-resampling effects separated.
- [ ] Training schedule validated.
- [ ] Second backbone added or single-backbone scope clearly bounded.
- [ ] Every checkpoint and metric is traceable.

---

# Phase 4 — Quantitative XAI Methodology

**Priority:** Major  
**Status:** New XAI analysis required after retraining

## 4.1 Regenerate all explanations

Generate all XAI maps from the final Stage 2 checkpoints:

- Grad-CAM;
- Grad-CAM++;
- Saliency Maps;
- Eigen-CAM;
- Score-CAM.

Do not reuse maps generated from Stage 1 models.

## 4.2 Define the explanation target

For every experiment specify whether the map explains:

- the predicted class;
- the true class;
- both predicted and true classes for errors;
- or all classes.

Recommended:

- correct predictions: predicted class;
- errors: both predicted and true class;
- method comparison: same predefined target rule for every method.

## 4.3 Document heatmap processing

Record:

- target layer;
- raw activation dimensions;
- ReLU use;
- resizing method;
- smoothing;
- normalization rule;
- zero-map handling;
- image overlay method;
- colour-map choice;
- whether maps are normalized per image or globally.

## 4.4 Use multiple spatial-agreement metrics

Cosine similarity alone is insufficient.

Recommended minimum battery:

1. Cosine similarity.
2. Structural Similarity Index (SSIM).
3. Spearman rank correlation.
4. Top-decile intersection-over-union.
5. Center-of-mass distance.
6. Optional Earth Mover’s Distance or another spatial distribution metric.

Report continuous scores rather than relying primarily on an arbitrary threshold.

## 4.5 Conduct threshold sensitivity

If a divergence threshold is retained, report results at:

```text
0.60, 0.70, 0.80, 0.90
```

The threshold must be labelled exploratory unless empirically calibrated.

## 4.6 Improve faithfulness testing

Deletion/insertion can create unrealistic ultrasound images.

### Required controls

Use at least two perturbation baselines, such as:

- blurred replacement;
- constant or mean-intensity replacement;
- optional noise or inpainting baseline.

Report whether method ranking changes across baselines.

Allowed wording:

> Perturbation-based sensitivity to the model output under the selected protocol.

Avoid:

> The explanation proves causal pathology.

## 4.7 Improve stability testing

Do not call transforms universally “clinically invariant” without clinical validation.

Use:

> small transformations expected to preserve the label.

Recommended transformations:

- small translation without anatomy removal;
- controlled brightness/contrast change;
- mild noise;
- optional small rotation.

Report transform magnitude and confirm that the model prediction remains unchanged for the evaluated sample.

## 4.8 Conduct model-randomization sanity checks

Recommended sequence:

- trained model;
- randomize classification head;
- randomize final residual block;
- progressively randomize deeper blocks;
- fully randomized model.

A reliable attribution method should change as learned parameters are destroyed.

## 4.9 Rename the border experiment correctly

A border-crop or margin test evaluates only:

> margin dependence

It is not a complete shortcut audit.

Use separate terminology:

- **margin-dependence test**;
- **group-associated shortcut diagnostic**;
- **near-duplicate audit**.

## 4.10 Prevent cherry-picking

Predefine qualitative image selection:

- fixed random seed;
- median-consistency sample;
- highest and lowest consistency sample;
- predefined correct/error cases;
- one sample per class selected by a documented rule.

Publish the full selection manifest.

## 4.11 Add clinician localization when possible

Without clinician-defined regions of interest, the paper cannot claim anatomical correctness.

Recommended future study:

- 100–200 images;
- one or two qualified radiologists/ultrasound experts;
- gallbladder/lesion ROI or bounding-box annotation;
- inter-rater agreement;
- pointing-game, IoU, Dice, or overlap evaluation;
- blinded rating of explanation usefulness.

This is required only for anatomical or clinical-localization claims, not for a carefully bounded model-behaviour reliability paper.

## Phase 4 completion gate

- [ ] XAI regenerated from final Stage 2 checkpoints.
- [ ] Target-class rule fixed.
- [ ] Multiple spatial-agreement metrics.
- [ ] Threshold sensitivity.
- [ ] Two perturbation baselines.
- [ ] Stability and sanity checks.
- [ ] Fixed qualitative sample manifest.
- [ ] Claims limited to model behaviour unless clinician ROI evidence exists.

---

# Phase 5 — Statistical Analysis and Uncertainty

**Priority:** Major submission blocker  
**Status:** New analysis required

## Required classification metrics

Report:

- accuracy;
- balanced accuracy;
- macro-F1;
- per-class F1;
- macro ROC-AUC;
- per-class ROC-AUC;
- macro PR-AUC;
- per-class PR-AUC;
- precision;
- sensitivity/recall;
- specificity;
- class support;
- confusion matrix.

For Carcinoma report separately:

- sensitivity;
- false-negative count;
- false-negative destination classes;
- confidence distribution;
- confidence interval.

## Group-aware confidence intervals

Because images within groups are correlated, prefer:

- group bootstrap;
- cluster bootstrap;
- repeated group split distribution.

Do not rely only on an image-level bootstrap.

## Separate uncertainty sources

Report:

```text
training-seed variability
group-split variability
group-bootstrap confidence interval
```

Do not collapse all uncertainty into one standard deviation without explanation.

## XAI statistical comparisons

For paired method comparisons:

- use the same images for every method;
- report mean paired differences;
- report 95% confidence intervals;
- report effect sizes;
- correct for multiple comparisons;
- consider Holm or false-discovery-rate correction.

## Calibration

Use:

```text
grouped training set → model fitting
grouped calibration set → temperature fitting
locked grouped test set → ECE and Brier evaluation
```

Report:

- pre-calibration ECE;
- post-calibration ECE;
- pre-calibration Brier score;
- post-calibration Brier score;
- reliability diagram;
- fitted temperature.

## Phase 5 completion gate

- [ ] Balanced accuracy and PR-AUC included.
- [ ] Group-aware confidence intervals.
- [ ] Seed and split uncertainty separated.
- [ ] Corrected paired XAI tests.
- [ ] Independent grouped calibration and test evaluation.
- [ ] Carcinoma-specific safety analysis.

---

# Phase 6 — Reorganize the Results

**Priority:** Major  
**Status:** Resolved structurally; update again after new experiments

## Required result order

### 6.1 Dataset and grouping audit

- released split overlap;
- pHash threshold sensitivity;
- connected components;
- cross-label audit;
- filename-prefix overlap;
- final grouped partition statistics.

### 6.2 Primary Stage 2 classification results

- repeated grouped evaluation;
- multi-seed mean;
- group-aware confidence intervals;
- per-class metrics;
- confusion matrix;
- ROC and PR curves;
- calibration.

### 6.3 Stage 2 error analysis

- high-confidence errors;
- Carcinoma false negatives;
- class-confusion patterns;
- error taxonomy.

### 6.4 Group-distinguishability diagnostic

Clearly label this as a diagnostic experiment, not the main disease-classification result.

### 6.5 Stage 2 XAI reliability

- cross-seed agreement;
- spatial agreement;
- faithfulness;
- stability;
- sanity checks;
- margin dependence;
- qualitative cases selected by fixed rules.

### 6.6 Stage 1 versus Stage 2 comparison

Use Stage 1 only to demonstrate:

- performance inflation;
- persistence of group correlation after duplicate control;
- change in XAI metrics under stricter splitting.

Move large Stage 1 galleries to supplementary material.

## Phase 6 completion gate

- [ ] Stage 2 is visually and narratively dominant.
- [ ] Stage 1 is always labelled shortcut-affected or group-confounded.
- [ ] Diagnostic experiments are separated from primary validation.
- [ ] Results report observations; interpretation is mainly in Discussion.

---

# Phase 7 — Clinical and Interpretive Claims

**Priority:** Major  
**Status:** Claim narrowing required

## Do not claim

- correct lesion localization;
- correct anatomical reasoning;
- radiologist-equivalent interpretation;
- causal pathology evidence;
- clinical decision-support readiness;
- external generalization;
- patient-independent validation;
- scanner-independent robustness.

## Preferred language

Instead of:

> The model attends to the correct anatomy.

Use:

> The attribution is visually concentrated in the central ultrasound field or apparent gallbladder region, but no clinician-defined localization reference is available.

Instead of:

> The batch signal is non-anatomical.

Use:

> The images contain learnable filename-prefix-group-associated information; its physical source remains undetermined.

Instead of:

> Grad-CAM++ is the best XAI method.

Use:

> Grad-CAM++ achieved the highest agreement under the reported ResNet-50, target-layer, sample, normalization, and metric protocol.

## Clinical review requirements

For stronger clinical interpretation:

- [ ] Radiologist review of confusion-pair explanations.
- [ ] Expert review of heatmap plausibility.
- [ ] Clinician-defined ROI annotations.
- [ ] Inter-rater agreement.
- [ ] External patient/site/scanner validation.

## Phase 7 completion gate

- [ ] Every clinical interpretation is cited or identified as a hypothesis.
- [ ] No localization claim without ROI evidence.
- [ ] No deployment claim.
- [ ] Model limitations are explicit.
- [ ] Carcinoma errors are discussed carefully.

---

# Phase 8 — Literature Review and Novelty

**Priority:** Major  
**Status:** Rewriting completed; reproducible search still required for strong novelty claims

## Organize related work into three groups

1. Gallbladder-ultrasound classification.
2. Quantitative XAI reliability and attribution evaluation.
3. Data leakage, grouped splitting, video-frame redundancy, and shortcut learning in medical imaging.

Do not mix foundational XAI method papers with gallbladder classifiers in a direct superiority table.

## Reproducible search protocol

Record:

- databases searched;
- exact search date;
- complete search strings;
- years covered;
- inclusion criteria;
- exclusion criteria;
- language restrictions;
- deduplication process;
- screening count;
- final included papers.

Recommended databases:

- IEEE Xplore;
- PubMed;
- Scopus or Web of Science where available;
- Google Scholar for supplementary snowballing.

## Novelty language

Use:

> To the best of our structured search, we identified no prior gallbladder-ultrasound study combining the selected leakage audit and quantitative XAI reliability battery.

Avoid:

> This is the first study ever.

## Phase 8 completion gate

- [ ] Reproducible search appendix.
- [ ] Qualified novelty claim.
- [ ] Fair study-comparison table.
- [ ] Strong leakage/group-splitting literature coverage.
- [ ] Every reference verified against the original paper.

---

# Phase 9 — Section-by-Section Rewrite

**Priority:** Major  
**Status:** Rewrite again after the experiments are frozen

## Recommended writing order

1. Methods.
2. Tables and figures.
3. Results.
4. Discussion.
5. Introduction.
6. Abstract.
7. Title and keywords.
8. Supplementary material.

This prevents the introduction and abstract from promising results that the completed experiments do not support.

## 9.1 Title

Must reflect:

- grouping/leakage;
- shortcut learning;
- bounded XAI reliability;
- gallbladder ultrasound classification.

## 9.2 Abstract

Use this structure:

1. Background problem.
2. Objective.
3. Dataset and grouping audit.
4. Model and XAI methods.
5. Primary Stage 2 multi-seed result.
6. Main XAI reliability result.
7. Interpretation.
8. Principal limitations.

Do not mix preliminary and final values.

## 9.3 Keywords

Use approximately 6–8 focused terms:

```text
Gallbladder ultrasound
Explainable artificial intelligence
Data leakage
Shortcut learning
Grouped validation
Grad-CAM
Deep learning
Internal validation
```

## 9.4 Introduction

Recommended flow:

1. Clinical and imaging context.
2. Limits of accuracy-only evaluation.
3. Risks of correlated-frame and group leakage.
4. Limits of visual heatmap inspection.
5. Research gap.
6. Research questions.
7. Four contributions.
8. Scope boundary.

## 9.5 Related Work

Suggested subsections:

- Gallbladder ultrasound AI.
- Quantitative evaluation of post-hoc explanations.
- Dataset leakage and shortcut learning in medical imaging.
- Summary of the unresolved gap.

## 9.6 Materials and Methods

Recommended subsections:

1. Dataset and label taxonomy.
2. Filename structure and grouping assumption.
3. Near-duplicate audit.
4. Grouped partition design.
5. Preprocessing and augmentation.
6. Model architectures.
7. Training and checkpoint selection.
8. XAI methods.
9. XAI reliability metrics.
10. Statistical analysis.
11. Calibration.
12. Reproducibility and software environment.

## 9.7 Results

Follow Phase 6 exactly.

## 9.8 Discussion

Recommended order:

1. Principal findings.
2. Why the released and Stage 1 results were inflated.
3. Meaning of Stage 2 generalization performance.
4. Reliable explanation of an unreliable or weakly generalizing classifier.
5. Comparison among XAI methods.
6. Comparison with prior literature.
7. Clinical implications and non-implications.
8. Limitations.
9. Next validation steps.

## 9.9 Limitations

Include at least:

- inferred rather than verified group identity;
- possible patient overlap across groups;
- missing hospital/scanner metadata;
- pHash threshold dependence;
- connected-component chaining;
- cross-label components;
- one or limited number of architectures;
- limited XAI sample sizes;
- absence of clinician ROI;
- internal validation only;
- possible residual correlation in diagnostic experiments;
- limited seed and group-split repetitions where applicable.

## 9.10 Conclusion

State only what the evidence demonstrates:

- released-split performance was not stable under stricter group separation;
- explanation agreement can remain high despite weak group generalization;
- XAI reliability metrics describe model behaviour, not clinical correctness;
- future validation needs patient/site/scanner metadata and clinician localization.

---

# Phase 10 — IEEE Layout, Tables, Figures, and Equations

**Priority:** Major presentation requirement  
**Status:** Use the VS Code workflow below

## Required IEEE structure

Typical journal structure:

```text
Title
Authors and affiliations
Abstract
Index Terms
I. Introduction
II. Background and Related Work
III. Materials and Methods
IV. Results
V. Discussion
VI. Limitations
VII. Conclusion
Acknowledgment, where applicable
CRediT Author Contributions
Funding
Conflict of Interest
Ethics Statement
Data and Code Availability
References
Supplementary Material statement
```

Adjust declarations to the selected journal's instructions.

## Tables

Required main tables may include:

1. Dataset image and group distribution.
2. pHash threshold-sensitivity audit.
3. Final grouped partition statistics.
4. Model and training configuration.
5. Multi-seed and repeated-group performance.
6. Per-class Stage 2 performance.
7. Calibration results.
8. XAI agreement and reliability metrics.
9. Paired method-comparison statistics.
10. Claim-boundary or limitations summary, if appropriate.

Use:

- automatic numbering;
- consistent decimal precision;
- confidence intervals;
- class support;
- concise captions;
- explanatory footnotes.

## Figures

Recommended main figures:

1. Complete leakage-audit and evaluation pipeline.
2. Performance progression across split definitions.
3. pHash threshold sensitivity.
4. Group-count and image-count distribution.
5. Stage 2 confusion matrix.
6. ROC and precision-recall curves.
7. Calibration reliability diagram.
8. Group-distinguishability diagnostic.
9. Cross-method XAI comparison using fixed sample selection.
10. XAI consistency/faithfulness/stability/sanity summary.

Move large galleries to supplementary material.

## Equations

For every equation:

- define all symbols;
- state tensor dimensions;
- state normalization;
- number only equations referenced later;
- avoid detached values;
- verify the compiled PDF at 100% and 200% zoom.

---

# Phase 11 — Reproducibility, Availability, Ethics, and Authorship

**Priority:** Submission blocker  
**Status:** External evidence and packaging required

## Required repository package

```text
README.md
LICENSE
CITATION.cff
environment.yml or requirements-lock.txt
Dockerfile, optional but recommended
configs/
data/manifests/
src/
scripts/
tests/
outputs/frozen_results/
figures/
paper/
supplementary/
```

## Required reproducibility items

- [ ] Anonymous review repository or public URL.
- [ ] Permanent archive plan, such as Zenodo after acceptance.
- [ ] Exact split manifests.
- [ ] Prefix-group IDs.
- [ ] pHash values and component IDs.
- [ ] Dataset and manifest checksums.
- [ ] Environment lock.
- [ ] Hardware details.
- [ ] Training configuration files.
- [ ] Seed files.
- [ ] Checkpoint hashes.
- [ ] Prediction files.
- [ ] One command that regenerates every table and figure.
- [ ] Tests for split overlap and leakage.

## Ethics and licence

Confirm:

- public dataset licence;
- permission to reproduce example images;
- whether derived manifests may be distributed;
- original dataset ethics statement;
- whether DIU requires a formal exemption or confirmation;
- exact target-journal ethics wording.

Do not state that ethical approval was unnecessary without institutional confirmation.

## Authorship

Confirm the CRediT roles reflect actual work:

- Conceptualization;
- Methodology;
- Software;
- Validation;
- Formal analysis;
- Investigation;
- Data curation;
- Visualization;
- Writing—original draft;
- Writing—review and editing;
- Supervision, where applicable.

---

# Phase 12 — Final Claim and Submission Audit

**Priority:** Final submission gate

## Claim–evidence matrix

Create:

```text
paper/claim_evidence_matrix.md
```

Recommended columns:

| Claim | Evidence file | Allowed wording | Limitation | Manuscript location |
|---|---|---|---|---|

Examples:

| Claim | Allowed wording | Boundary |
|---|---|---|
| Near-duplicate overlap exists | Extensive detected overlap under the selected pHash rule | Threshold sensitivity and manual review required |
| Performance changes after prefix-group separation | Performance does not transfer to unseen filename-prefix groups | Prefixes are not verified patient/site IDs |
| Group-associated information is learnable | Strong within-class group distinguishability under the corrected diagnostic | Physical mechanism unknown |
| Grad-CAM++ has high agreement | Highest agreement under the specified ResNet-50 protocol | Not universal across models or clinical tasks |
| Maps influence model output | Perturbation-sensitive under the selected protocol | Not causal pathology or clinician-valid localization |
| Model is clinically ready | **Do not claim** | External patient/site validation and clinical study required |

## Search the final manuscript for overclaims

Remove or qualify:

```text
leakage-free
patient-independent
multicentre validation
clinically validated
correct anatomy
causal explanation
ready for deployment
universal best method
first-ever study
confirmed non-anatomical shortcut
```

## Final submission checklist

- [ ] Target journal selected.
- [ ] Correct journal-specific IEEE template used.
- [ ] Page/word limits checked.
- [ ] Anonymous-review rules followed.
- [ ] Figures meet resolution and font-size requirements.
- [ ] Supplementary files comply with size rules.
- [ ] Every number is traceable.
- [ ] Every citation is verified.
- [ ] Code/manifests are accessible to reviewers.
- [ ] Ethics and licence wording confirmed.
- [ ] Two expert reviews completed:
  - machine-learning/methodology reviewer;
  - radiologist or ultrasound-domain reviewer.

---

# 4. VS Code Toolchain for Proper IEEE Paper Structure

## 4.1 Core software

### Required

1. **Visual Studio Code**
2. **LaTeX Workshop** VS Code extension
3. **TeX Live** LaTeX distribution  
   MiKTeX can also work, but the default `latexmk` workflow requires Perl.
4. **Git**
5. **Zotero**
6. **Better BibTeX for Zotero**

### Recommended VS Code extensions

- **LaTeX Workshop** — compile, preview, SyncTeX, navigation, diagnostics.
- **LTeX+** — grammar and language checking for LaTeX text.
- **Code Spell Checker** — terminology and spelling checks.
- **GitLens** — detailed Git history and contribution tracking.
- **Todo Tree** — track `TODO`, `FIXME`, and `VERIFY` items.
- **YAML** — configuration-file validation.
- **Python** — scripts for metrics, tables, and figures.
- **Jupyter** — exploratory analysis only; final results should be generated by scripts.
- **EditorConfig** — consistent formatting across collaborators.

Use AI coding assistants only for scaffolding, refactoring, tests, and documentation. Never allow them to invent data, results, references, or statistical conclusions.

## 4.2 Install LaTeX Workshop

Open VS Code Quick Open:

```text
Ctrl+P
```

Run:

```text
ext install latex-workshop
```

Then ensure the LaTeX distribution binaries are available in the system `PATH`.

The default LaTeX Workshop build uses `latexmk`.

## 4.3 Recommended project folder structure

```text
uidatagb-paper/
├── main.tex
├── IEEEtran.cls
├── references.bib
├── README.md
├── CITATION.cff
├── Makefile
├── latexmkrc
├── .gitignore
├── .editorconfig
│
├── .vscode/
│   ├── settings.json
│   ├── extensions.json
│   └── tasks.json
│
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_materials_methods.tex
│   ├── 04_results.tex
│   ├── 05_discussion.tex
│   ├── 06_limitations.tex
│   ├── 07_conclusion.tex
│   └── 08_declarations.tex
│
├── figures/
│   ├── pipeline.pdf
│   ├── split_progression.pdf
│   ├── phash_sensitivity.pdf
│   ├── stage2_confusion_matrix.pdf
│   ├── stage2_roc_pr.pdf
│   ├── calibration.pdf
│   ├── xai_comparison.pdf
│   └── xai_reliability.pdf
│
├── tables/
│   ├── dataset_groups.tex
│   ├── training_config.tex
│   ├── stage2_metrics.tex
│   ├── per_class_metrics.tex
│   ├── calibration_metrics.tex
│   └── xai_metrics.tex
│
├── supplementary/
│   ├── supplementary.tex
│   ├── cross_label_audit.tex
│   ├── additional_curves.tex
│   └── full_xai_gallery.tex
│
├── configs/
│   ├── stage2_resnet50.yaml
│   ├── stage2_second_backbone.yaml
│   └── xai_evaluation.yaml
│
├── data/
│   └── manifests/
│       ├── train_groups.csv
│       ├── development_groups.csv
│       ├── calibration_groups.csv
│       ├── test_groups.csv
│       └── checksums.txt
│
├── scripts/
│   ├── validate_manifests.py
│   ├── train.py
│   ├── evaluate.py
│   ├── calibrate.py
│   ├── generate_xai.py
│   ├── evaluate_xai.py
│   ├── statistical_analysis.py
│   ├── build_tables.py
│   └── build_figures.py
│
├── outputs/
│   ├── frozen_results/
│   ├── predictions/
│   ├── metrics/
│   └── xai/
│
└── build/
```

Do not manually edit generated files inside `tables/`, `figures/`, or `outputs/` without updating the generation script.

## 4.4 Recommended `main.tex`

```latex
\documentclass[journal]{IEEEtran}

\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{cite}
\usepackage{url}
\usepackage{hyperref}
\usepackage{xcolor}

\title{When Explainable AI Explains a Shortcut:
Data Leakage and XAI Reliability in Gallbladder Ultrasound Classification}

\author{
Sayed Ifti Ahmed and Mst. Khadiza Akter Sammi%
\thanks{The authors are with the Department of Computer Science and Engineering,
Daffodil International University, Dhaka, Bangladesh.}
}

\begin{document}

\maketitle

\input{sections/00_abstract}

\begin{IEEEkeywords}
Gallbladder ultrasound, explainable artificial intelligence,
data leakage, shortcut learning, grouped validation, Grad-CAM,
deep learning, internal validation.
\end{IEEEkeywords}

\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_materials_methods}
\input{sections/04_results}
\input{sections/05_discussion}
\input{sections/06_limitations}
\input{sections/07_conclusion}
\input{sections/08_declarations}

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
```

## 4.5 Recommended `.vscode/settings.json`

```json
{
  "latex-workshop.latex.outDir": "%DIR%/build",
  "latex-workshop.latex.autoBuild.run": "onSave",
  "latex-workshop.view.pdf.viewer": "tab",
  "latex-workshop.synctex.afterBuild.enabled": true,
  "latex-workshop.latex.clean.enabled": true,
  "latex-workshop.message.error.show": true,
  "latex-workshop.message.warning.show": true,
  "editor.wordWrap": "on",
  "editor.formatOnSave": true,
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true
}
```

Use project-level settings so every collaborator builds the manuscript consistently.

## 4.6 Recommended `.vscode/extensions.json`

```json
{
  "recommendations": [
    "james-yu.latex-workshop",
    "valentjn.vscode-ltex",
    "streetsidesoftware.code-spell-checker",
    "eamodio.gitlens",
    "gruntfuggly.todo-tree",
    "ms-python.python",
    "ms-toolsai.jupyter",
    "redhat.vscode-yaml",
    "editorconfig.editorconfig"
  ]
}
```

Extension identifiers can change; verify them in the VS Code Marketplace before committing the file.

## 4.7 Build commands

### Build the PDF

```text
Ctrl+Alt+B
```

Or use the Command Palette:

```text
LaTeX Workshop: Build LaTeX project
```

### View the PDF

```text
LaTeX Workshop: View LaTeX PDF
```

### Clean temporary files

```text
LaTeX Workshop: Clean up auxiliary files
```

## 4.8 Reference management with Zotero

Recommended workflow:

1. Create a Zotero collection for this paper.
2. Install Better BibTeX.
3. Export the collection as Better BibTeX.
4. Enable **Keep updated**.
5. Save the file as:

```text
references.bib
```

6. Cite in LaTeX:

```latex
\cite{selvarajuGradCAM2017}
```

7. Never manually invent a BibTeX record.
8. Verify title, authors, venue, year, pages, DOI, and article number against the original source.

## 4.9 Automated manuscript data flow

Recommended principle:

```text
analysis outputs
      ↓
Python table/figure scripts
      ↓
tables/*.tex and figures/*.pdf
      ↓
main.tex
      ↓
compiled manuscript
```

Example:

```python
# scripts/build_tables.py
# Reads frozen CSV/JSON results and writes LaTeX tables.
```

Then include:

```latex
\input{tables/stage2_metrics}
```

This prevents transcription errors.

## 4.10 Git workflow

Recommended branches:

```text
main
experiment/stage2-retraining
experiment/phash-sensitivity
experiment/xai-reliability
paper/methods
paper/results
paper/final-submission
```

Commit messages:

```text
data: freeze grouped split manifests
train: add five-seed ResNet-50 Stage 2 run
xai: add SSIM and top-decile IoU
stats: add group-bootstrap confidence intervals
paper: revise Stage 2 results section
```

Tag the exact submission version:

```text
v1.0-submission
```

Record the Git commit in the reproducibility statement.

---

# 5. Recommended End-to-End Execution Order

```text
1. Verify filename-prefix meaning
2. Run pHash threshold and component audits
3. Freeze grouped train/development/calibration/test manifests
4. Validate manifests automatically
5. Retrain Stage 2 models with at least five same-data seeds
6. Repeat grouped splits or grouped cross-validation
7. Add second backbone or narrow architecture claims
8. Generate predictions and group-aware statistics
9. Fit calibration on grouped calibration data
10. Regenerate all XAI maps
11. Run multi-metric XAI reliability battery
12. Conduct corrected group-ID diagnostic
13. Freeze all tables and figures
14. Rewrite Methods and Results
15. Rewrite Discussion, Introduction, Abstract, and Title
16. Complete repository, ethics, licence, and authorship statements
17. Conduct final claim audit
18. Compile with IEEEtran in VS Code
19. Obtain methodology and clinical expert review
20. Submit only after every blocker is closed or explicitly bounded
```

---

# 6. Minimum Submission Blockers

The paper should not be submitted with strong clinical or generalization claims until these items are complete:

- [ ] Filename-prefix meaning verified or explicitly bounded.
- [ ] pHash threshold sensitivity and manual duplicate audit.
- [ ] Cross-label component audit.
- [ ] Grouped train/development/calibration/test design.
- [ ] Stage 2 retraining from correct manifests.
- [ ] At least five same-data seeds.
- [ ] Repeated grouped evaluation or grouped cross-validation.
- [ ] Group-aware confidence intervals.
- [ ] Balanced accuracy and PR-AUC.
- [ ] Clean group-ID diagnostic after duplicate control.
- [ ] XAI regenerated from final checkpoints.
- [ ] Multiple XAI spatial-agreement metrics.
- [ ] Independent calibration evaluation.
- [ ] Repository, manifests, environment lock, and checkpoint hashes.
- [ ] Verified ethics, licence, and authorship statements.

---

# 7. Final Quality Target

After completing the phases, the paper should support the following bounded conclusion:

> The released UIdataGB evaluation protocol contains extensive image correlation under the selected perceptual-hash audit, and model performance does not transfer reliably to unseen filename-prefix groups under the stricter internal-validation protocol. Some XAI methods may produce comparatively stable and perturbation-sensitive attribution maps even when classification generalization is weak. These properties characterize the behaviour of the trained model under the reported protocol; they do not establish clinician-valid localization, patient-level generalization, or clinical deployment readiness.
