# Strategic plan: countering the bottlenecks we've actually hit

This is not a redo of `UIdataGB_Journal_Revision_Phases_and_VSCode_Setup.md`
(the content-revision plan). This is about the things that have slowed the
*process* down or put the *work itself* at risk of being wrong without
anyone noticing, across every session on this project so far. Each item:
what happened, why it happened, and the concrete countermeasure going
forward.

---

## Category A — Research-integrity bottlenecks (highest priority)

These are the ones that matter most: if unaddressed, the paper is wrong,
not just poorly worded.

### A1. Two divergent paper versions went out of sync
**What happened:** `uidatagb-paper/`'s own README names `UIdataGB_Paper.html`
as content source-of-truth and says to re-sync `.tex` after every HTML
revision. That sync never happened. Several turns of LaTeX polish (Figure 1
alignment, abstract rewrite, audit-response wording fixes) were spent on a
`.tex` snapshot that was missing the entire Stage 2 finding.
**Root cause:** no enforced single source of truth; two formats, one
manual, unversioned sync step.
**Countermeasure:**
- Pick ONE file format as the actual working document going forward. Given
  the paper is close to submission-ready structurally, recommend making
  **`.tex` the new source of truth** and treating the HTML as a stale
  export, OR formally retire the HTML now that Stage 2 content will be
  written directly into `.tex`.
- Before any further prose editing session, run a one-line diff check:
  `grep -c "Stage 2\|batch.id\|29.59\|34.97" sections/*.tex` against the
  same grep on the HTML, to catch drift immediately rather than after
  several turns of work on the wrong file.

### A2. The consistency metric may not be measuring anything
**What happened:** External review pointed out Grad-CAM-family heatmaps are
non-negative (ReLU) and min-max normalized, which mathematically inflates
cosine similarity between unrelated maps — and no baseline exists showing
0.94 is meaningfully above that floor.
**Root cause:** the null-control experiment was never built; the metric was
trusted because it "looked reasonable," not because it was checked against
a floor.
**Countermeasure:** already addressed — `kaggle_retrain/uidatagb_full_retrain.py`
Sections 5–6 build both null controls (untrained-model floor, random-heatmap
floor). **This must be the first thing read once the Kaggle run finishes,
before any other result is trusted** — if either floor sits near 0.9, every
downstream consistency claim in the paper needs to be rewritten as
"undetermined," not just re-worded.

### A3. "Replicate" models conflated three different effects
**What happened:** M1/M2/M3 were trained on disjoint 2,749/2,089/2,661-image
subsets, not the same data with different seeds — confirmed in the training
script's own docstring. Every consistency number in the old tables measured
data-sensitivity, seed-sensitivity, and class-composition-shift all at once,
mislabeled as pure reproducibility.
**Root cause:** methodology was designed once, early, and never revisited
even as later Stage 2 work reused the same disjoint-subset pattern.
**Countermeasure:** already addressed — the Kaggle script trains all 4
models (primary + 3 replicates) on the identical 70% training split, seed
varied only. Going forward: **any new "replicate" or "ablation" script must
state in its own header comment which single variable it isolates**, since
this exact bug was copy-pasted from Stage 1 into Stage 2 once already.

### A4. No held-out test set — every design decision saw the "validation" data
**What happened:** epoch count, pHash threshold, θ=0.70, subsample sizes
were all chosen while looking at performance on the same split used for
every reported number.
**Root cause:** 70/30 train/val was adopted early and never revisited as
more design decisions accumulated on top of it.
**Countermeasure:** already addressed — 70/20/10 split with the test set
touched exactly once, at the end, in the new Kaggle script. **Discipline
going forward:** once the test set is evaluated, no further tuning is
allowed to happen and then get re-evaluated on it. If a future change
requires re-tuning, that invalidates the test-set number and it must be
re-drawn or clearly flagged as no longer blind.

### A5. Numeric reporting bugs shipped silently (Table V/VI std column)
**What happened:** the reported std column was pooled per-image std, not
std-of-the-three-replicate-means, but the caption and prose read as if they
were the same number. This wasn't caught until an external review
hand-recomputed it.
**Root cause:** no self-check step compares a table's prose claims (e.g.,
"Grad-CAM is the most volatile") against the actual numbers backing them.
**Countermeasure:**
- The new Kaggle script now computes and labels BOTH quantities explicitly
  (`per_replicate_mean_std_of_means` vs `pooled_all_scores_std`) so this
  specific ambiguity can't recur silently.
- Going forward, adopt a **"recompute one number by hand per table before
  it ships"** habit — pick one cell in every new results table and verify
  it against the raw JSON before it goes into the manuscript. This is
  cheap and would have caught this bug immediately.

---

## Category B — Tooling / environment bottlenecks

These didn't corrupt results, but burned real turns.

### B1. MiKTeX/kpsewhich silently resolved `references.bib` to a bundled package
**What happened:** several rebuild cycles produced an empty bibliography
with zero warnings pointing at the actual cause, because kpsewhich resolved
the filename to an unrelated MiKTeX package instead of the local file.
**Countermeasure:** already fixed for both `.tex` projects — `.bib` is
copied into `build/` before each compile (`Makefile`/manual step documented
in each project's README). **Do not rename `.bib` files back to
`references.bib` in any new LaTeX project on this machine** without
re-applying the same copy-into-build workaround.

### B2. Broken Python venv (stale user path from a different machine)
**What happened:** `.venv` pointed at `C:\Users\ifti2\...`, which doesn't
exist on this machine, silently blocking any `pip`/`python` call until
diagnosed.
**Countermeasure:** already fixed — venv rebuilt fresh with the correct
Python 3.12.10 and the exact `requirements-lock.txt` pins. **Before
starting any new heavy compute task, run one cheap sanity check first**
(`python -c "import torch; print(torch.cuda.is_available())"`) rather than
discovering an environment problem mid-task.

### B3. Kaggle GPU quota (30 hrs/week) is a hard ceiling on this plan
**What happened:** not yet hit, but the full retrain (4 models × up to 20
epochs + 2 null controls) is estimated at several hours; running it more
than once due to a bug wastes quota that resets weekly.
**Countermeasure:**
- The script is resumable by design (every step checks for its own output
  file first) specifically so a bug or timeout mid-run doesn't require a
  full restart.
- **Before the first full Kaggle run, do a smoke test**: set `EPOCHS = 1`
  temporarily and run just the primary model end-to-end, confirm the
  output files look right, THEN switch back to the real `EPOCHS = 20` and
  launch the full run. This costs minutes, not hours, and catches path/
  config bugs before they cost real quota.
- If quota runs out mid-pipeline, the classification training and the XAI
  battery can be split across two separate weekly quota resets — already
  noted in `kaggle_retrain/README.md`.

### B4. Kaggle doesn't persist `/kaggle/working/` across sessions
**What happened:** not yet hit, but is a known Kaggle behavior that has
caused lost work in other projects.
**Countermeasure:** `README.md` already documents **Save Version → Save &
Run All** as the mandatory last step before closing any Kaggle session
that produced results worth keeping.

---

## Category C — Verification / trust bottlenecks

### C1. No public code repository — reviewers can't verify the leakage audit
**What happened:** the code-availability statement says a DOI will be
"assigned upon acceptance," meaning nothing is actually checkable right
now, which a reviewer explicitly flagged as blocking verification of the
paper's own strongest claim.
**Countermeasure:** create a private (or public, once ready) GitHub
repository for `fedgb/` now, not at acceptance. Doesn't need to be public
immediately — a private repo with a reviewer-accessible link satisfies most
venues' "available to reviewers" requirement well before public release.
This also directly satisfies the workflow checklist's "save code in Kaggle
notebooks or a private GitHub repository" instruction.

### C2. Conflicting numbers coexisted across files with no version discipline
**What happened:** 29.59% (single run) vs. 33.41%±2.52% (3-seed, per
`claim_evidence_matrix.md`) vs. 34.97%±2.79% (5-seed, current
`multiseed_summary.json`) — three real numbers from three real runs, but
nothing marked which one is current/canonical at a glance.
**Countermeasure:** once the Kaggle retrain produces the new canonical
numbers, **retire the older JSON files into an `outputs/archive/` folder**
rather than leaving multiple "current-looking" result files in the same
directory. Update `claim_evidence_matrix.md` in the same pass that new
numbers land — it should never describe a superseded run.

---

## Priority order

1. **A2 + A3 + A4** (null controls, real replicates, real test set) — these
   are already built into `kaggle_retrain/uidatagb_full_retrain.py`. Next
   action is simply: run it.
2. **B3 smoke test** — do this immediately before the real Kaggle run, costs
   minutes.
3. **A1** (pick one source-of-truth file) — decide before writing any new
   Stage 2 prose, or this bottleneck recurs immediately.
4. **A5 habit** (hand-check one number per table) — adopt starting with the
   very next table the Kaggle results produce.
5. **C1** (stand up the repo) — do this in parallel with the Kaggle run
   waiting; it's independent, cheap, and directly unblocks the reviewer's
   verification complaint.
6. **C2** (archive stale JSON, update claim-evidence matrix) — do this the
   moment the new Kaggle numbers land, before they get a chance to become
   a fourth conflicting figure.
