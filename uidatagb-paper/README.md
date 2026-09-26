# UIdataGB paper — IEEEtran two-column LaTeX build

LaTeX source for "Quantitative Evaluation of Explainable AI Methods for
Gallbladder Diseases through Ultrasound Images," built per Phase 10 of
`../UIdataGB_Journal_Revision_Phases_and_VSCode_Setup.md`.

**Content source of truth:** `../UIdataGB_Paper.html`. The section files
here were generated from that HTML; if the HTML is revised further, re-sync
the affected `sections/*.tex` file rather than editing both independently.

**Data/results source of truth:** `../fedgb/outputs/frozen_results/results_master.csv`
and `../fedgb/paper/claim_evidence_matrix.md` (Phase 0/12).

## Build

Requires a LaTeX distribution (MiKTeX or TeX Live) with `latexmk` on `PATH`.

```
latexmk -pdf -outdir=build main.tex
```

or, from VS Code with the LaTeX Workshop extension installed, `Ctrl+Alt+B`.

## Structure

```
main.tex              entry point
sections/              00_abstract .. 08_declarations
figures/                copy PNGs from ../fedgb/outputs/figures/ here
tables/                 (currently inlined in sections/*.tex; split out if they grow)
references.bib          BibTeX, keys ref1..ref56 matching the HTML's [N] numbering
```

Figures are referenced by the section files but not yet copied in — each
`\includegraphics` has a `% TODO` comment naming its source path under
`../fedgb/outputs/figures/`. Copy the needed files into `figures/` before
the first full compile (see the TODOs `latexmk` will list as missing).
