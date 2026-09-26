# Phase 8 status — reproducible literature search

**I did not run a real database search (IEEE Xplore / PubMed / Scopus), and
will not fabricate one.** Phase 8 exists specifically to replace
"first-ever study" language with a claim traceable to an actual, dated,
repeatable search — inventing search dates, strings, and hit counts would
recreate exactly the kind of untraceable claim Phase 0/12 are designed to
eliminate.

## What's already correctly in place (no change needed)

The manuscript's current novelty language is already appropriately hedged
and does NOT claim to be "the first ever" study:

- "no prior work on this dataset is known to check for near-duplicate
  leakage" (line 868-869)
- "No prior published study... is known to have audited this dataset for
  near-duplicate frame leakage" (line 1006-1007)
- "no prior published study is known to have audited UIdataGB for..."
  (line 1109)
- "None is known to compare Grad-CAM, [...] CAM variant qualitatively"
  (line 1113)

This phrasing already matches Phase 8's recommended pattern ("to the best
of our... search, we identified no prior study combining X and Y") in
substance, just without a literal dated search log backing it.

## What would still be needed to fully close Phase 8

To make the "no prior work is known to" claims fully defensible under
Phase 8's bar, you (not me) would need to actually run and log:

1. Searches in IEEE Xplore, PubMed, and Scopus or Web of Science for terms
   combining "UIdataGB" / "gallbladder ultrasound" with "data leakage" /
   "near-duplicate" / "shortcut learning" / "explainable AI reliability".
2. The exact search date, search strings, years covered, and inclusion/
   exclusion criteria.
3. A count of papers screened and included, ideally in a small PRISMA-style
   flow note in Supplementary Material.

I can draft the search-string list and a template log/table for you to fill
in as you actually run each search (this is mechanical once you have
database access), but I should not populate result counts or "no prior work
found" conclusions myself, since I have no way to actually query IEEE
Xplore/PubMed/Scopus from here. Say the word if you want that template.
