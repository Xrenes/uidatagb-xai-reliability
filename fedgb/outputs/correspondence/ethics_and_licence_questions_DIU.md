# Draft — questions for DIU (institutional ethics/licence confirmation)

**Status: DRAFT ONLY. Needs to go to your department/institutional review
contact at Daffodil International University — I cannot obtain this
confirmation myself (Phase 11).**

## Questions to ask your department / IRB or research office

1. Does a study using only a pre-existing, publicly released, de-identified
   image dataset (UIdataGB) require formal DIU ethics committee review, or
   does it qualify for exemption as secondary analysis of public data?
   - If exemption applies, is a written exemption letter/certificate
     available for the manuscript's Ethics Statement?
2. Does DIU have a standard Ethics Statement wording for secondary-analysis
   studies that the target journal will accept, or does the journal need a
   specific IRB protocol number regardless?
3. Is there an institutional policy on redistributing derived artifacts
   (train/val manifests, pHash cluster IDs, filename-prefix group IDs)
   alongside the public dataset in a code/data repository? Any restriction
   from the dataset's own license needs to be checked separately (see below).

## Separate item: dataset license check (not a DIU question)

- [ ] Locate the UIdataGB release's license terms (repository/Kaggle/Zenodo
      page) and confirm: (a) redistribution of the dataset itself is not
      required (paper links to original source), (b) redistribution of
      *derived* manifests (filenames + cluster/group IDs, no raw pixel data)
      is permitted, (c) reproduction of a small number of example images in
      the manuscript/supplementary figures is permitted or falls under
      fair-use/academic-use terms.
- [ ] If the license is silent or ambiguous, contact the dataset maintainers
      (see dataset_provenance_email_draft.md) and ask explicitly about
      derived-manifest redistribution and figure reproduction.

## What NOT to do

Per the revision plan (Phase 11), do not state in the manuscript that
"ethical approval was unnecessary" without an actual institutional
confirmation in hand — leave the Ethics Statement placeholder marked
`[CONFIRM WITH DIU]` until one of the above questions is answered.
