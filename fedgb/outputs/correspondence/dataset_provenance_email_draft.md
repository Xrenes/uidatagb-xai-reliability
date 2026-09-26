# Draft — email to UIdataGB dataset creators/maintainers

**Status: DRAFT ONLY. Not sent. Needs a recipient address and your review before sending.**
Per Phase 2.1 of the revision plan, this cannot be resolved without the dataset
provider's confirmation — I am providing the draft, not sending it.

---

**Subject:** Question about UIdataGB filename structure (prefix/index/session meaning)

Dear UIdataGB dataset team,

We are preparing a manuscript ("Quantitative Evaluation of Explainable AI
Methods for Gallbladder Diseases through Ultrasound Images") that
uses the publicly released UIdataGB dataset for a 9-class gallbladder disease
classification and explainable-AI reliability study.

During our analysis we found that images sharing a filename prefix (e.g. the
"h14", "h10", ... pattern, or the letter+number+parenthetical-index pattern
seen across classes) are highly separable by a classifier even within a
single disease class, and that released train/validation splits place large
numbers of same-prefix images on both sides of the split. Before we
characterize this in the manuscript, we want to confirm what the filename
prefix actually denotes, since we do not want to mischaracterize it.

Specifically, could you confirm:

1. Does the filename prefix (or any other part of the filename) correspond to
   a unique patient, a unique imaging study/session, a unique source video,
   a specific scanner/device, or a specific hospital/site?
2. Were images extracted from ultrasound video as sequential frames? If so,
   what is the typical frame spacing or temporal window represented by
   images sharing a prefix?
3. Was the released train/validation split constructed with any
   patient-level, session-level, or site-level separation, or was it a
   simple random/stratified image-level split?
4. Do any two prefixes ever refer to the same underlying patient (e.g. a
   patient imaged at two different sessions with two different prefixes)?
5. Is there any metadata (patient ID, session ID, scanner model, hospital)
   associated with the release, even if not currently published, that could
   be shared under a data-use agreement for research purposes?

We are happy to share our current filename-grouping analysis in return if
useful. A written response (even informal) would let us cite it directly in
the manuscript's dataset description and avoid over- or under-stating what
the released split actually separates.

Thank you for making this dataset publicly available and for any guidance
you can offer.

Best regards,
Sayed Ifti Ahmed; Mst. Khadiza Akter Sammi
Department of Computer Science and Engineering, Daffodil International
University, Dhaka, Bangladesh

---

## Notes for follow-up if no reply

- Search the original dataset publication and any supplementary files for
  an explicit filename-convention description (Phase 2.1 checklist).
- Check for a GitHub/Kaggle/Zenodo dataset page issue tracker or contact
  form as an alternative channel.
- If no confirmation is obtainable before submission, keep using the
  bounded term "filename-prefix group" throughout the manuscript (already
  done in the current draft) and state explicitly in Limitations that group
  identity is inferred, not confirmed — this is already the paper's current
  posture and does not block submission on its own.
