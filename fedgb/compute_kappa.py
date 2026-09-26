"""
compute_kappa.py
----------------
Analyses returned radiologist ratings for the XAI-FedGB heatmap validation
study (see build_radiologist_study.py). Reports:

  - per-rater mean plausibility (1-5 Likert) overall and by class
  - inter-rater agreement:
      * Cohen's quadratic-weighted kappa on the 1-5 plausibility scale
      * Cohen's kappa on the binary "attends gallbladder region" judgement
  - correlation between mean plausibility and the model's own cosine-overlap
    flags / correctness (does low overlap or a wrong prediction coincide with
    low radiologist-rated plausibility?)

Usage (run from fedgb/):
  python compute_kappa.py \
      --raters outputs/radiologist_study/rater_A.csv,outputs/radiologist_study/rater_B.csv \
      --manifest outputs/radiologist_study/manifest.json

Each rater CSV is a filled copy of rating_template.csv.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from itertools import combinations

import numpy as np


def cohen_kappa(a, b, weights=None, n_cats=5):
    a = np.asarray(a); b = np.asarray(b)
    cats = list(range(1, n_cats + 1)) if weights else sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    O /= O.sum()
    r = O.sum(1); c = O.sum(0)
    E = np.outer(r, c)
    if weights == "quadratic":
        W = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                W[i, j] = ((cats[i] - cats[j]) ** 2) / ((k - 1) ** 2)
        po = 1 - (W * O).sum(); pe = 1 - (W * E).sum()
    else:
        po = np.trace(O); pe = np.trace(E)
    return (po - pe) / (1 - pe) if (1 - pe) else float("nan")


def load_rater(path):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rid = int(row["rating_id"])
            pl = row.get("plausibility_1to5", "").strip()
            gb = row.get("attends_gb_region_yn", "").strip().lower()
            out[rid] = {"pl": int(pl) if pl else None,
                        "gb": 1 if gb in ("y", "yes", "1", "true") else
                              (0 if gb in ("n", "no", "0", "false") else None)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", required=True, help="comma-separated rater CSVs")
    ap.add_argument("--manifest", default="outputs/radiologist_study/manifest.json")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    items = {m["rating_id"]: m for m in manifest["items"]}
    rater_paths = [p.strip() for p in args.raters.split(",") if p.strip()]
    raters = {os.path.basename(p): load_rater(p) for p in rater_paths}
    names = list(raters)
    ids = sorted(items)

    print(f"=== {len(rater_paths)} raters, {len(ids)} images ===\n")
    for nm, r in raters.items():
        pls = [r[i]["pl"] for i in ids if r.get(i, {}).get("pl")]
        print(f"{nm}: mean plausibility = {np.mean(pls):.2f} (n={len(pls)})")

    print("\n-- inter-rater agreement (pairwise) --")
    for a, b in combinations(names, 2):
        pa = [raters[a][i]["pl"] for i in ids if raters[a].get(i, {}).get("pl") and raters[b].get(i, {}).get("pl")]
        pb = [raters[b][i]["pl"] for i in ids if raters[a].get(i, {}).get("pl") and raters[b].get(i, {}).get("pl")]
        kw = cohen_kappa(pa, pb, weights="quadratic")
        ga = [raters[a][i]["gb"] for i in ids if raters[a].get(i, {}).get("gb") is not None and raters[b].get(i, {}).get("gb") is not None]
        gb_ = [raters[b][i]["gb"] for i in ids if raters[a].get(i, {}).get("gb") is not None and raters[b].get(i, {}).get("gb") is not None]
        kg = cohen_kappa(ga, gb_, weights=None) if ga else float("nan")
        print(f"  {a} vs {b}: quadratic-weighted kappa (plausibility) = {kw:.3f};  "
              f"kappa (attends-GB) = {kg:.3f}")

    # mean plausibility vs model correctness
    print("\n-- mean plausibility by model correctness --")
    mean_pl = {}
    for i in ids:
        vals = [raters[nm][i]["pl"] for nm in names if raters[nm].get(i, {}).get("pl")]
        if vals:
            mean_pl[i] = np.mean(vals)
    corr = [mean_pl[i] for i in mean_pl if items[i]["model_correct"]]
    wrong = [mean_pl[i] for i in mean_pl if not items[i]["model_correct"]]
    if corr:
        print(f"  correctly classified:   mean plausibility = {np.mean(corr):.2f} (n={len(corr)})")
    if wrong:
        print(f"  misclassified:          mean plausibility = {np.mean(wrong):.2f} (n={len(wrong)})")
    print("\n[note] Landis-Koch: kappa >0.60 substantial, >0.80 almost perfect.")


if __name__ == "__main__":
    main()
