"""
compute_theta_sensitivity.py
-----------------------------
The overlap_all_methods.json file stores only per-(class,client,method)
aggregates (mean/std/flagged-count at the single fixed theta=0.70). To
report a sensitivity analysis of the divergence threshold theta, we need the
raw per-image cosine-overlap values so the flagged fraction can be recomputed
at several thresholds.

This script recomputes raw per-image O_k^c for Grad-CAM++ (the paper's primary
XAI method) — global FedProx model vs. each local-only client model — and
reports, for a sweep of theta values, the fraction of (image x client) pairs
flagged as explanation-divergent (O < theta), overall and per class.

Grad-CAM++ is used because it is cheap (no extra forward passes) and is the
method the paper foregrounds; the qualitative threshold behaviour is
representative across methods.

Run from fedgb/:
  python compute_theta_sensitivity.py \
      --global_ckpt ./outputs/checkpoints/round_010.pt \
      --local_dir   ./outputs/seeds/seed_42

Output: outputs/theta_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CLASS_NAMES, GBCUDataset, get_transforms
from gradcam import GradCAMPlusPlus, cosine_overlap
from model import build_model, get_target_layer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path):
    m = build_model(pretrained=False).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global_ckpt", default="./outputs/checkpoints/round_010.pt")
    ap.add_argument("--local_dir", default="./outputs/seeds/seed_42")
    ap.add_argument("--data_dir", default="./data/data")
    ap.add_argument("--num_clients", type=int, default=3)
    ap.add_argument("--out", default="./outputs/theta_sensitivity.json")
    ap.add_argument("--thetas", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85")
    args = ap.parse_args()

    thetas = [float(t) for t in args.thetas.split(",")]
    print(f"[info] device={DEVICE}  thetas={thetas}")

    g_model = load_model(args.global_ckpt)
    l_models = [load_model(os.path.join(args.local_dir, f"local_only_client{k}.pt"))
                for k in range(args.num_clients)]

    val = GBCUDataset(os.path.join(args.data_dir, "validation"),
                      transform=get_transforms(train=False))
    print(f"[info] |val|={len(val)}")

    g_cam = GradCAMPlusPlus(g_model, get_target_layer(g_model))
    l_cams = [GradCAMPlusPlus(m, get_target_layer(m)) for m in l_models]

    # raw O values, keyed by class then flat list across (image x client)
    raw = {c: [] for c in CLASS_NAMES}
    raw_all = []
    for idx in range(len(val)):
        tensor, label = val[idx]
        cls = CLASS_NAMES[label]
        hg, _, _ = g_cam.generate(tensor.unsqueeze(0))
        for lc in l_cams:
            hl, _, _ = lc.generate(tensor.unsqueeze(0))
            o = cosine_overlap(hg, hl)
            raw[cls].append(o)
            raw_all.append(o)
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(val)}")

    g_cam.remove_hooks()
    for lc in l_cams:
        lc.remove_hooks()

    raw_all = np.array(raw_all)
    result = {"method": "gradcam_pp", "n_pairs": int(raw_all.size),
              "overall_mean": float(raw_all.mean()),
              "overall_std": float(raw_all.std(ddof=1)),
              "thetas": thetas, "flagged_fraction_overall": {}, "flagged_fraction_per_class": {}}

    for t in thetas:
        result["flagged_fraction_overall"][f"{t:.2f}"] = float((raw_all < t).mean())

    for cls in CLASS_NAMES:
        arr = np.array(raw[cls])
        result["flagged_fraction_per_class"][cls] = {
            f"{t:.2f}": float((arr < t).mean()) for t in thetas
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[done] wrote {args.out}")
    print("theta  flagged%(overall)")
    for t in thetas:
        print(f"  {t:.2f}   {100*result['flagged_fraction_overall'][f'{t:.2f}']:.1f}%")


if __name__ == "__main__":
    main()
