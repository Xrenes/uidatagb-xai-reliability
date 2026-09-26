"""
compute_gradcam_overlap.py
--------------------------
Computes the quantitative Grad-CAM "cosine overlap" statistic O_k^c
required by Table VIII of the FedGB paper:

    O_k^c = mean_{x in V_k^c} cosine(H_global(x), H_local_k(x))

where:
  - V_k^c   = validation images of class c that belong to client k's
              held-in distribution (here all validation images of class c).
  - H_global(x) = Grad-CAM heatmap from the final-round FedProx global model.
  - H_local_k(x) = Grad-CAM heatmap from client k's local-only model
                    (saved by run_seeds.py --methods local_only).

We also flag images where O_k^c < theta (default theta=0.70) as
"divergent" attention -- the proposed screening rule the paper discusses.

Inputs (expected):
  outputs/checkpoints/round_010.pt                       (global FedProx)
  outputs/seeds/seed_42/local_only_client{0,1,2}.pt      (local-only per client)
  data/data/validation/{nml,bmt,stn,abn,malg}/*.png      (val images)

Output:
  outputs/gradcam_overlap.json   - consumed by Table VIII

Usage (run from fedgb/):
  python compute_gradcam_overlap.py \
         --global_ckpt ./outputs/checkpoints/round_010.pt \
         --local_dir   ./outputs/seeds/seed_42 \
         --data_dir    ./data/data \
         --theta 0.70
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CLASS_NAMES, GBCUDataset, get_transforms
from gradcam import GradCAM
from model import build_model, get_target_layer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str):
    model = build_model(pretrained=False).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (na * nb))


def compute_heatmap(gcam: GradCAM, tensor: torch.Tensor) -> np.ndarray:
    heatmap, _, _ = gcam.generate(tensor.unsqueeze(0))
    return heatmap


def overlap_per_class_client(
    global_model, client_models: List, val_dataset, theta: float,
) -> Dict:
    target_layer_g = get_target_layer(global_model)
    gcam_global = GradCAM(global_model, target_layer_g)
    gcam_locals = []
    for cm in client_models:
        gcam_locals.append(GradCAM(cm, get_target_layer(cm)))

    n_classes = len(CLASS_NAMES)
    per_image: List[dict] = []
    # Per (class, client): list of cosine values
    sums: Dict[str, Dict[int, List[float]]] = {
        c: {k: [] for k in range(len(client_models))} for c in CLASS_NAMES
    }

    for idx in range(len(val_dataset)):
        tensor, label = val_dataset[idx]
        cls = CLASS_NAMES[label]
        h_g = compute_heatmap(gcam_global, tensor)
        per_image_record = {
            "index":  int(idx),
            "true":   cls,
            "client_cosines": {},
            "flagged_divergent_for": [],
        }
        for k, gcam_l in enumerate(gcam_locals):
            h_l = compute_heatmap(gcam_l, tensor)
            o = cosine(h_g, h_l)
            sums[cls][k].append(o)
            per_image_record["client_cosines"][f"client_{k}"] = round(o, 4)
            if o < theta:
                per_image_record["flagged_divergent_for"].append(f"client_{k}")
        per_image.append(per_image_record)
        if (idx + 1) % 50 == 0:
            print(f"  processed {idx+1}/{len(val_dataset)} images")

    # Aggregate per (class, client)
    aggregate: Dict[str, Dict[str, dict]] = {}
    for cls in CLASS_NAMES:
        aggregate[cls] = {}
        for k in range(len(client_models)):
            arr = np.array(sums[cls][k], dtype=np.float64)
            if arr.size == 0:
                aggregate[cls][f"client_{k}"] = {
                    "n": 0, "mean": None, "std": None,
                    "n_flagged_lt_theta": 0,
                }
                continue
            aggregate[cls][f"client_{k}"] = {
                "n":      int(arr.size),
                "mean":   float(arr.mean()),
                "std":    float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "min":    float(arr.min()),
                "max":    float(arr.max()),
                "n_flagged_lt_theta": int(np.sum(arr < theta)),
            }

    gcam_global.remove_hooks()
    for g in gcam_locals:
        g.remove_hooks()

    return {"per_class_per_client": aggregate, "per_image": per_image}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_ckpt", type=str,
                        default="./outputs/checkpoints/round_010.pt")
    parser.add_argument("--local_dir",   type=str,
                        default="./outputs/seeds/seed_42")
    parser.add_argument("--data_dir",    type=str,
                        default="./data/data")
    parser.add_argument("--theta",       type=float, default=0.70)
    parser.add_argument("--num_clients", type=int,   default=3)
    parser.add_argument("--out",         type=str,
                        default="./outputs/gradcam_overlap.json")
    args = parser.parse_args()

    if not os.path.exists(args.global_ckpt):
        # Fallback: highest available round checkpoint.
        ck_dir = os.path.dirname(args.global_ckpt)
        candidates = sorted([f for f in os.listdir(ck_dir)
                              if f.startswith("round_") and f.endswith(".pt")])
        if not candidates:
            raise FileNotFoundError(f"No checkpoints in {ck_dir}")
        args.global_ckpt = os.path.join(ck_dir, candidates[-1])
        print(f"[info] global checkpoint not found, using {args.global_ckpt}")

    local_paths = []
    for k in range(args.num_clients):
        p = os.path.join(args.local_dir, f"local_only_client{k}.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing local-only checkpoint {p}. Run "
                f"`python run_seeds.py --seeds 42 --methods local_only` first.")
        local_paths.append(p)

    print(f"[info] global = {args.global_ckpt}")
    for p in local_paths:
        print(f"[info] local  = {p}")

    global_model  = load_model(args.global_ckpt)
    client_models = [load_model(p) for p in local_paths]

    val_dir = os.path.join(args.data_dir, "validation")
    val_dataset = GBCUDataset(val_dir, transform=get_transforms(train=False))
    print(f"[info] |val| = {len(val_dataset)} images, theta = {args.theta}")

    result = overlap_per_class_client(global_model, client_models,
                                      val_dataset, args.theta)
    result["meta"] = {
        "theta":        args.theta,
        "global_ckpt":  os.path.abspath(args.global_ckpt),
        "local_ckpts":  [os.path.abspath(p) for p in local_paths],
        "n_val":        len(val_dataset),
        "class_names":  CLASS_NAMES,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] wrote {args.out}")
    # Pretty summary
    print("\n=== Grad-CAM cosine overlap O_k^c (mean +/- std) ===")
    for cls, rows in result["per_class_per_client"].items():
        for ck, stats in rows.items():
            if stats["mean"] is None:
                continue
            print(f"  class={cls:5s} {ck}: "
                  f"{stats['mean']:.3f} +/- {stats['std']:.3f}  "
                  f"(n={stats['n']}, flagged<theta={stats['n_flagged_lt_theta']})")


if __name__ == "__main__":
    main()
