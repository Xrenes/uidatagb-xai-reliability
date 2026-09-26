"""
compute_overlap_all_methods.py
-------------------------------
Extends compute_gradcam_overlap.py to compute the cosine-overlap metric
O_k^c for ALL FIVE XAI methods used in the paper (Grad-CAM, Grad-CAM++,
Saliency Maps, Eigen-CAM, Score-CAM). Writes a live progress JSON so an
HTML dashboard can poll status, and a final results JSON consumed to
extend Table V/VI into a full method x class x client comparison.

Score-CAM is far more expensive than the other four methods (one extra
forward pass per sampled channel per image). On CPU this is impractical
at full validation-set scale; use --methods to restrict to the cheap
methods on CPU and run Score-CAM separately on GPU (see --device).

Usage (run from fedgb/):
  python compute_overlap_all_methods.py \
         --global_ckpt ./outputs/checkpoints/round_010.pt \
         --local_dir   ./outputs/seeds/seed_42 \
         --data_dir    ./data/data \
         --theta 0.70 \
         --methods gradcam,gradcam_pp,saliency,eigencam,scorecam

Outputs:
  outputs/overlap_all_methods.json   - full per-class/per-client/per-method results
  outputs/overlap_progress.json      - live progress (polled by progress HTML page)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CLASS_NAMES, GBCUDataset, get_transforms
from gradcam import (
    GradCAM, GradCAMPlusPlus, SaliencyMap, EigenCAM, ScoreCAM, cosine_overlap,
)
from model import build_model, get_target_layer

PROGRESS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "outputs", "overlap_progress.json")

ALL_METHODS = ["gradcam", "gradcam_pp", "saliency", "eigencam", "scorecam"]
METHOD_LABELS = {
    "gradcam": "Grad-CAM",
    "gradcam_pp": "Grad-CAM++",
    "saliency": "Saliency Map",
    "eigencam": "Eigen-CAM",
    "scorecam": "Score-CAM",
}


# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, device: torch.device):
    model = build_model(pretrained=False).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def make_explainer(method: str, model: torch.nn.Module):
    if method == "gradcam":
        return GradCAM(model, get_target_layer(model))
    if method == "gradcam_pp":
        return GradCAMPlusPlus(model, get_target_layer(model))
    if method == "saliency":
        return SaliencyMap(model)
    if method == "eigencam":
        return EigenCAM(model, get_target_layer(model))
    if method == "scorecam":
        return ScoreCAM(model, get_target_layer(model), max_channels=64, batch_size=32)
    raise ValueError(f"Unknown method: {method}")


def compute_heatmap(explainer, tensor: torch.Tensor) -> np.ndarray:
    heatmap, _, _ = explainer.generate(tensor.unsqueeze(0))
    return heatmap


def write_progress(state: dict):
    tmp_path = PROGRESS_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, PROGRESS_PATH)


# ---------------------------------------------------------------------------
def overlap_all_methods(
    global_model, client_models: List, val_dataset, theta: float, methods: List[str],
) -> Dict:
    n_clients = len(client_models)
    n_images = len(val_dataset)
    total_steps = len(methods) * n_images
    step = 0
    start_time = time.time()

    results: Dict[str, dict] = {}

    for method in methods:
        gcam_global = make_explainer(method, global_model)
        gcam_locals = [make_explainer(method, cm) for cm in client_models]

        sums: Dict[str, Dict[int, List[float]]] = {
            c: {k: [] for k in range(n_clients)} for c in CLASS_NAMES
        }
        flagged_counts: Dict[str, Dict[int, int]] = {
            c: {k: 0 for k in range(n_clients)} for c in CLASS_NAMES
        }

        for idx in range(n_images):
            tensor, label = val_dataset[idx]
            cls = CLASS_NAMES[label]
            h_g = compute_heatmap(gcam_global, tensor)
            for k, gcam_l in enumerate(gcam_locals):
                h_l = compute_heatmap(gcam_l, tensor)
                o = cosine_overlap(h_g, h_l)
                sums[cls][k].append(o)
                if o < theta:
                    flagged_counts[cls][k] += 1

            step += 1
            if (idx + 1) % 25 == 0 or idx == n_images - 1:
                elapsed = time.time() - start_time
                rate = step / elapsed if elapsed > 0 else 0
                remaining = (total_steps - step) / rate if rate > 0 else 0
                write_progress({
                    "status": "running",
                    "current_method": method,
                    "current_method_label": METHOD_LABELS[method],
                    "methods_done": methods.index(method),
                    "methods_total": len(methods),
                    "image_index": idx + 1,
                    "images_total": n_images,
                    "overall_step": step,
                    "overall_total": total_steps,
                    "percent": round(100 * step / total_steps, 1),
                    "elapsed_sec": round(elapsed, 1),
                    "eta_sec": round(remaining, 1),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                print(f"[{method}] {idx+1}/{n_images} images "
                      f"({100*step/total_steps:.1f}% overall, ETA {remaining/60:.1f} min)")

        aggregate: Dict[str, Dict[str, dict]] = {}
        for cls in CLASS_NAMES:
            aggregate[cls] = {}
            for k in range(n_clients):
                arr = np.array(sums[cls][k], dtype=np.float64)
                aggregate[cls][f"client_{k}"] = {
                    "n": int(arr.size),
                    "mean": float(arr.mean()) if arr.size else None,
                    "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                    "n_flagged_lt_theta": flagged_counts[cls][k],
                }

        gcam_global.remove_hooks()
        for g in gcam_locals:
            g.remove_hooks()

        results[method] = aggregate
        print(f"[done] method={method} complete")

    return results


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
                        default="./outputs/overlap_all_methods.json")
    parser.add_argument("--methods",     type=str,
                        default="gradcam,gradcam_pp,saliency,eigencam,scorecam",
                        help="Comma-separated subset of: " + ",".join(ALL_METHODS))
    parser.add_argument("--device",      type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="auto = use CUDA if available, else CPU")
    parser.add_argument("--merge",       action="store_true",
                        help="Merge results into an existing --out file instead "
                             "of overwriting it (keeps previously computed methods).")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[info] device = {device}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in ALL_METHODS:
            raise ValueError(f"Unknown method '{m}'. Choose from: {ALL_METHODS}")

    write_progress({
        "status": "starting",
        "current_method": None,
        "methods_done": 0,
        "methods_total": len(methods),
        "image_index": 0,
        "images_total": 0,
        "overall_step": 0,
        "overall_total": 0,
        "percent": 0.0,
        "elapsed_sec": 0,
        "eta_sec": None,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    local_paths = []
    for k in range(args.num_clients):
        p = os.path.join(args.local_dir, f"local_only_client{k}.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing local-only checkpoint {p}")
        local_paths.append(p)

    print(f"[info] global = {args.global_ckpt}")
    for p in local_paths:
        print(f"[info] local  = {p}")

    global_model = load_model(args.global_ckpt, device)
    client_models = [load_model(p, device) for p in local_paths]

    val_dir = os.path.join(args.data_dir, "validation")
    val_dataset = GBCUDataset(val_dir, transform=get_transforms(train=False))
    print(f"[info] |val| = {len(val_dataset)} images, theta = {args.theta}, "
          f"methods = {methods}")

    result = overlap_all_methods(global_model, client_models, val_dataset, args.theta, methods)

    if args.merge and os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        existing.update(result)  # new methods overwrite same-named keys, others kept
        existing.setdefault("meta", {})
        existing["meta"]["methods"] = sorted(set(existing["meta"].get("methods", [])) | set(methods))
        existing["meta"]["device_last_run"] = str(device)
        result = existing
        print(f"[info] merged with existing {args.out}")
    else:
        result["meta"] = {
            "theta": args.theta,
            "global_ckpt": os.path.abspath(args.global_ckpt),
            "local_ckpts": [os.path.abspath(p) for p in local_paths],
            "n_val": len(val_dataset),
            "class_names": CLASS_NAMES,
            "methods": methods,
            "device_last_run": str(device),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    write_progress({
        "status": "done",
        "current_method": None,
        "methods_done": len(methods),
        "methods_total": len(methods),
        "image_index": len(val_dataset),
        "images_total": len(val_dataset),
        "overall_step": len(methods) * len(val_dataset),
        "overall_total": len(methods) * len(val_dataset),
        "percent": 100.0,
        "elapsed_sec": None,
        "eta_sec": 0,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    print(f"\n[done] wrote {args.out}")
    print("\n=== Cosine overlap O_k^c summary (mean +/- std) ===")
    for method in methods:
        print(f"\n-- {METHOD_LABELS[method]} --")
        for cls, rows in result[method].items():
            for ck, stats in rows.items():
                if stats["mean"] is None:
                    continue
                print(f"  class={cls:5s} {ck}: {stats['mean']:.3f} +/- {stats['std']:.3f} "
                      f"(n={stats['n']}, flagged<theta={stats['n_flagged_lt_theta']})")


if __name__ == "__main__":
    main()
