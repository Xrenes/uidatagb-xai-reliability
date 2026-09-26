"""
compute_localization.py
------------------------
Analyses returned radiologist ROI annotations (see build_roi_study.py /
roi_annotation_tool.html) against the primary model's Grad-CAM++ heatmaps.
This is the metric the plausibility study (compute_kappa.py) cannot give:
a geometric answer to "does the heatmap actually land on the gallbladder /
pathology region", not just a subjective 1-5 rating.

For each image with a rater-drawn gallbladder box (and, where present, a
pathology box):
  - IoU / Dice between the box and the top-X% thresholded heatmap region
  - Pointing-game hit: is the heatmap's single max-attribution pixel inside
    the box? (Zhang et al., 2018 pointing game -- the standard weak-
    localization metric in the XAI literature)
  - Energy-in-ROI: fraction of total (unthresholded) heatmap mass inside
    the box -- threshold-independent complement to IoU/Dice

If multiple raters' JSON exports are given, also reports inter-rater box
agreement (IoU between raters' own boxes) as a sanity check on the ground
truth itself, the geometric analogue of compute_kappa.py's Cohen's kappa.

Usage (run from fedgb/):
  python compute_localization.py \
      --raters outputs/radiologist_study/roi_AR.json,outputs/radiologist_study/roi_BK.json \
      --manifest outputs/radiologist_study/manifest.json \
      --top_pct 25
"""
from __future__ import annotations
import argparse
import json
import os
from itertools import combinations

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CLASS_NAMES
from gradcam import GradCAMPlusPlus
from model import build_model, get_target_layer
from _p0common import upsample224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_CKPT = "./outputs/checkpoints/round_010.pt"
VAL_DIR = "./data/data/validation"
OUT_DIR = "./outputs/phase1"
FIG_DIR = "./outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_image_tensor(path, size=224):
    from PIL import Image
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return tf(Image.open(path).convert("RGB"))


def box_mask(box, h, w):
    m = np.zeros((h, w), dtype=bool)
    if box is None:
        return m
    x1, y1, x2, y2 = box
    x1, x2 = int(x1 * w), int(x2 * w)
    y1, y2 = int(y1 * h), int(y2 * h)
    m[y1:y2, x1:x2] = True
    return m


def heat_mask(heatmap, top_pct):
    thresh = np.percentile(heatmap, 100 - top_pct)
    return heatmap >= thresh


def iou_dice(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    iou = float(inter / union) if union > 0 else float("nan")
    dice = float(2 * inter / (mask_a.sum() + mask_b.sum())) \
        if (mask_a.sum() + mask_b.sum()) > 0 else float("nan")
    return iou, dice


def pointing_game_hit(heatmap, mask):
    if mask.sum() == 0:
        return None
    peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return bool(mask[peak])


def energy_in_roi(heatmap, mask):
    total = heatmap.sum()
    if total <= 0 or mask.sum() == 0:
        return None
    return float(heatmap[mask].sum() / total)


def load_rater_json(path):
    d = json.load(open(path))
    return d["rater"], {it["id"]: it for it in d["items"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", required=True, help="comma-separated roi_<rater>.json exports")
    ap.add_argument("--manifest", default="outputs/radiologist_study/manifest.json")
    ap.add_argument("--top_pct", type=float, default=25.0,
                    help="top-X%% of heatmap pixels used for the thresholded IoU/Dice")
    args = ap.parse_args()

    manifest = {m["rating_id"]: m for m in json.load(open(args.manifest))["items"]}
    rater_paths = [p.strip() for p in args.raters.split(",") if p.strip()]
    raters = {}
    for p in rater_paths:
        name, items = load_rater_json(p)
        raters[name] = items
    names = list(raters)
    print(f"=== {len(names)} rater(s): {names} ===")

    model = build_model(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(GLOBAL_CKPT, map_location=DEVICE))
    model.eval()
    cam = GradCAMPlusPlus(model, get_target_layer(model))

    per_rater_results = {n: [] for n in names}
    heatmap_cache = {}

    for rid, m in manifest.items():
        img_path = os.path.join(VAL_DIR, m["true_class"], m["src"])
        if rid not in heatmap_cache:
            x = load_image_tensor(img_path)
            xn = ((x - MEAN) / STD).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(xn).argmax(1).item()
            heat, _, _ = cam.generate(xn, target_class=pred)
            heatmap_cache[rid] = upsample224(heat)  # raw output is 7x7 (layer4)

        heat = heatmap_cache[rid]
        h, w = heat.shape
        thresh_mask = heat_mask(heat, args.top_pct)

        for name in names:
            it = raters[name].get(rid)
            if it is None:
                continue
            row = {"rating_id": rid, "true_class": m["true_class"],
                  "model_correct": m["model_correct"]}
            if it.get("gb_box"):
                gm = box_mask(it["gb_box"], h, w)
                iou, dice = iou_dice(thresh_mask, gm)
                row["gb_iou"] = iou
                row["gb_dice"] = dice
                row["gb_pointing_hit"] = pointing_game_hit(heat, gm)
                row["gb_energy_in_roi"] = energy_in_roi(heat, gm)
            if it.get("path_box"):
                pm = box_mask(it["path_box"], h, w)
                iou, dice = iou_dice(thresh_mask, pm)
                row["path_iou"] = iou
                row["path_dice"] = dice
                row["path_pointing_hit"] = pointing_game_hit(heat, pm)
                row["path_energy_in_roi"] = energy_in_roi(heat, pm)
            per_rater_results[name].append(row)

    cam.remove_hooks()

    # ---- summary ----
    summary = {"top_pct": args.top_pct, "n_raters": len(names), "per_rater": {}}
    for name, rows in per_rater_results.items():
        gb_iou = [r["gb_iou"] for r in rows if "gb_iou" in r]
        gb_dice = [r["gb_dice"] for r in rows if "gb_dice" in r]
        gb_hit = [r["gb_pointing_hit"] for r in rows if r.get("gb_pointing_hit") is not None]
        gb_energy = [r["gb_energy_in_roi"] for r in rows if r.get("gb_energy_in_roi") is not None]
        path_iou = [r["path_iou"] for r in rows if "path_iou" in r]
        path_hit = [r["path_pointing_hit"] for r in rows if r.get("path_pointing_hit") is not None]

        s = {
            "n_images_annotated": len(rows),
            "gallbladder": {
                "mean_iou": float(np.mean(gb_iou)) if gb_iou else None,
                "mean_dice": float(np.mean(gb_dice)) if gb_dice else None,
                "pointing_game_accuracy": float(np.mean(gb_hit)) if gb_hit else None,
                "mean_energy_in_roi": float(np.mean(gb_energy)) if gb_energy else None,
                "n": len(gb_iou),
            },
            "pathology": {
                "mean_iou": float(np.mean(path_iou)) if path_iou else None,
                "pointing_game_accuracy": float(np.mean(path_hit)) if path_hit else None,
                "n": len(path_iou),
            },
        }
        summary["per_rater"][name] = s
        print(f"\n[{name}] n={s['n_images_annotated']}")
        print(f"  gallbladder: IoU={s['gallbladder']['mean_iou']}, "
              f"Dice={s['gallbladder']['mean_dice']}, "
              f"pointing-game={s['gallbladder']['pointing_game_accuracy']}, "
              f"energy-in-ROI={s['gallbladder']['mean_energy_in_roi']}")
        if s["pathology"]["n"] > 0:
            print(f"  pathology:   IoU={s['pathology']['mean_iou']}, "
                  f"pointing-game={s['pathology']['pointing_game_accuracy']}")

    # ---- inter-rater box agreement (if >=2 raters) ----
    if len(names) >= 2:
        print("\n-- inter-rater gallbladder-box agreement (IoU between raters' own boxes) --")
        summary["inter_rater_box_iou"] = {}
        for a, b in combinations(names, 2):
            ious = []
            for rid in manifest:
                ia, ib = raters[a].get(rid), raters[b].get(rid)
                if ia and ib and ia.get("gb_box") and ib.get("gb_box"):
                    m = manifest[rid]
                    ma = box_mask(ia["gb_box"], 224, 224)
                    mb = box_mask(ib["gb_box"], 224, 224)
                    iou, _ = iou_dice(ma, mb)
                    ious.append(iou)
            mean_iou = float(np.mean(ious)) if ious else None
            summary["inter_rater_box_iou"][f"{a}_vs_{b}"] = mean_iou
            print(f"  {a} vs {b}: mean box IoU = {mean_iou} (n={len(ious)})")

    with open(os.path.join(OUT_DIR, "localization.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- figure: IoU / pointing-game bar chart per rater ----
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(names))
    ious = [summary["per_rater"][n]["gallbladder"]["mean_iou"] or 0 for n in names]
    hits = [summary["per_rater"][n]["gallbladder"]["pointing_game_accuracy"] or 0 for n in names]
    ax.bar(xs - 0.18, ious, width=0.35, label=f"IoU (top {args.top_pct:.0f}% heatmap)", color="#1768c4")
    ax.bar(xs + 0.18, hits, width=0.35, label="Pointing-game accuracy", color="#d9601a")
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_title("Grad-CAM++ vs. radiologist gallbladder ROI")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "localization.png"), dpi=170)
    print(f"\n[done] wrote {OUT_DIR}/localization.json and {FIG_DIR}/localization.png")


if __name__ == "__main__":
    main()
