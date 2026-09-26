"""
generate_fig9_emphasis.py
----------------------------
Regenerates Fig. 9 (cross-method comparison grid) with one change from
generate_fig9_fig10_with_raw_heatmap.py: for Grad-CAM++ and Eigen-CAM
specifically, an extra duplicate/enlarged raw-heatmap panel is added
beside their existing (raw, overlay) pair, per explicit user request to
give those two methods additional visual emphasis in the grid. The other
three methods (Grad-CAM, Saliency, Score-CAM) are unchanged: one raw +
one overlay panel each.

Column layout: Original, GradCAM(raw,overlay), GradCAM++(raw,overlay,
raw-again), Saliency(raw,overlay), EigenCAM(raw,overlay,raw-again),
ScoreCAM(raw,overlay) -> 13 columns x 9 rows (was 11).

Uses the same representative-image selection (seed=42) as the existing
Fig. 9, so the same images already described in the paper's text/caption
are reproduced, not new ones.

Run from fedgb/:  python generate_fig9_emphasis.py
Outputs: outputs/figures/xai_cross_method_grid.png (overwritten, Fig. 9)
"""
import os
import random

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _p0common import load_primary_model, val_dataset, DEVICE, FIG_DIR, MEAN, STD
from dataset import CLASS_NAMES, CLASS_LABELS
from model import get_target_layer
from gradcam import GradCAM, GradCAMPlusPlus, SaliencyMap, EigenCAM, ScoreCAM, overlay_heatmap

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
EMPHASIZED = {"Grad-CAM++", "Eigen-CAM"}


def to_tensor(img):
    t = torch.tensor(np.array(img.resize((224, 224))).transpose(2, 0, 1) / 255.0,
                     dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return (t - MEAN.to(DEVICE)) / STD.to(DEVICE)


def main():
    model = load_primary_model()
    target_layer = get_target_layer(model)
    ds = val_dataset()

    rng = random.Random(SEED)
    by_class = {}
    for i, (_, label) in enumerate(ds.samples):
        by_class.setdefault(label, []).append(i)

    representative = {}
    for label in sorted(by_class):
        idxs = by_class[label]
        chosen = rng.sample(idxs, min(6, len(idxs)))
        representative[label] = chosen[0]

    print("[fig9-emphasis] generating cross-method grid with extra "
          "Grad-CAM++/Eigen-CAM panels...")
    n_classes = len(CLASS_NAMES)
    method_names = ["Grad-CAM", "Grad-CAM++", "Saliency", "Eigen-CAM", "Score-CAM"]
    # Original + per-method columns (2 normally, 3 for emphasized methods)
    n_cols = 1 + sum(3 if m in EMPHASIZED else 2 for m in method_names)
    fig, axes = plt.subplots(n_classes, n_cols, figsize=(2.4 * n_cols, 2.6 * n_classes))

    col_titles = ["Original"]
    for m in method_names:
        if m in EMPHASIZED:
            col_titles += [f"{m}\n(raw)", f"{m}\n(overlay)", f"{m}\n(raw, enlarged)"]
        else:
            col_titles += [f"{m}\n(raw)", f"{m}\n(overlay)"]

    for row, label in enumerate(sorted(representative)):
        cname = CLASS_NAMES[label]
        idx = representative[label]
        path, _ = ds.samples[idx]
        img = Image.open(path).convert("RGB").resize((224, 224))
        x = to_tensor(img)

        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(CLASS_LABELS[cname], fontsize=8, rotation=0,
                                ha="right", va="center", labelpad=40)

        methods = [
            ("Grad-CAM", GradCAM(model, target_layer)),
            ("Grad-CAM++", GradCAMPlusPlus(model, target_layer)),
            ("Saliency", SaliencyMap(model)),
            ("Eigen-CAM", EigenCAM(model, target_layer)),
            ("Score-CAM", ScoreCAM(model, target_layer, max_channels=64, batch_size=32)),
        ]
        col = 1
        pred_report, conf_report = None, None
        for name, cam in methods:
            heat, pred, conf = cam.generate(x, target_class=label)
            if hasattr(cam, "remove_hooks"):
                cam.remove_hooks()
            if name == "Grad-CAM":
                pred_report, conf_report = pred, conf
            axes[row, col].imshow(heat, cmap="jet")
            axes[row, col + 1].imshow(overlay_heatmap(img, heat))
            if name in EMPHASIZED:
                # duplicate/enlarged view: same heatmap, shown again at a
                # visually larger scale via interpolation for emphasis
                axes[row, col + 2].imshow(heat, cmap="jet", interpolation="bilinear")
                col += 3
            else:
                col += 2

        for c in range(n_cols):
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])
            if row == 0:
                axes[row, c].set_title(col_titles[c], fontsize=8)
        print(f"  row {row+1}/{n_classes}: {CLASS_LABELS[cname]} done "
              f"(pred={CLASS_LABELS[CLASS_NAMES[pred_report]]}, conf={conf_report:.3f})")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "xai_cross_method_grid.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig9-emphasis] wrote {out_path}")


if __name__ == "__main__":
    main()
