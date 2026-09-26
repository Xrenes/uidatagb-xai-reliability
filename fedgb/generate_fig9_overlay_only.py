"""
generate_fig9_overlay_only.py
--------------------------------
Regenerates Fig. 9 back to a simple, single-image-per-method layout:
Original + one overlay image for each of the five XAI methods
(Grad-CAM, Grad-CAM++, Saliency, Eigen-CAM, Score-CAM) -- 6 columns x 9
rows. No separate raw-heatmap panels; each method shows just its
standard overlay visualisation, per explicit user request ("dont give
any heatmap images there originatl 5 methords images").

This also fixes a print-legibility problem: the previous 11-column
version (with raw + overlay pairs) was so wide that the print CSS's
max-height:300px cap shrank every panel, including the Original column,
to the point of being hard to read. At 6 columns, each panel renders
roughly twice as large under the same page-width constraint.

Uses the same representative-image selection (seed=42) as the existing
Fig. 9, so the same images already described in the paper's text/caption
are reproduced, not new ones.

Run from fedgb/:  python generate_fig9_overlay_only.py
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

    print("[fig9-overlay-only] generating 6-column cross-method grid...")
    n_classes = len(CLASS_NAMES)
    method_names = ["Grad-CAM", "Grad-CAM++", "Saliency", "Eigen-CAM", "Score-CAM"]
    n_img_cols = 1 + len(method_names)  # Original + 5 methods
    col_titles = ["Original"] + method_names

    # Deterministic square-cell layout: a dedicated label column (fixed
    # width, no image) plus n_img_cols image columns, all exactly
    # cell_size x cell_size inches -- avoids fighting matplotlib's layout
    # engine (tight_layout/constrained_layout/box_aspect all previously
    # left inconsistent gaps because they compute row/column sizing from
    # label text metrics rather than a fixed grid).
    cell_size = 2.3
    label_col_width = 3.1  # inches; fits the longest class name ("Membranous/Gangrenous Cholecystitis") at fontsize=10 without clipping
    title_margin = 0.35    # inches, added on top of the row grid, not carved out of it
    fig_w = label_col_width + cell_size * n_img_cols
    fig_h = cell_size * n_classes + title_margin
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_classes, 1 + n_img_cols,
        width_ratios=[label_col_width] + [cell_size] * n_img_cols,
        left=0, right=1, top=1 - (title_margin / fig_h), bottom=0,
        wspace=0.0, hspace=0.0,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(1 + n_img_cols)]
            for r in range(n_classes)]
    label_axes = []
    for r in range(n_classes):
        label_axes.append(axes[r][0])
        axes[r][0].axis("off")

    for row, label in enumerate(sorted(representative)):
        cname = CLASS_NAMES[label]
        idx = representative[label]
        path, _ = ds.samples[idx]
        img = Image.open(path).convert("RGB").resize((224, 224))
        x = to_tensor(img)

        label_axes[row].text(0.92, 0.5, CLASS_LABELS[cname], fontsize=10,
                             ha="right", va="center", clip_on=False,
                             transform=label_axes[row].transAxes)

        axes[row][1].imshow(img)

        methods = [
            ("Grad-CAM", GradCAM(model, target_layer)),
            ("Grad-CAM++", GradCAMPlusPlus(model, target_layer)),
            ("Saliency", SaliencyMap(model)),
            ("Eigen-CAM", EigenCAM(model, target_layer)),
            ("Score-CAM", ScoreCAM(model, target_layer, max_channels=64, batch_size=32)),
        ]
        col = 2
        pred_report, conf_report = None, None
        for name, cam in methods:
            heat, pred, conf = cam.generate(x, target_class=label)
            if hasattr(cam, "remove_hooks"):
                cam.remove_hooks()
            if name == "Grad-CAM":
                pred_report, conf_report = pred, conf
            axes[row][col].imshow(overlay_heatmap(img, heat))
            col += 1

        for c in range(1, 1 + n_img_cols):
            axes[row][c].set_xticks([])
            axes[row][c].set_yticks([])
            for spine in axes[row][c].spines.values():
                spine.set_visible(False)
            if row == 0:
                axes[row][c].set_title(col_titles[c - 1], fontsize=11)
        print(f"  row {row+1}/{n_classes}: {CLASS_LABELS[cname]} done "
              f"(pred={CLASS_LABELS[CLASS_NAMES[pred_report]]}, conf={conf_report:.3f})")

    out_path = os.path.join(FIG_DIR, "xai_cross_method_grid.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[fig9-overlay-only] wrote {out_path}")


if __name__ == "__main__":
    main()
