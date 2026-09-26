"""
generate_fig9_fig10_with_raw_heatmap.py
------------------------------------------
Regenerates Fig. 9 (cross-method comparison grid) and Fig. 10 (per-class
Grad-CAM++ grid) with an added *raw, unblended* heatmap panel next to
each existing overlay panel, so the heatmap itself (not just the
image+heatmap blend) is directly visible.

Fig. 9: Original, then (raw heatmap | overlay) pairs for each of the 5
        XAI methods -> 11 columns x 9 rows.
Fig. 10: Original | raw Grad-CAM++ heatmap | overlay, per class
         -> 3 columns x 9 rows (uses the same second-sample selection
         as the existing per_class gallery, sample_02 per class).

Uses the same representative-image selection (seed=42) as
generate_xai_galleries.py / generate_second_sample_grid.py for
consistency with the samples already shown elsewhere in the paper.

Run from fedgb/:  python generate_fig9_fig10_with_raw_heatmap.py
Outputs:
  outputs/figures/xai_cross_method_grid.png   (overwritten, Fig. 9)
  outputs/figures/per_class_heatmap_grid.png  (overwritten, Fig. 10)
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

    # Reproduce the exact same sampling order as generate_xai_galleries.py
    # so "representative" (sample_01) matches Fig. 9's original selection,
    # and the second draw (sample_02) matches Fig. 10's existing selection.
    representative = {}
    second_sample = {}
    for label in sorted(by_class):
        idxs = by_class[label]
        chosen = rng.sample(idxs, min(6, len(idxs)))
        representative[label] = chosen[0]
        second_sample[label] = chosen[1] if len(chosen) > 1 else chosen[0]

    # ================= Fig. 9: cross-method grid with raw + overlay =================
    print("[fig9] generating cross-method grid with raw heatmap panels...")
    n_classes = len(CLASS_NAMES)
    method_names = ["Grad-CAM", "Grad-CAM++", "Saliency", "Eigen-CAM", "Score-CAM"]
    n_cols = 1 + 2 * len(method_names)  # Original + (raw, overlay) x 5
    fig, axes = plt.subplots(n_classes, n_cols, figsize=(2.4 * n_cols, 2.6 * n_classes))

    col_titles = ["Original"]
    for m in method_names:
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
    print(f"[fig9] wrote {out_path}")

    # ================= Fig. 10: per-class grid, original+raw+overlay =================
    print("[fig10] generating per-class grid with raw heatmap panels...")
    fig2, axes2 = plt.subplots(n_classes, 3, figsize=(7.2, 2.6 * n_classes))
    col_titles2 = ["Original", "Grad-CAM++ (raw)", "Overlay"]

    for row, label in enumerate(sorted(second_sample)):
        cname = CLASS_NAMES[label]
        idx = second_sample[label]
        path, _ = ds.samples[idx]
        img = Image.open(path).convert("RGB").resize((224, 224))
        x = to_tensor(img)

        cam = GradCAMPlusPlus(model, target_layer)
        heat, pred, conf = cam.generate(x, target_class=label)
        cam.remove_hooks()

        axes2[row, 0].imshow(img)
        axes2[row, 0].set_ylabel(CLASS_LABELS[cname], fontsize=8, rotation=0,
                                 ha="right", va="center", labelpad=55)
        axes2[row, 1].imshow(heat, cmap="jet")
        axes2[row, 2].imshow(overlay_heatmap(img, heat))

        for c in range(3):
            axes2[row, c].set_xticks([])
            axes2[row, c].set_yticks([])
            if row == 0:
                axes2[row, c].set_title(col_titles2[c], fontsize=9)
        print(f"  row {row+1}/{n_classes}: {CLASS_LABELS[cname]} done "
              f"(pred={CLASS_LABELS[CLASS_NAMES[pred]]}, conf={conf:.3f})")

    plt.tight_layout()
    out_path2 = os.path.join(FIG_DIR, "per_class_heatmap_grid.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[fig10] wrote {out_path2}")


if __name__ == "__main__":
    main()
