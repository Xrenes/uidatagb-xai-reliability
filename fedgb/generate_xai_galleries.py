"""
generate_xai_galleries.py
--------------------------
Runs all 5 XAI methods (Grad-CAM, Grad-CAM++, Saliency, Eigen-CAM, Score-CAM)
on the primary UIdataGB model and produces:
  1. A per-class Grad-CAM++ gallery (6 samples/class x 9 classes = 54 images)
     for the paper's main-text figure + appendix gallery.
  2. A cross-method comparison figure: one representative image per class,
     all 5 methods side by side (9 rows x 5 methods).

Run from fedgb/:  python generate_xai_galleries.py
Outputs:
  outputs/gradcam_uidatagb/per_class/<class>_<n>.png
  outputs/gradcam_uidatagb/cross_method/<class>_comparison.png
  outputs/figures/xai_cross_method_grid.png
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
OUT_ROOT = os.path.join(HERE, "outputs", "gradcam_uidatagb")
PER_CLASS_DIR = os.path.join(OUT_ROOT, "per_class")
CROSS_DIR = os.path.join(OUT_ROOT, "cross_method")
os.makedirs(PER_CLASS_DIR, exist_ok=True)
os.makedirs(CROSS_DIR, exist_ok=True)

SAMPLES_PER_CLASS = 6
SEED = 42


def to_tensor(img):
    """PIL image (224x224 already via dataset resize) -> normalized batch tensor."""
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

    # ---- 1. Per-class Grad-CAM++ gallery ----
    print("[gallery] generating per-class Grad-CAM++ gallery...")
    representative = {}  # class_idx -> chosen image index (for cross-method fig)
    for label, idxs in by_class.items():
        cname = CLASS_NAMES[label]
        chosen = rng.sample(idxs, min(SAMPLES_PER_CLASS, len(idxs)))
        representative[label] = chosen[0]
        for n, idx in enumerate(chosen, 1):
            path, _ = ds.samples[idx]
            img = Image.open(path).convert("RGB")
            x = to_tensor(img)
            cam = GradCAMPlusPlus(model, target_layer)
            heat, pred, conf = cam.generate(x)
            cam.remove_hooks()
            overlay = overlay_heatmap(img.resize((224, 224)), heat)
            Image.fromarray(overlay).save(
                os.path.join(PER_CLASS_DIR, f"{cname}_{n:02d}.png"))
        print(f"  {CLASS_LABELS[cname]:38s} {len(chosen)} images")

    # ---- 2. Cross-method comparison grid (1 image/class x 5 methods) ----
    print("[gallery] generating cross-method comparison grid...")
    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(n_classes, 6, figsize=(15, 2.6 * n_classes))
    method_names = ["Original", "Grad-CAM", "Grad-CAM++", "Saliency", "Eigen-CAM", "Score-CAM"]

    for row, label in enumerate(sorted(representative)):
        cname = CLASS_NAMES[label]
        idx = representative[label]
        path, _ = ds.samples[idx]
        img = Image.open(path).convert("RGB").resize((224, 224))
        x = to_tensor(img)

        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(CLASS_LABELS[cname], fontsize=8, rotation=0,
                                ha="right", va="center", labelpad=40)

        gc = GradCAM(model, target_layer)
        h_gc, pred, conf = gc.generate(x, target_class=label)
        gc.remove_hooks()
        axes[row, 1].imshow(overlay_heatmap(img, h_gc))

        gcpp = GradCAMPlusPlus(model, target_layer)
        h_gcpp, _, _ = gcpp.generate(x, target_class=label)
        gcpp.remove_hooks()
        axes[row, 2].imshow(overlay_heatmap(img, h_gcpp))

        sal = SaliencyMap(model)
        h_sal, _, _ = sal.generate(x, target_class=label)
        axes[row, 3].imshow(overlay_heatmap(img, h_sal))

        eig = EigenCAM(model, target_layer)
        h_eig, _, _ = eig.generate(x, target_class=label)
        eig.remove_hooks()
        axes[row, 4].imshow(overlay_heatmap(img, h_eig))

        sc = ScoreCAM(model, target_layer, max_channels=64, batch_size=32)
        h_sc, _, _ = sc.generate(x, target_class=label)
        sc.remove_hooks()
        axes[row, 5].imshow(overlay_heatmap(img, h_sc))

        for col in range(6):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(method_names[col], fontsize=9)
        print(f"  row {row+1}/{n_classes}: {CLASS_LABELS[cname]} done "
              f"(pred={CLASS_LABELS[CLASS_NAMES[pred]]}, conf={conf:.3f})")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "xai_cross_method_grid.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[gallery] wrote {out_path}")
    print(f"[gallery] wrote {PER_CLASS_DIR}/ ({n_classes * SAMPLES_PER_CLASS} images)")


if __name__ == "__main__":
    main()
