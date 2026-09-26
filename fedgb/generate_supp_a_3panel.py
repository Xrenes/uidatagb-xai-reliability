"""
generate_supp_a_3panel.py
---------------------------
Regenerates 20 of Supplementary Material A's per-class Grad-CAM++ samples
(roughly 2-3 per class across the 9 classes) as 3-panel figures --
original ultrasound | raw Grad-CAM++ heatmap | overlay -- matching the
style already used for the main-text failure-case figure (Fig. 8). Same
seeded sampling as generate_xai_galleries.py (seed=42) so the selected
images are drawn from the same per-class pool already used elsewhere in
the paper.

Unlike gradcam.py's save_gradcam_figure() (which uses bbox_inches="tight"
and a variable-length "Pred:/True:" title, producing a different output
width per class), this script uses a fixed figure size and a short,
uniform per-panel caption so every one of the 20 output PNGs is
pixel-identical in size -- required for a clean, evenly-aligned CSS grid.

Run from fedgb/:  python generate_supp_a_3panel.py
Outputs: outputs/gradcam_uidatagb/per_class_3panel/<class>_<n>.png (20 images, all 1800x620px)
"""
import os
import random

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _p0common import load_primary_model, val_dataset, DEVICE, MEAN, STD
from dataset import CLASS_NAMES, CLASS_LABELS
from model import get_target_layer
from gradcam import GradCAMPlusPlus, overlay_heatmap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs", "gradcam_uidatagb", "per_class_3panel")
os.makedirs(OUT_DIR, exist_ok=True)

FIG_W_IN, FIG_H_IN, DPI = 12.0, 4.13, 150  # fixed size -> 1800x620px every time


def save_fixed_size_3panel(original_image, heatmap, pred_class, confidence,
                            true_class, save_path):
    overlay = overlay_heatmap(original_image, heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_IN, FIG_H_IN))

    axes[0].imshow(original_image.resize((224, 224)))
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM++ (raw)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    correct = "✓" if pred_class == true_class else "✗"
    axes[2].set_title(f"Overlay ({confidence:.0%} conf. {correct})", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=DPI)  # fixed figsize, no bbox_inches="tight"
    plt.close(fig)

SEED = 42
TOTAL_SAMPLES = 20
N_CLASSES = 9
# 20 / 9 classes -> 2 each for 7 classes, 3 each for 2 classes (round-robin)
PER_CLASS_COUNTS = [3, 3] + [2] * 7  # sums to 20, longest classes first (CARC, GS by size)


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

    # Order classes by descending pool size so the "3-sample" classes are
    # well-populated ones (matches Table 1: Carcinoma largest, then Gallstones).
    class_order = sorted(by_class.keys(), key=lambda l: -len(by_class[l]))

    written = 0
    for label, n_take in zip(class_order, PER_CLASS_COUNTS):
        cname = CLASS_NAMES[label]
        idxs = by_class[label]
        chosen = rng.sample(idxs, min(n_take, len(idxs)))
        for n, idx in enumerate(chosen, 1):
            path, true_label = ds.samples[idx]
            img = Image.open(path).convert("RGB")
            x = to_tensor(img)

            cam = GradCAMPlusPlus(model, target_layer)
            heat, pred, conf = cam.generate(x)
            cam.remove_hooks()

            save_path = os.path.join(OUT_DIR, f"{cname}_{n:02d}.png")
            save_fixed_size_3panel(
                original_image=img,
                heatmap=heat,
                pred_class=pred,
                confidence=conf,
                true_class=true_label,
                save_path=save_path,
            )
            written += 1
        print(f"[supp-a-3panel] {CLASS_LABELS[cname]:38s} {len(chosen)} images "
              f"({written}/{TOTAL_SAMPLES} total)")

    print(f"[supp-a-3panel] wrote {written} three-panel images to {OUT_DIR}")


if __name__ == "__main__":
    main()
