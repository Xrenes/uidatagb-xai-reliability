"""
generate_fig10_individual.py
-------------------------------
Regenerates Fig. 10 as 9 individual, uniformly-sized 3-panel images (one
per class: original / raw Grad-CAM++ heatmap / overlay) instead of one
composite matplotlib PNG -- so it uses the same structure as
Supplementary Material A2 (a CSS grid of individually captioned
<figure> elements, all the same pixel size) rather than a single baked
image, per user request for structural consistency between the two.

Uses the exact same seeded second-sample selection (seed=42, sample_02
per class) as the original per_class_heatmap_grid.png, so the images
shown are unchanged -- only how they're packaged/laid out changes.

Run from fedgb/:  python generate_fig10_individual.py
Outputs: outputs/figures/fig10_individual/<class>.png (9 images, all identical size)
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
OUT_DIR = os.path.join(HERE, "outputs", "figures", "fig10_individual")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
FIG_W_IN, FIG_H_IN, DPI = 12.0, 4.13, 150  # same fixed size as generate_supp_a_3panel.py


def to_tensor(img):
    t = torch.tensor(np.array(img.resize((224, 224))).transpose(2, 0, 1) / 255.0,
                     dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return (t - MEAN.to(DEVICE)) / STD.to(DEVICE)


def save_fixed_size_3panel(original_image, heatmap, save_path, class_label):
    overlay = overlay_heatmap(original_image, heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_IN, FIG_H_IN))

    axes[0].imshow(original_image)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM++ (raw)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=DPI)
    plt.close(fig)


def main():
    model = load_primary_model()
    target_layer = get_target_layer(model)
    ds = val_dataset()

    rng = random.Random(SEED)
    by_class = {}
    for i, (_, label) in enumerate(ds.samples):
        by_class.setdefault(label, []).append(i)

    for label in sorted(by_class):
        cname = CLASS_NAMES[label]
        idxs = by_class[label]
        chosen = rng.sample(idxs, min(6, len(idxs)))
        idx = chosen[1] if len(chosen) > 1 else chosen[0]  # sample_02, matches existing Fig. 10

        path, _ = ds.samples[idx]
        img = Image.open(path).convert("RGB").resize((224, 224))
        x = to_tensor(img)

        cam = GradCAMPlusPlus(model, target_layer)
        heat, pred, conf = cam.generate(x)
        cam.remove_hooks()

        save_path = os.path.join(OUT_DIR, f"{cname}.png")
        save_fixed_size_3panel(img, heat, save_path, CLASS_LABELS[cname])
        print(f"[fig10-individual] wrote {save_path} "
              f"(pred={CLASS_LABELS[CLASS_NAMES[pred]]}, conf={conf:.3f})")


if __name__ == "__main__":
    main()
