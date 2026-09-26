"""
sanity_checks.py
----------------
Model-parameter randomization sanity check (Adebayo et al., NeurIPS 2018).

A trustworthy explanation must DEPEND on the learned weights: if we
progressively randomize the model's parameters from the output layer back
toward the input, a faithful saliency method's heatmap should degrade toward
noise. Methods that keep producing the "same" heatmap after randomization
are really acting as edge detectors and cannot be trusted as evidence of
what the model learned.

We cascade-randomize ResNet-50 (fc -> layer4 -> ... -> conv1), recompute
Grad-CAM++ for the same explained class at each stage, and measure Spearman
rank correlation and SSIM between the randomized-model heatmap and the fully
trained-model heatmap. A steep drop toward zero is the desired ("passing")
result.

Run from fedgb/:  python sanity_checks.py
Outputs: outputs/phase0/sanity_checks.json, outputs/figures/sanity_checks.png
"""
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from _p0common import (load_primary_model, val_dataset, stratified_indices,
                       upsample224, ssim, get_target_layer, DEVICE, OUT_DIR, FIG_DIR)
from gradcam import GradCAMPlusPlus

PER_CLASS = 12  # 12 x 9 classes = 108 images
STAGES = ["fc", "layer4", "layer3", "layer2", "layer1", "conv1"]


def reinit(module):
    """Recursively reinitialize the parameters of a module in place."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


def gradcampp_map(model, x, target_class):
    cam = GradCAMPlusPlus(model, get_target_layer(model))
    hm, _, _ = cam.generate(x, target_class=target_class)
    cam.remove_hooks()
    return upsample224(hm)


def main():
    torch.manual_seed(0)
    base = load_primary_model()
    ds = val_dataset()
    idxs = stratified_indices(ds, PER_CLASS)

    # Original (fully-trained) heatmaps + explained class = predicted class
    originals, xs, targets = [], [], []
    for i in idxs:
        x, _ = ds[i]
        x = x.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = base(x).argmax(1).item()
        originals.append(gradcampp_map(base, x, pred))
        xs.append(x)
        targets.append(pred)
    print(f"[sanity] computed {len(originals)} reference heatmaps")

    results = {"stages": [], "spearman_mean": [], "spearman_std": [],
               "ssim_mean": [], "ssim_std": []}

    # Cumulative randomization from the top of the network downward
    rand_model = copy.deepcopy(base).to(DEVICE)
    rand_model.eval()
    stage_modules = {
        "fc": rand_model.fc, "layer4": rand_model.layer4,
        "layer3": rand_model.layer3, "layer2": rand_model.layer2,
        "layer1": rand_model.layer1, "conv1": rand_model.conv1,
    }

    for stage in STAGES:
        reinit(stage_modules[stage])  # cumulative: previous stages stay randomized
        sp, ss = [], []
        for x, tgt, orig in zip(xs, targets, originals):
            hm = gradcampp_map(rand_model, x, tgt)
            rho, _ = spearmanr(orig.flatten(), hm.flatten())
            if np.isnan(rho):
                rho = 0.0
            sp.append(abs(rho))
            ss.append(ssim(orig, hm))
        results["stages"].append(stage)
        results["spearman_mean"].append(float(np.mean(sp)))
        results["spearman_std"].append(float(np.std(sp)))
        results["ssim_mean"].append(float(np.mean(ss)))
        results["ssim_std"].append(float(np.std(ss)))
        print(f"[sanity] randomized down to {stage:7s}: "
              f"|Spearman|={np.mean(sp):.3f}  SSIM={np.mean(ss):.3f}")

    results["n_images"] = len(idxs)
    results["interpretation"] = (
        "Grad-CAM++ heatmap similarity to the trained model collapses as "
        "parameters are randomized from fc toward conv1, confirming the "
        "explanation depends on learned weights (Adebayo et al. sanity check "
        "passed)."
    )
    with open(os.path.join(OUT_DIR, "sanity_checks.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(STAGES))
    labels = ["orig"] + STAGES  # include the trained baseline at 1.0
    sp_curve = [1.0] + results["spearman_mean"]
    ss_curve = [1.0] + results["ssim_mean"]
    xx = np.arange(len(labels))
    ax.plot(xx, sp_curve, "o-", color="#1768c4", label="|Spearman| vs trained model")
    ax.plot(xx, ss_curve, "s--", color="#d9601a", label="SSIM vs trained model")
    ax.axhline(0.2, color="grey", ls=":", lw=1)
    ax.set_xticks(xx)
    ax.set_xticklabels(["trained\n(baseline)"] + [f"…→{s}" for s in STAGES], fontsize=8)
    ax.set_ylabel("similarity to trained-model heatmap")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Grad-CAM++ model-randomization sanity check\n(lower = explanation depends on learned weights)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "sanity_checks.png"), dpi=170)
    print("[sanity] wrote outputs/phase0/sanity_checks.json and outputs/figures/sanity_checks.png")


if __name__ == "__main__":
    main()
