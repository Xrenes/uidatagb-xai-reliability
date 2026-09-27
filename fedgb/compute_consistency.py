"""
compute_consistency.py
------------------------
Cross-replicate explanation-consistency score O^c (cosine similarity between
the primary model's heatmap and each of the 3 replicate models' heatmaps for
the same image/class), computed across all 5 XAI methods -- same design as
the GBCU study's Table V.

Uses a stratified sample (not the full 3,193-image validation set, since
Score-CAM alone would need 3,193 x 3 replicates x 64 forward passes) --
disclosed explicitly in the output and paper text.

Run from fedgb/:  python compute_consistency.py
Outputs: outputs/phase0/consistency.json, outputs/figures/consistency.png
"""
import json
import os
import random

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _p0common import load_primary_model, val_dataset, DEVICE, OUT_DIR, FIG_DIR, MEAN, STD
from dataset import CLASS_NAMES, CLASS_LABELS
from model import build_model, get_target_layer
from gradcam import (GradCAM, GradCAMPlusPlus, SaliencyMap, EigenCAM, ScoreCAM,
                     cosine_overlap)

HERE = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("P0_REPLICATE_CKPTS"):
    REPLICATE_CKPTS = os.environ["P0_REPLICATE_CKPTS"].split(os.pathsep)
else:
    REPLICATE_CKPTS = [
        os.path.join(HERE, "outputs", "seeds", "seed_42", f"local_only_client{c}.pt")
        for c in range(3)
    ]
PER_CLASS = 8  # 8 x 9 classes = 72 images
SEED = 42


def to_tensor(img):
    t = torch.tensor(np.array(img.resize((224, 224))).transpose(2, 0, 1) / 255.0,
                     dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return (t - MEAN.to(DEVICE)) / STD.to(DEVICE)


def load_replicate(path):
    m = build_model(pretrained=False).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


def main():
    primary = load_primary_model()
    primary_layer = get_target_layer(primary)
    replicates = [load_replicate(p) for p in REPLICATE_CKPTS]
    replicate_layers = [get_target_layer(m) for m in replicates]
    print(f"[consistency] loaded primary + {len(replicates)} replicate models")

    ds = val_dataset()
    rng = random.Random(SEED)
    by_class = {}
    for i, (_, label) in enumerate(ds.samples):
        by_class.setdefault(label, []).append(i)
    idxs = []
    for label, lst in by_class.items():
        idxs += rng.sample(lst, min(PER_CLASS, len(lst)))
    idxs.sort()
    print(f"[consistency] evaluating {len(idxs)} stratified images "
          f"({PER_CLASS}/class x {len(CLASS_NAMES)} classes)")

    methods = ["Grad-CAM", "Grad-CAM++", "Saliency", "Eigen-CAM", "Score-CAM"]
    # scores[method][replicate_idx] -> list of per-image cosine overlaps
    scores = {m: [[] for _ in range(3)] for m in methods}
    per_class_scores = {m: {c: [] for c in CLASS_NAMES} for m in methods}

    def make_explainer(name, model, layer):
        if name == "Grad-CAM":
            return GradCAM(model, layer)
        if name == "Grad-CAM++":
            return GradCAMPlusPlus(model, layer)
        if name == "Saliency":
            return SaliencyMap(model)
        if name == "Eigen-CAM":
            return EigenCAM(model, layer)
        if name == "Score-CAM":
            return ScoreCAM(model, layer, max_channels=64, batch_size=128)
        raise ValueError(name)

    for n_done, idx in enumerate(idxs):
        path, label = ds.samples[idx]
        cname = CLASS_NAMES[label]
        img = Image.open(path).convert("RGB")
        x = to_tensor(img)

        with torch.no_grad():
            pred = primary(x).argmax(1).item()

        for method in methods:
            cam_p = make_explainer(method, primary, primary_layer)
            h_p, _, _ = cam_p.generate(x, target_class=pred)
            if hasattr(cam_p, "remove_hooks"):
                cam_p.remove_hooks()

            for r, (rmodel, rlayer) in enumerate(zip(replicates, replicate_layers)):
                cam_r = make_explainer(method, rmodel, rlayer)
                h_r, _, _ = cam_r.generate(x, target_class=pred)
                if hasattr(cam_r, "remove_hooks"):
                    cam_r.remove_hooks()
                o = cosine_overlap(h_p, h_r)
                scores[method][r].append(o)
                per_class_scores[method][cname].append(o)

        if (n_done + 1) % 10 == 0:
            print(f"[consistency] {n_done + 1}/{len(idxs)} images done")

    # Bootstrap 95% CI on the macro mean, resampling images with replacement
    # (not individual scores) so the resample respects the same
    # image-level correlation structure used for the classification-
    # accuracy CIs in Section IV.1 -- added in response to a review
    # asking for confidence intervals on every XAI table, not just
    # classification accuracy.
    N_BOOT = 2000
    boot_rng = np.random.RandomState(SEED)

    def bootstrap_ci(per_image_scores_by_replicate):
        # per_image_scores_by_replicate: list of 3 lists, each length
        # n_images, aligned by image index (scores[method][r][i] is
        # image i's score against replicate r).
        n_images = len(per_image_scores_by_replicate[0])
        arr = np.array(per_image_scores_by_replicate)  # shape (3, n_images)
        boot_means = []
        for _ in range(N_BOOT):
            sample_idx = boot_rng.randint(0, n_images, size=n_images)
            boot_means.append(arr[:, sample_idx].mean())
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        return float(lo), float(hi)

    summary = {"n_images": len(idxs), "per_class_n": PER_CLASS,
               "bootstrap_n_resamples": N_BOOT, "methods": {}}
    for method in methods:
        all_scores = [o for r in scores[method] for o in r]
        per_rep = [float(np.mean(r)) for r in scores[method]]
        ci_lo, ci_hi = bootstrap_ci(scores[method])
        summary["methods"][method] = {
            "macro_mean": float(np.mean(all_scores)),
            "macro_std": float(np.std(all_scores)),
            "per_replicate_mean": per_rep,
            "per_replicate_mean_std_of_means": float(np.std(per_rep)),
            "bootstrap_95ci": [ci_lo, ci_hi],
            "per_class_mean": {c: float(np.mean(v)) if v else None
                              for c, v in per_class_scores[method].items()},
        }
        print(f"[consistency] {method:12s} macro O^c = {summary['methods'][method]['macro_mean']:.3f} "
              f"+/- {summary['methods'][method]['macro_std']:.3f}  "
              f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

    with open(os.path.join(OUT_DIR, "consistency.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- figure: bar chart of macro O^c per method ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = [summary["methods"][m]["macro_mean"] for m in methods]
    errs = [summary["methods"][m]["macro_std"] for m in methods]
    colors = ["#1768c4", "#0f6e5c", "#d9601a", "#6a2fa6", "#b8752f"]
    ax.bar(methods, vals, yerr=errs, capsize=4, color=colors)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Cosine consistency $O^c$ (primary vs. 3 replicates)")
    ax.set_title(f"Cross-replicate explanation consistency (n={len(idxs)} images)")
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.04, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "consistency.png"), dpi=160)
    print(f"[consistency] wrote {OUT_DIR}/consistency.json and {FIG_DIR}/consistency.png")


if __name__ == "__main__":
    main()
