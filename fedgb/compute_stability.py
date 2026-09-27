"""
compute_stability.py
--------------------
Explanation stability under clinically-invariant transforms (research-plan
RQ4). A small change that should NOT alter the diagnosis (mild brightness
shift, a few-pixel translation, low-level noise) should also not drastically
change WHERE the model says it is looking. We apply each transform, recompute
Grad-CAM++ for the originally predicted class, geometrically realign the
translated map, and measure agreement with the original heatmap via SSIM,
Pearson correlation, and top-10% pixel overlap (IoU). We also record whether
the predicted class is unchanged.

Run from fedgb/:  python compute_stability.py
Outputs: outputs/phase0/stability.json, outputs/figures/stability.png
"""
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from _p0common import (load_primary_model, val_dataset, stratified_indices,
                       upsample224, normalize, raw_tensor, ssim,
                       get_target_layer, DEVICE, OUT_DIR, FIG_DIR)
from gradcam import GradCAMPlusPlus

PER_CLASS = 24  # 24 x 9 classes = 216 images
SHIFT = 10      # pixels for translation test
BRIGHT = 1.15   # brightness multiplier
NOISE = 0.03    # gaussian noise std (in [0,1] pixel space)

HERE = os.path.dirname(os.path.abspath(__file__))
CLUSTER_MANIFEST_PATH = os.environ.get(
    "P0_CLUSTER_MANIFEST",
    os.path.join(HERE, "outputs", "phase0", "dihedral_cluster_manifest.json"))


def _rel_key(path):
    norm = path.replace("\\", "/")
    parts = norm.split("/")
    for i, p in enumerate(parts):
        if p.lower() in ("training", "validation"):
            return "/".join(parts[i:])
    return "/".join(parts[-3:])


def load_cluster_lookup():
    if not os.path.exists(CLUSTER_MANIFEST_PATH):
        print(f"[stability] WARNING: cluster manifest not found at "
              f"{CLUSTER_MANIFEST_PATH}; group-aware bootstrap will be "
              f"skipped, only image-level CI reported.")
        return {}
    with open(CLUSTER_MANIFEST_PATH) as f:
        manifest = json.load(f)
    return {_rel_key(p): cid for p, cid in manifest["path_to_cluster_id"].items()}


def gcpp(model, cam, x_norm, target_class):
    hm, pred, _ = cam.generate(x_norm, target_class=target_class)
    return upsample224(hm), pred


def topk_iou(a, b, frac=0.10):
    k = int(a.size * frac)
    ta = a.flatten().argsort()[-k:]
    tb = b.flatten().argsort()[-k:]
    sa, sb = set(ta.tolist()), set(tb.tolist())
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def main():
    model = load_primary_model()
    cam = GradCAMPlusPlus(model, get_target_layer(model))
    ds = val_dataset()
    idxs = stratified_indices(ds, PER_CLASS)
    cluster_lookup = load_cluster_lookup()
    image_cluster_ids = [
        cluster_lookup.get(_rel_key(ds.samples[i][0]), f"singleton_{i}")
        for i in idxs
    ]

    transforms_list = ["brightness", "translation", "noise"]
    agg = {t: {"ssim": [], "pearson": [], "topk_iou": [], "pred_same": []}
           for t in transforms_list}

    rng = torch.Generator().manual_seed(0)
    for i in idxs:
        path = ds.samples[i][0]
        x0 = raw_tensor(path).unsqueeze(0)  # (1,3,224,224) in [0,1]
        xn = normalize(x0).to(DEVICE)
        with torch.no_grad():
            pred0 = model(xn).argmax(1).item()
        h0, _ = gcpp(model, cam, xn, pred0)

        for t in transforms_list:
            if t == "brightness":
                xt = torch.clamp(x0 * BRIGHT, 0, 1)
                realign = None
            elif t == "translation":
                xt = torch.roll(x0, shifts=(SHIFT, SHIFT), dims=(2, 3))
                realign = (SHIFT, SHIFT)
            else:  # noise
                noise = torch.randn(x0.shape, generator=rng) * NOISE
                xt = torch.clamp(x0 + noise, 0, 1)
                realign = None
            xtn = normalize(xt).to(DEVICE)
            ht, predt = gcpp(model, cam, xtn, pred0)
            if realign is not None:
                # shift the heatmap back so it aligns with the original frame
                ht = np.roll(ht, shift=(-realign[0], -realign[1]), axis=(0, 1))
            r, _ = pearsonr(h0.flatten(), ht.flatten())
            if np.isnan(r):
                r = 0.0
            agg[t]["ssim"].append(ssim(h0, ht))
            agg[t]["pearson"].append(float(r))
            agg[t]["topk_iou"].append(topk_iou(h0, ht))
            agg[t]["pred_same"].append(1.0 if predt == pred0 else 0.0)
    cam.remove_hooks()

    # Bootstrap 95% CI on each mean metric, resampling images with
    # replacement -- added in response to a review noting no XAI table
    # in the paper carried a confidence interval.
    N_BOOT = 2000
    boot_rng = np.random.default_rng(0)

    def bootstrap_ci(values):
        arr = np.array(values)
        n = len(arr)
        boot_means = [arr[boot_rng.integers(0, n, size=n)].mean() for _ in range(N_BOOT)]
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        return float(lo), float(hi)

    def group_bootstrap_ci(values, cluster_ids):
        # Same group-aware correction as compute_consistency.py /
        # compute_faithfulness.py: resample whole dihedral-hash duplicate
        # clusters rather than individual images.
        arr = np.array(values)
        unique_clusters = sorted(set(cluster_ids))
        if len(unique_clusters) < 2:
            return None
        cluster_to_indices = {c: [i for i, cid in enumerate(cluster_ids) if cid == c]
                               for c in unique_clusters}
        n_clusters = len(unique_clusters)
        boot_means = []
        for _ in range(N_BOOT):
            sampled_clusters = boot_rng.choice(unique_clusters, size=n_clusters, replace=True)
            sample_idx = [i for c in sampled_clusters for i in cluster_to_indices[c]]
            boot_means.append(arr[sample_idx].mean())
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        return float(lo), float(hi), n_clusters

    results = {"n_images": len(idxs), "shift_px": SHIFT,
               "brightness_mult": BRIGHT, "noise_std": NOISE,
               "bootstrap_n_resamples": N_BOOT, "transforms": {}}
    for t in transforms_list:
        ssim_ci = bootstrap_ci(agg[t]["ssim"])
        pearson_ci = bootstrap_ci(agg[t]["pearson"])
        iou_ci = bootstrap_ci(agg[t]["topk_iou"])
        ssim_group = group_bootstrap_ci(agg[t]["ssim"], image_cluster_ids)
        results["transforms"][t] = {
            "ssim_mean": float(np.mean(agg[t]["ssim"])),
            "ssim_std": float(np.std(agg[t]["ssim"])),
            "ssim_bootstrap_95ci": list(ssim_ci),
            "pearson_mean": float(np.mean(agg[t]["pearson"])),
            "pearson_std": float(np.std(agg[t]["pearson"])),
            "pearson_bootstrap_95ci": list(pearson_ci),
            "topk_iou_mean": float(np.mean(agg[t]["topk_iou"])),
            "topk_iou_std": float(np.std(agg[t]["topk_iou"])),
            "topk_iou_bootstrap_95ci": list(iou_ci),
            "pred_consistency": float(np.mean(agg[t]["pred_same"])),
        }
        m = results["transforms"][t]
        if ssim_group is not None:
            g_lo, g_hi, n_clusters = ssim_group
            m["ssim_group_bootstrap_95ci"] = [g_lo, g_hi]
            m["group_bootstrap_n_clusters"] = n_clusters
            width_ratio = (g_hi - g_lo) / (ssim_ci[1] - ssim_ci[0]) if ssim_ci[1] > ssim_ci[0] else None
            m["group_vs_image_width_ratio"] = width_ratio
            print(f"[stability] {t:12s} SSIM={m['ssim_mean']:.3f} image-CI [{ssim_ci[0]:.3f},{ssim_ci[1]:.3f}] "
                  f"group-CI [{g_lo:.3f},{g_hi:.3f}] ({n_clusters} clusters, {width_ratio:.2f}x width) "
                  f"Pearson={m['pearson_mean']:.3f} topkIoU={m['topk_iou_mean']:.3f} "
                  f"pred-consistency={m['pred_consistency']:.3f}")
        else:
            print(f"[stability] {t:12s} SSIM={m['ssim_mean']:.3f} [{ssim_ci[0]:.3f},{ssim_ci[1]:.3f}] "
                  f"Pearson={m['pearson_mean']:.3f} topkIoU={m['topk_iou_mean']:.3f} "
                  f"pred-consistency={m['pred_consistency']:.3f} (group bootstrap skipped)")

    with open(os.path.join(OUT_DIR, "stability.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = ["ssim_mean", "pearson_mean", "topk_iou_mean"]
    mlabels = ["SSIM", "Pearson r", "Top-10% IoU"]
    xg = np.arange(len(transforms_list))
    w = 0.25
    colors = ["#1768c4", "#0f6e5c", "#d9601a"]
    for j, (mk, ml) in enumerate(zip(metrics, mlabels)):
        vals = [results["transforms"][t][mk] for t in transforms_list]
        ax.bar(xg + (j - 1) * w, vals, w, label=ml, color=colors[j])
    ax.set_xticks(xg)
    ax.set_xticklabels([f"{t}\n(pred kept {results['transforms'][t]['pred_consistency']*100:.0f}%)"
                        for t in transforms_list], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("agreement with original heatmap")
    ax.set_title("Grad-CAM++ stability under clinically-invariant transforms")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "stability.png"), dpi=170)
    print("[stability] wrote outputs/phase0/stability.json and outputs/figures/stability.png")


if __name__ == "__main__":
    main()
