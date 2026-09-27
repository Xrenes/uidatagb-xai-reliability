"""
compute_faithfulness.py
-----------------------
Deletion / insertion faithfulness curves (Petsiuk et al., 2018; Hooker et
al. ROAR, 2019). This is the test that tells us whether a heatmap reflects
what the model actually USES, rather than merely what looks anatomically
plausible.

  Deletion : starting from the original image, progressively replace the
             most-attributed pixels with a neutral baseline and watch the
             predicted-class probability fall. A faithful map makes it fall
             FAST -> low deletion AUC is better.
  Insertion: starting from a blurred baseline, progressively restore the
             most-attributed pixels and watch the probability rise. A
             faithful map makes it rise FAST -> high insertion AUC is better.

Each method is compared against a random-ordering control on the same
images: a faithful method must beat random.

Run from fedgb/:  python compute_faithfulness.py
Outputs: outputs/phase0/faithfulness.json, outputs/figures/faithfulness.png
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _p0common import (load_primary_model, val_dataset, stratified_indices,
                       upsample224, normalize, raw_tensor,
                       get_target_layer, DEVICE, OUT_DIR, FIG_DIR)
from gradcam import GradCAM, GradCAMPlusPlus, ScoreCAM

PER_CLASS = 16          # 16 x 9 classes = 144 images
STEPS = 20              # number of reveal/removal steps
N_PIX = 224 * 224


def make_baseline(x0):
    """Blurred version of the image, used as deletion target / insertion start."""
    k = 15
    pad = k // 2
    blur = F.avg_pool2d(F.pad(x0, (pad, pad, pad, pad), mode="reflect"), k, stride=1)
    return blur


def curve_for_order(model, x0_norm, base_norm, order, target, mode):
    """Return probability trajectory over STEPS for a given pixel order."""
    B = STEPS + 1
    flat_x = x0_norm.view(3, -1)
    flat_b = base_norm.view(3, -1)
    imgs = torch.empty((B, 3, N_PIX), device=DEVICE)
    step = N_PIX // STEPS
    for s in range(B):
        n = s * step
        if mode == "deletion":
            # start from original, replace top-n attributed pixels with baseline
            img = flat_x.clone()
            if n > 0:
                idx = order[:n]
                img[:, idx] = flat_b[:, idx]
        else:  # insertion: start from baseline, restore top-n attributed pixels
            img = flat_b.clone()
            if n > 0:
                idx = order[:n]
                img[:, idx] = flat_x[:, idx]
        imgs[s] = img
    imgs = imgs.view(B, 3, 224, 224)
    with torch.no_grad():
        probs = F.softmax(model(imgs), dim=1)[:, target].cpu().numpy()
    return probs


def auc(curve):
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(_trap(curve, dx=1.0 / (len(curve) - 1)))


def main():
    model = load_primary_model()
    ds = val_dataset()
    idxs = stratified_indices(ds, PER_CLASS)

    explainers = {
        "Grad-CAM": lambda: GradCAM(model, get_target_layer(model)),
        "Grad-CAM++": lambda: GradCAMPlusPlus(model, get_target_layer(model)),
        "Score-CAM": lambda: ScoreCAM(model, get_target_layer(model),
                                      max_channels=64, batch_size=128),
    }

    res = {m: {"deletion": [], "insertion": []} for m in explainers}
    res["Random"] = {"deletion": [], "insertion": []}
    del_curves = {m: [] for m in list(explainers) + ["Random"]}
    ins_curves = {m: [] for m in list(explainers) + ["Random"]}

    rng = np.random.default_rng(0)

    for n_done, i in enumerate(idxs):
        path = ds.samples[i][0]
        x0 = raw_tensor(path).unsqueeze(0)
        x0n = normalize(x0).to(DEVICE)
        basen = normalize(make_baseline(x0)).to(DEVICE)
        with torch.no_grad():
            target = model(x0n).argmax(1).item()

        # random order (shared control for this image)
        rand_order = torch.tensor(rng.permutation(N_PIX), device=DEVICE)
        for mode, store in (("deletion", del_curves), ("insertion", ins_curves)):
            c = curve_for_order(model, x0n[0], basen[0], rand_order, target, mode)
            store["Random"].append(c)
            res["Random"][mode].append(auc(c))

        for name, factory in explainers.items():
            cam = factory()
            hm, _, _ = cam.generate(x0n, target_class=target)
            cam.remove_hooks()
            hm = upsample224(hm)
            order = torch.tensor(np.argsort(hm.flatten())[::-1].copy(), device=DEVICE)
            for mode, store in (("deletion", del_curves), ("insertion", ins_curves)):
                c = curve_for_order(model, x0n[0], basen[0], order, target, mode)
                store[name].append(c)
                res[name][mode].append(auc(c))
        if (n_done + 1) % 20 == 0:
            print(f"[faithfulness] {n_done + 1}/{len(idxs)} images done")

    # Bootstrap 95% CI on the faithfulness gap (insertion - deletion),
    # resampling images with replacement, paired within each resample so
    # the deletion/insertion AUCs for the same resampled image stay
    # linked -- added in response to a review noting close faithfulness
    # gaps (Grad-CAM 0.480 vs Grad-CAM++ 0.473 vs Score-CAM 0.456) were
    # never accompanied by an interval showing whether that ranking is
    # actually distinguishable from noise.
    N_BOOT = 2000
    boot_rng = np.random.default_rng(0)

    def bootstrap_gap_ci(del_arr, ins_arr):
        n = len(del_arr)
        boot_gaps = []
        for _ in range(N_BOOT):
            sample_idx = boot_rng.integers(0, n, size=n)
            boot_gaps.append(ins_arr[sample_idx].mean() - del_arr[sample_idx].mean())
        lo, hi = np.percentile(boot_gaps, [2.5, 97.5])
        return float(lo), float(hi)

    summary = {"n_images": len(idxs), "steps": STEPS,
               "bootstrap_n_resamples": N_BOOT, "methods": {}}
    for m in list(explainers) + ["Random"]:
        d = np.array(res[m]["deletion"])
        ins = np.array(res[m]["insertion"])
        gap_ci_lo, gap_ci_hi = bootstrap_gap_ci(d, ins)
        summary["methods"][m] = {
            "deletion_auc_mean": float(d.mean()), "deletion_auc_std": float(d.std()),
            "insertion_auc_mean": float(ins.mean()), "insertion_auc_std": float(ins.std()),
            # single headline number: higher = more faithful
            "faithfulness_gap": float(ins.mean() - d.mean()),
            "faithfulness_gap_bootstrap_95ci": [gap_ci_lo, gap_ci_hi],
        }
        s = summary["methods"][m]
        print(f"[faithfulness] {m:11s} del-AUC={s['deletion_auc_mean']:.3f} "
              f"ins-AUC={s['insertion_auc_mean']:.3f} gap={s['faithfulness_gap']:.3f} "
              f"95% CI [{gap_ci_lo:.3f}, {gap_ci_hi:.3f}]")

    with open(os.path.join(OUT_DIR, "faithfulness.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- figure: mean deletion & insertion curves ----
    fig, (axd, axi) = plt.subplots(1, 2, figsize=(10, 4))
    xs = np.linspace(0, 1, STEPS + 1)
    colors = {"Grad-CAM": "#1768c4", "Grad-CAM++": "#0f6e5c",
              "Score-CAM": "#d9601a", "Random": "#888888"}
    for m in ["Grad-CAM", "Grad-CAM++", "Score-CAM", "Random"]:
        dc = np.array(del_curves[m]).mean(0)
        ic = np.array(ins_curves[m]).mean(0)
        ls = "--" if m == "Random" else "-"
        axd.plot(xs, dc, ls, color=colors[m], label=m)
        axi.plot(xs, ic, ls, color=colors[m], label=m)
    axd.set_title("Deletion (lower AUC = better)")
    axd.set_xlabel("fraction of pixels removed")
    axd.set_ylabel("predicted-class probability")
    axi.set_title("Insertion (higher AUC = better)")
    axi.set_xlabel("fraction of pixels inserted")
    axd.legend(fontsize=8)
    axi.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "faithfulness.png"), dpi=170)
    print("[faithfulness] wrote outputs/phase0/faithfulness.json and outputs/figures/faithfulness.png")


if __name__ == "__main__":
    main()
