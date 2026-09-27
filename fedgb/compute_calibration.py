"""
compute_calibration.py
----------------------
Probability calibration (Guo et al., ICML 2017): a model can be accurate yet
badly miscalibrated, so its confidence is not a trustworthy probability. We
report Expected Calibration Error (ECE, 15 bins) and Brier score before and
after temperature scaling, with a reliability diagram. Temperature is fit on
one stratified half of the validation set and evaluated on the held-out half
so the reported numbers are not fit on the same data they score.

This directly resolves the "uncalibrated raw-softmax" caveat flagged in the
paper's failure-case section.

Run from fedgb/:  python compute_calibration.py
Outputs: outputs/phase0/calibration.json, outputs/figures/calibration.png
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from _p0common import (load_primary_model, val_dataset, DEVICE, OUT_DIR, FIG_DIR)

N_BINS = 15


def get_logits(model, loader):
    logits, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(DEVICE)).cpu())
            labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def ece_and_brier(probs, labels, n_bins=N_BINS):
    conf, pred = probs.max(1)
    acc = pred.eq(labels).float()
    ece = 0.0
    bins = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += (m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()).item()
    onehot = F.one_hot(labels, probs.shape[1]).float()
    brier = ((probs - onehot) ** 2).sum(1).mean().item()
    return ece, brier


def reliability(probs, labels, n_bins=N_BINS):
    conf, pred = probs.max(1)
    acc = pred.eq(labels).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean().item())
            ys.append(acc[m].mean().item())
    return xs, ys


def fit_temperature(logits, labels):
    T = torch.nn.Parameter(torch.ones(1) * 1.0)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=100)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / T.clamp_min(1e-3), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp_min(1e-3).item())


def main():
    model = load_primary_model()
    ds = val_dataset()

    # stratified 50/50 split into calibration- and test-halves
    random.seed(42)
    by = {}
    for i, (_, l) in enumerate(ds.samples):
        by.setdefault(l, []).append(i)
    calib_idx, test_idx = [], []
    for l, idxs in by.items():
        random.shuffle(idxs)
        h = len(idxs) // 2
        calib_idx += idxs[:h]
        test_idx += idxs[h:]

    calib_loader = DataLoader(Subset(ds, calib_idx), batch_size=128)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=128)

    calib_logits, calib_labels = get_logits(model, calib_loader)
    test_logits, test_labels = get_logits(model, test_loader)

    T = fit_temperature(calib_logits, calib_labels)

    probs_before = F.softmax(test_logits, dim=1)
    probs_after = F.softmax(test_logits / T, dim=1)

    ece_b, brier_b = ece_and_brier(probs_before, test_labels)
    ece_a, brier_a = ece_and_brier(probs_after, test_labels)

    results = {
        "temperature": T,
        "n_calib": len(calib_idx), "n_test": len(test_idx),
        "ece_before": ece_b, "ece_after": ece_a,
        "brier_before": brier_b, "brier_after": brier_a,
        "accuracy_test": float(probs_before.argmax(1).eq(test_labels).float().mean()),
        "interpretation": (
            f"Temperature scaling (T={T:.2f}) reduces ECE from {ece_b:.3f} to "
            f"{ece_a:.3f} and Brier score from {brier_b:.3f} to {brier_a:.3f} on "
            f"the held-out validation half, without changing discrimination "
            f"(accuracy is unchanged by construction)."
        ),
    }
    print(f"[calibration] T={T:.3f}")
    print(f"[calibration] ECE  {ece_b:.4f} -> {ece_a:.4f}")
    print(f"[calibration] Brier {brier_b:.4f} -> {brier_a:.4f}")

    with open(os.path.join(OUT_DIR, "calibration.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- reliability diagram ----
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect calibration")
    xb, yb = reliability(probs_before, test_labels)
    xa, ya = reliability(probs_after, test_labels)
    ax.plot(xb, yb, "o-", color="#c0392b", label=f"before (ECE={ece_b:.3f})")
    ax.plot(xa, ya, "s-", color="#1768c4", label=f"after T={T:.2f} (ECE={ece_a:.3f})")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Reliability diagram (held-out validation half)")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "calibration.png"), dpi=170)
    print("[calibration] wrote outputs/phase0/calibration.json and outputs/figures/calibration.png")


if __name__ == "__main__":
    main()
