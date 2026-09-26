"""
figures_extended.py
-------------------
Generates the five additional publication-quality figures for the
FedGB extended experiments (E1–E5).

Figures produced:
  scalability.png   — Accuracy vs number of clients (N=3,10,50,100)
  fedbn.png         — FedAvg vs FedBN convergence + final-round bars
  dp.png            — DP accuracy-privacy trade-off (accuracy vs σ)
  mixed_skew.png    — Method comparison under label-only vs mixed skew
  ditto.png         — Global vs personalised accuracy per client (Ditto)

Run standalone:
    python figures_extended.py --results fedgb/outputs/extended_results.json

Or called programmatically:
    from figures_extended import generate_all_extended_figures
    generate_all_extended_figures("fedgb/outputs/extended_results.json")
"""

import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":  150,
})

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
          "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


# ─────────────────────────────────────────────────────────────────────────────
# E1 — Scalability
# ─────────────────────────────────────────────────────────────────────────────
def scalability(e1_data: dict, out_path: str):
    """
    Two-panel figure:
      Left:  Convergence curves for N=3,10,50,100 clients over rounds.
      Right: Bar chart of final-round accuracy + F1 vs N.
    """
    ns     = [3, 10, 50, 100]
    keys   = [f"n{n}" for n in ns]
    labels = [f"N={n}" for n in ns]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("E1 — Scalability: FedProx Accuracy vs Number of Clients",
                 fontsize=13, fontweight="bold")

    # Left: convergence curves
    ax = axes[0]
    for i, (k, lbl) in enumerate(zip(keys, labels)):
        if k not in e1_data:
            continue
        curve = e1_data[k]["accuracy_curve"]
        rounds = list(range(1, len(curve) + 1))
        ax.plot(rounds, curve, marker="o", markersize=4,
                color=COLORS[i % len(COLORS)], label=lbl, linewidth=1.8)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Test Accuracy")
    ax.set_title("Convergence Curves")
    ax.legend(framealpha=0.8)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)

    # Right: final accuracy + F1 bars
    ax2 = axes[1]
    present = [(k, lbl) for k, lbl in zip(keys, labels) if k in e1_data]
    x = np.arange(len(present))
    w = 0.35
    accs = [e1_data[k]["final_accuracy"] for k, _ in present]
    f1s  = [e1_data[k]["final_f1_macro"]  for k, _ in present]
    ax2.bar(x - w/2, accs, w, label="Accuracy",  color="#4C72B0", alpha=0.85)
    ax2.bar(x + w/2, f1s,  w, label="Macro-F1",  color="#DD8452", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in present])
    ax2.set_ylabel("Score")
    ax2.set_title("Final-Round Performance vs N")
    ax2.legend()
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", alpha=0.3)
    for xi, (a, f) in enumerate(zip(accs, f1s)):
        ax2.text(xi - w/2, a + 0.01, f"{a:.3f}", ha="center",
                 va="bottom", fontsize=8)
        ax2.text(xi + w/2, f + 0.01, f"{f:.3f}", ha="center",
                 va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Figures] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# E2 — FedBN
# ─────────────────────────────────────────────────────────────────────────────
def fedbn(e2_data: dict, out_path: str):
    """
    Two-panel figure:
      Left:  Convergence curves FedAvg vs FedBN.
      Right: Grouped bar (Accuracy / F1 / AUC) for FedAvg vs FedBN.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("E2 — FedBN: Local vs Global Batch-Normalisation Aggregation",
                 fontsize=13, fontweight="bold")

    col_map = {"fedavg": "#4C72B0", "fedbn": "#DD8452"}
    nice    = {"fedavg": "FedAvg (global BN)", "fedbn": "FedBN (local BN)"}

    # Left: convergence
    ax = axes[0]
    for k in ("fedavg", "fedbn"):
        if k not in e2_data:
            continue
        curve = e2_data[k]["accuracy_curve"]
        rounds = list(range(1, len(curve) + 1))
        ax.plot(rounds, curve, marker="o", markersize=4,
                color=col_map[k], label=nice[k], linewidth=1.8)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Test Accuracy")
    ax.set_title("Convergence Curves (N=3, Label Skew)")
    ax.legend(framealpha=0.8)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)

    # Right: final metrics bar
    ax2 = axes[1]
    metrics_keys = ["final_accuracy", "final_f1_macro", "final_auc_macro"]
    metric_labels = ["Accuracy", "Macro-F1", "AUC-ROC"]
    strats_present = [k for k in ("fedavg", "fedbn") if k in e2_data]
    x  = np.arange(len(metric_labels))
    w  = 0.35
    for si, strat in enumerate(strats_present):
        vals = [e2_data[strat][m] for m in metrics_keys]
        offset = (si - (len(strats_present) - 1) / 2) * w
        bars = ax2.bar(x + offset, vals, w, label=nice[strat],
                       color=col_map[strat], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_labels)
    ax2.set_ylabel("Score")
    ax2.set_title("Final-Round Metrics")
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Figures] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# E3 — Differential Privacy trade-off
# ─────────────────────────────────────────────────────────────────────────────
def dp(e3_data: dict, out_path: str):
    """
    Two-panel figure:
      Left:  Accuracy vs noise multiplier σ (privacy trade-off curve).
      Right: F1 and AUC vs σ.
    """
    sigma_map = {
        "no_dp":   0.0,
        "sigma0.1": 0.1,
        "sigma0.5": 0.5,
        "sigma1.0": 1.0,
        "sigma2.0": 2.0,
        "sigma5.0": 5.0,
    }
    sigmas, accs, f1s, aucs = [], [], [], []
    for tag, sigma in sorted(sigma_map.items(), key=lambda x: x[1]):
        if tag not in e3_data:
            continue
        sigmas.append(sigma)
        accs.append(e3_data[tag]["final_accuracy"])
        f1s.append(e3_data[tag]["final_f1_macro"])
        aucs.append(e3_data[tag]["final_auc_macro"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "E3 — Differential Privacy: Accuracy–Privacy Trade-Off "
        "(DP-FedAvg, N=3)", fontsize=13, fontweight="bold")

    sigma_labels = ["No DP\n(ε=∞)"] + [f"σ={s}" for s in sigmas[1:]]

    for ax, vals, ylabel, title, color in [
        (axes[0], accs, "Test Accuracy",  "Accuracy vs Noise Level",  "#4C72B0"),
        (axes[1], [f1s, aucs], "Score", "F1 & AUC vs Noise Level", "#55A868"),
    ]:
        if ylabel == "Test Accuracy":
            ax.plot(range(len(sigmas)), vals, marker="s", linewidth=2,
                    color=color, markersize=6)
            # Highlight the no-DP point
            ax.axhline(vals[0], color="gray", linestyle="--", alpha=0.6,
                       label=f"No DP acc={vals[0]:.3f}")
            ax.set_xticks(range(len(sigmas)))
            ax.set_xticklabels(sigma_labels)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1.0)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.015, f"{v:.3f}", ha="center",
                        fontsize=8, color=color)
        else:
            f1_vals, auc_vals = vals
            ax.plot(range(len(sigmas)), f1_vals,  marker="o", linewidth=2,
                    color="#DD8452", label="Macro-F1", markersize=6)
            ax.plot(range(len(sigmas)), auc_vals, marker="^", linewidth=2,
                    color="#55A868", label="AUC-ROC",  markersize=6)
            ax.set_xticks(range(len(sigmas)))
            ax.set_xticklabels(sigma_labels)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Figures] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# E4 — Mixed-skew
# ─────────────────────────────────────────────────────────────────────────────
def mixed_skew(e4_data: dict, out_path: str):
    """
    Grouped bar chart comparing FedProx under different skew conditions
    (label-only vs mixed) for N=3 and N=10.
    """
    keys  = ["n3_label_skew", "n3_mixed_skew",
             "n10_dirichlet",  "n10_mixed_skew"]
    nices = ["N=3 Label\nSkew",   "N=3 Mixed\nSkew",
             "N=10 Dirichlet\n(Label)", "N=10 Mixed\nSkew"]

    metrics_keys   = ["final_accuracy", "final_f1_macro", "final_auc_macro"]
    metric_labels  = ["Accuracy", "Macro-F1", "AUC-ROC"]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(
        "E4 — Mixed-Skew: FedProx under Label-Only vs Combined Skew",
        fontsize=13, fontweight="bold")

    present = [(k, n) for k, n in zip(keys, nices) if k in e4_data]
    n_conf  = len(present)
    n_met   = len(metric_labels)
    x       = np.arange(n_met)
    w       = 0.8 / n_conf

    for ci, (k, lbl) in enumerate(present):
        offset = (ci - (n_conf - 1) / 2) * w
        vals   = [e4_data[k][m] for m in metrics_keys]
        bars   = ax.bar(x + offset, vals, w, label=lbl,
                        color=COLORS[ci % len(COLORS)], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Figures] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# E5 — Ditto personalised FL
# ─────────────────────────────────────────────────────────────────────────────
def ditto(e5_data: dict, out_path: str):
    """
    Two-panel figure:
      Left:  Convergence curves (global model) for FedAvg / FedProx / Ditto.
      Right: Per-client personalised accuracy (Ditto) vs global baselines.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "E5 — Personalised FL: Ditto vs Global Baselines (N=3, Label Skew)",
        fontsize=13, fontweight="bold")

    col_map  = {"fedavg": "#4C72B0", "fedprox": "#DD8452", "ditto": "#55A868"}
    nice_map = {"fedavg": "FedAvg", "fedprox": "FedProx", "ditto": "Ditto (global)"}

    # Left: convergence
    ax = axes[0]
    for k in ("fedavg", "fedprox", "ditto"):
        if k not in e5_data:
            continue
        curve  = e5_data[k]["accuracy_curve"]
        rounds = list(range(1, len(curve) + 1))
        ax.plot(rounds, curve, marker="o", markersize=4,
                color=col_map[k], label=nice_map[k], linewidth=1.8)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Test Accuracy")
    ax.set_title("Global Model Convergence")
    ax.legend(framealpha=0.8)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)

    # Right: per-client personalised vs global
    ax2 = axes[1]
    if "ditto" in e5_data:
        pm = e5_data["ditto"]["personal_metrics"]
        client_ids  = [f"Client {m['client_id']}" for m in pm]
        pers_accs   = [m["accuracy"] for m in pm]
        global_acc  = e5_data["ditto"]["final_accuracy"]
        fa_acc      = e5_data["fedavg"]["final_accuracy"]  if "fedavg"  in e5_data else 0
        fp_acc      = e5_data["fedprox"]["final_accuracy"] if "fedprox" in e5_data else 0

        x  = np.arange(len(client_ids))
        ax2.bar(x, pers_accs, color="#55A868", alpha=0.85, label="Ditto (personal)")
        ax2.axhline(global_acc, color="#55A868", linestyle="--", linewidth=1.5,
                    label=f"Ditto global ({global_acc:.3f})")
        ax2.axhline(fa_acc, color="#4C72B0", linestyle=":",  linewidth=1.5,
                    label=f"FedAvg global ({fa_acc:.3f})")
        ax2.axhline(fp_acc, color="#DD8452", linestyle="-.", linewidth=1.5,
                    label=f"FedProx global ({fp_acc:.3f})")
        for xi, v in enumerate(pers_accs):
            ax2.text(xi, v + 0.01, f"{v:.3f}", ha="center",
                     va="bottom", fontsize=9)
        ax2.set_xticks(x)
        ax2.set_xticklabels(client_ids)
        ax2.set_ylabel("Test Accuracy")
        ax2.set_title("Per-Client Personalised vs Global Accuracy")
        ax2.legend(fontsize=8.5, framealpha=0.8)
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Ditto results not available",
                 ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[Figures] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Master generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_extended_figures(extended_results_path: str):
    """Load extended_results.json and generate all E1–E5 figures."""
    with open(extended_results_path) as f:
        data = json.load(f)

    if "E1_scalability" in data:
        scalability(
            data["E1_scalability"],
            os.path.join(OUTPUT_DIR, "scalability.png"))
    if "E2_fedbn" in data:
        fedbn(
            data["E2_fedbn"],
            os.path.join(OUTPUT_DIR, "fedbn.png"))
    if "E3_dp" in data:
        dp(
            data["E3_dp"],
            os.path.join(OUTPUT_DIR, "dp.png"))
    if "E4_mixed_skew" in data:
        mixed_skew(
            data["E4_mixed_skew"],
            os.path.join(OUTPUT_DIR, "mixed_skew.png"))
    if "E5_ditto" in data:
        ditto(
            data["E5_ditto"],
            os.path.join(OUTPUT_DIR, "ditto.png"))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str,
                        default="./outputs/extended_results.json")
    args = parser.parse_args()
    generate_all_extended_figures(args.results)
    print("[Done] All extended figures saved to", OUTPUT_DIR)
