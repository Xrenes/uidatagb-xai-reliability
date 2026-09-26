"""
run_experiments.py
------------------
Orchestration script that executes all five pre-publication
experimental extensions for the FedGB paper.

Experiments run (in order, each saves its own results JSON):
  E1  — Scalability: FedAvg with N=10 / 50 / 100 clients (Dirichlet α=0.5)
  E2  — FedBN:      3-client label-skew, keep BN layers local
  E3  — DP-FedAvg:  3-client, vary ε-proxy via noise_multiplier ∈ {0.1,0.5,1,2,5}
  E4  — Mixed-skew: FedProx + SCAFFOLD under label+quantity skew (N=3,10)
  E5  — Ditto:      Personalised FL baseline (N=3, label_skew)

Results are collected and saved to:
  fedgb/outputs/extended_results.json

Usage (run from fedgb/):
    python run_experiments.py [--rounds 10] [--skip_gradcam]
    python run_experiments.py --rounds 5 --skip_gradcam   # fast ablation
    python run_experiments.py --only E3                   # run one group
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Allow imports from fedgb/ when run from within fedgb/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_client_datasets, CLASS_NAMES, NUM_CLASSES
from model import build_model, get_model_parameters, set_model_parameters
from evaluate import run_inference, compute_metrics, print_metrics
from run_simulation import (
    train_client, eval_model,
    fedavg_aggregate, fedbn_aggregate, dp_fedavg_aggregate, ditto_personalize,
    get_non_bn_indices, write_progress,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation loop (reusable for all experiments)
# ─────────────────────────────────────────────────────────────────────────────
def run_fl(
    data_dir: str,
    num_clients: int,
    rounds: int,
    strategy: str,
    partition: str = "label_skew",
    proximal_mu: float = 0.01,
    dp_noise_multiplier: float = 1.0,
    dp_max_grad_norm: float = 1.0,
    ditto_lambda: float = 0.1,
    ditto_personal_epochs: int = 2,
    dirichlet_alpha: float = 0.5,
    mixed_alpha_label: float = 0.5,
    mixed_alpha_qty: float = 0.3,
    mixed_noise_std: float = 0.05,
    batch_size: int = 16,
    local_epochs: int = 3,
    seed: int = 42,
) -> dict:
    """
    Run one complete FL simulation and return a metrics dictionary.
    Internally mirrors the logic of run_simulation.py::main() but
    is callable as a Python function (no subprocess).
    """
    # ── Data ────────────────────────────────────────────────────────────────
    client_datasets, test_dataset = get_client_datasets(
        data_dir,
        num_clients=num_clients,
        partition=partition,
        dirichlet_alpha=dirichlet_alpha,
        mixed_alpha_label=mixed_alpha_label,
        mixed_alpha_qty=mixed_alpha_qty,
        mixed_noise_std=mixed_noise_std,
        seed=seed,
    )
    client_loaders = []
    for cdata in client_datasets:
        n_val   = max(1, int(0.2 * len(cdata)))
        n_train = len(cdata) - n_val
        tr, va  = random_split(cdata, [n_train, n_val],
                               generator=torch.Generator().manual_seed(seed))
        client_loaders.append((
            DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=0),
            DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=0),
        ))
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    # ── Model / aux structures ───────────────────────────────────────────────
    global_model = build_model(pretrained=True).to(DEVICE)
    criterion    = nn.CrossEntropyLoss()
    non_bn_idx   = get_non_bn_indices(global_model) if strategy == "fedbn" else set()
    personal_models = (
        [copy.deepcopy(global_model) for _ in range(num_clients)]
        if strategy == "ditto" else []
    )

    accuracy_curve, loss_curve = [], []

    # ── FL rounds ────────────────────────────────────────────────────────────
    for rnd in range(1, rounds + 1):
        global_params_snapshot = [p.detach().clone()
                                   for p in global_model.parameters()]
        results = []
        for cid, (train_ldr, _) in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model)
            optimizer   = torch.optim.Adam(
                local_model.parameters(), lr=1e-4, weight_decay=1e-4)
            train_client(
                local_model, train_ldr, optimizer, criterion,
                epochs=local_epochs,
                proximal_mu=(proximal_mu
                             if strategy in ("fedprox", "dp_fedprox") else 0.0),
                global_params=global_params_snapshot,
                desc=f"  [{strategy}] R{rnd} C{cid}",
            )
            results.append((get_model_parameters(local_model),
                             len(train_ldr.dataset)))

        # Aggregation
        global_params_np = get_model_parameters(global_model)
        if strategy in ("fedavg", "fedprox", "ditto"):
            agg = fedavg_aggregate(results)
        elif strategy == "fedbn":
            agg = fedbn_aggregate(results, non_bn_idx, global_params_np)
        elif strategy in ("dp_fedavg", "dp_fedprox"):
            agg = dp_fedavg_aggregate(
                results, global_params_np,
                max_grad_norm=dp_max_grad_norm,
                noise_multiplier=dp_noise_multiplier)
        else:
            agg = fedavg_aggregate(results)
        set_model_parameters(global_model, agg)

        # Ditto personalisation step
        if strategy == "ditto":
            gp = get_model_parameters(global_model)
            for cid, (tl, _) in enumerate(client_loaders):
                ditto_personalize(personal_models[cid], gp, tl, criterion,
                                  lambda_reg=ditto_lambda,
                                  epochs=ditto_personal_epochs)

        # Global eval
        g_loss, g_acc = eval_model(global_model, test_loader, criterion)
        accuracy_curve.append(round(float(g_acc), 4))
        loss_curve.append(round(float(g_loss), 4))
        print(f"    R{rnd:02d}/{rounds}  acc={g_acc:.4f}  loss={g_loss:.4f}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    y_true, y_pred, y_probs = run_inference(global_model, test_loader)
    metrics = compute_metrics(y_true, y_pred, y_probs)

    # Ditto per-client personalised evaluation
    personal_metrics_list = []
    if strategy == "ditto":
        for cid, pm in enumerate(personal_models):
            yt, yp, ypr = run_inference(pm, test_loader)
            pm_m = compute_metrics(yt, yp, ypr)
            personal_metrics_list.append({
                "client_id": cid,
                "accuracy":  float(pm_m["accuracy"]),
                "f1_macro":  float(pm_m["f1_macro"]),
                "auc_macro": float(pm_m["auc_macro"]),
            })

    return {
        "strategy":           strategy,
        "partition":          partition,
        "num_clients":        num_clients,
        "rounds":             rounds,
        "accuracy_curve":     accuracy_curve,
        "loss_curve":         loss_curve,
        "final_accuracy":     float(metrics["accuracy"]),
        "final_f1_macro":     float(metrics["f1_macro"]),
        "final_auc_macro":    float(metrics["auc_macro"]),
        "f1_per_class":       [float(v) for v in metrics["f1_per_class"]],
        "auc_per_class":      [float(v) for v in metrics["auc_per_class"]],
        "sensitivity":        [float(v) for v in metrics["sensitivity"]],
        "specificity":        [float(v) for v in metrics["specificity"]],
        "personal_metrics":   personal_metrics_list,
        # Hyperparams for reference
        "proximal_mu":        proximal_mu,
        "dp_noise_multiplier":dp_noise_multiplier,
        "dp_max_grad_norm":   dp_max_grad_norm,
        "ditto_lambda":       ditto_lambda,
        "dirichlet_alpha":    dirichlet_alpha,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Individual experiment groups
# ─────────────────────────────────────────────────────────────────────────────
def exp_e1_scalability(data_dir, rounds, batch_size, local_epochs):
    """E1 — Scale clients: N=3 (baseline) / 10 / 50 / 100 with Dirichlet α=0.5."""
    print("\n" + "="*60)
    print("  E1 — Scalability: N=3 / 10 / 50 / 100 clients")
    print("="*60)
    results = {}
    for n in [3, 10, 50, 100]:
        part = "label_skew" if n == 3 else "dirichlet"
        print(f"\n  [E1] num_clients={n}  partition={part}")
        r = run_fl(data_dir, num_clients=n, rounds=rounds,
                   strategy="fedprox",
                   partition=part,
                   proximal_mu=0.01,
                   batch_size=batch_size, local_epochs=local_epochs)
        results[f"n{n}"] = r
        print(f"  [E1] n={n:3d}  final_acc={r['final_accuracy']:.4f}"
              f"  F1={r['final_f1_macro']:.4f}  AUC={r['final_auc_macro']:.4f}")
    return results


def exp_e2_fedbn(data_dir, rounds, batch_size, local_epochs):
    """E2 — FedBN: compare FedAvg vs FedBN (3 clients, label skew)."""
    print("\n" + "="*60)
    print("  E2 — FedBN: local BN vs global BN aggregation")
    print("="*60)
    results = {}
    for strat in ["fedavg", "fedbn"]:
        print(f"\n  [E2] strategy={strat}")
        r = run_fl(data_dir, num_clients=3, rounds=rounds,
                   strategy=strat, partition="label_skew",
                   batch_size=batch_size, local_epochs=local_epochs)
        results[strat] = r
        print(f"  [E2] {strat:6s}  final_acc={r['final_accuracy']:.4f}"
              f"  F1={r['final_f1_macro']:.4f}  AUC={r['final_auc_macro']:.4f}")
    return results


def exp_e3_dp(data_dir, rounds, batch_size, local_epochs):
    """E3 — Differential Privacy: vary noise_multiplier ∈ {0.1,0.5,1,2,5,∞}.
    ∞ corresponds to noise_multiplier=0 (no DP = standard FedAvg).
    The effective ε decreases as sigma increases (tighter privacy, lower acc)."""
    print("\n" + "="*60)
    print("  E3 — DP-FedAvg: accuracy vs privacy noise level")
    print("="*60)
    results = {}
    # noise_multiplier=0.0 ≡ no DP (ε=∞)
    noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    for sigma in noise_levels:
        tag = "no_dp" if sigma == 0.0 else f"sigma{sigma}"
        strat = "fedavg" if sigma == 0.0 else "dp_fedavg"
        print(f"\n  [E3] sigma={sigma}  strategy={strat}")
        r = run_fl(data_dir, num_clients=3, rounds=rounds,
                   strategy=strat, partition="label_skew",
                   dp_noise_multiplier=sigma, dp_max_grad_norm=1.0,
                   batch_size=batch_size, local_epochs=local_epochs)
        r["dp_sigma_label"] = f"σ={sigma}" if sigma > 0 else "No DP (ε=∞)"
        results[tag] = r
        print(f"  [E3] sigma={sigma:.1f}  final_acc={r['final_accuracy']:.4f}"
              f"  F1={r['final_f1_macro']:.4f}  AUC={r['final_auc_macro']:.4f}")
    return results


def exp_e4_mixed_skew(data_dir, rounds, batch_size, local_epochs):
    """E4 — Mixed-skew: label_skew vs mixed_skew, for FedProx (N=3 and N=10)."""
    print("\n" + "="*60)
    print("  E4 — Mixed-skew: label-only vs label+quantity+feature skew")
    print("="*60)
    results = {}
    for n, part in [(3, "label_skew"), (3, "mixed_skew"),
                    (10, "dirichlet"), (10, "mixed_skew")]:
        key = f"n{n}_{part}"
        print(f"\n  [E4] num_clients={n}  partition={part}")
        r = run_fl(data_dir, num_clients=n, rounds=rounds,
                   strategy="fedprox",
                   partition=part,
                   proximal_mu=0.01,
                   mixed_alpha_label=0.5,
                   mixed_alpha_qty=0.3,
                   mixed_noise_std=0.05,
                   batch_size=batch_size, local_epochs=local_epochs)
        results[key] = r
        print(f"  [E4] {key:25s}  final_acc={r['final_accuracy']:.4f}"
              f"  F1={r['final_f1_macro']:.4f}  AUC={r['final_auc_macro']:.4f}")
    return results


def exp_e5_ditto(data_dir, rounds, batch_size, local_epochs):
    """E5 — Personalised FL: FedAvg vs FedProx vs Ditto (N=3, label_skew)."""
    print("\n" + "="*60)
    print("  E5 — Personalised FL: Ditto vs global baselines")
    print("="*60)
    results = {}
    for strat, mu, lam in [("fedavg", 0.0, 0.0),
                            ("fedprox", 0.01, 0.0),
                            ("ditto", 0.0, 0.1)]:
        tag = strat
        print(f"\n  [E5] strategy={strat}")
        r = run_fl(data_dir, num_clients=3, rounds=rounds,
                   strategy=strat, partition="label_skew",
                   proximal_mu=mu, ditto_lambda=lam,
                   ditto_personal_epochs=2,
                   batch_size=batch_size, local_epochs=local_epochs)
        results[tag] = r
        print(f"  [E5] {strat:8s}  global_acc={r['final_accuracy']:.4f}"
              f"  F1={r['final_f1_macro']:.4f}  AUC={r['final_auc_macro']:.4f}")
        if r["personal_metrics"]:
            avg_p = np.mean([m["accuracy"] for m in r["personal_metrics"]])
            print(f"  [E5] Ditto avg personal acc = {avg_p:.4f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Run all 5 FedGB experimental extensions.")
    parser.add_argument("--data_dir",    type=str, default="./data/data")
    parser.add_argument("--rounds",      type=int, default=10)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--local_epochs",type=int, default=3)
    parser.add_argument("--skip_gradcam", action="store_true")
    parser.add_argument(
        "--only", type=str, default="",
        help="Run only one experiment group: E1, E2, E3, E4, or E5 "
             "(default: run all).",
    )
    args = parser.parse_args()

    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  FedGB Extended Experiments")
    print(f"  Device  : {DEVICE}")
    print(f"  Rounds  : {args.rounds}")
    print(f"  Epochs  : {args.local_epochs}")
    print(f"  Batch   : {args.batch_size}")
    print(f"{'='*60}")

    kw = dict(data_dir=args.data_dir, rounds=args.rounds,
              batch_size=args.batch_size, local_epochs=args.local_epochs)

    only = args.only.upper().strip()
    extended = {}

    if not only or only == "E1":
        extended["E1_scalability"] = exp_e1_scalability(**kw)
    if not only or only == "E2":
        extended["E2_fedbn"]       = exp_e2_fedbn(**kw)
    if not only or only == "E3":
        extended["E3_dp"]          = exp_e3_dp(**kw)
    if not only or only == "E4":
        extended["E4_mixed_skew"]  = exp_e4_mixed_skew(**kw)
    if not only or only == "E5":
        extended["E5_ditto"]       = exp_e5_ditto(**kw)

    # ── Save master results ────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "extended_results.json")
    with open(out_path, "w") as f:
        json.dump(extended, f, indent=2)
    print(f"\n[Done] Extended results → {out_path}")
    print(f"[Done] Total wall-clock: {(time.time()-t0)/60:.1f} min")

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"  {'Experiment':<30}  {'Config':<20}  {'Acc':>6}  {'F1':>6}  {'AUC':>6}")
    print("="*72)
    for group, group_data in extended.items():
        for tag, r in group_data.items():
            conf = f"N={r['num_clients']} {r['partition'][:8]}"
            print(f"  {group:<30}  {conf:<20}  "
                  f"{r['final_accuracy']:6.4f}  "
                  f"{r['final_f1_macro']:6.4f}  "
                  f"{r['final_auc_macro']:6.4f}")
    print("="*72)

    # ── Generate extended figures ──────────────────────────────────────────
    try:
        from figures_extended import generate_all_extended_figures
        generate_all_extended_figures(out_path)
        print("[Figures] Extended figures generated.")
    except Exception as e:
        print(f"[Figures] Could not generate extended figures: {e}")


if __name__ == "__main__":
    main()
