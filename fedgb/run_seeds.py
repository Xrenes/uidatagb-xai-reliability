"""
run_seeds.py
------------
Multi-seed driver for the four primary baselines reported in the FedGB paper:

  - centralized   : single ResNet-50 trained on the union of all client data
  - local_only    : one ResNet-50 per simulated hospital, trained on that
                    hospital's data only and then evaluated on the global
                    held-out test set (per-client and macro-averaged metrics)
  - fedavg        : standard Federated Averaging (10 rounds, 3 clients)
  - fedprox       : FedAvg with proximal term mu=0.1

For each (seed, method) combination this script writes a JSON file containing:
  - final scalar metrics (accuracy, macro-F1, macro-AUC, per-class metrics)
  - per-image predicted class and probability vector on the global test set
    (required for paired McNemar and paired-bootstrap statistical tests)
  - convergence curves
  - the local-only baseline additionally saves per-client model checkpoints
    in outputs/seeds/seed_<n>/local_only_client<k>.pt so the Grad-CAM
    cosine-overlap script can reload them later.

Outputs:
  outputs/seeds/seed_<n>/<method>.json
  outputs/seeds/seed_<n>/local_only_client<k>.pt   (local_only only)
  outputs/seeds/aggregate.json   (written by compute_stats.py)

Usage (run from fedgb/):
  python run_seeds.py --seeds 7,42,123 \
                      --methods centralized,local_only,fedavg,fedprox \
                      --rounds 10 --local_epochs 3

  # Faster smoke test:
  python run_seeds.py --seeds 42 --methods centralized --rounds 2 --local_epochs 1

This script is CPU-safe. On a typical workstation CPU each full method-seed
combination takes roughly 2-4 hours; the full 4-method x 3-seed sweep is
therefore best launched in a background terminal.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (
    CLASS_NAMES,
    NUM_CLASSES,
    GBCUDataset,
    get_client_datasets,
    get_transforms,
)
from model import build_model, get_model_parameters, set_model_parameters
from evaluate import compute_metrics, run_inference
from run_simulation import (
    eval_model,
    fedavg_aggregate,
    train_client,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", "seeds")
os.makedirs(OUT_ROOT, exist_ok=True)

# Fixed 224x224 input shape every step -> let cuDNN autotune the fastest
# conv algorithms instead of re-searching (or under-using) the GPU each call.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Tried num_workers=2/4 (separate spawned processes for JPEG decode/resize):
# on Windows, DataLoader workers use spawn + real inter-process tensor
# transfer (no fork/shared memory), and for this dataset (1605 images,
# batch_size 16-ish) that IPC overhead measured SLOWER than in-process
# loading (epoch time went 65-70s -> ~119s). So loading stays in the main
# process; pin_memory + non_blocking transfers + cudnn.benchmark below are
# the actual wins here.
NUM_WORKERS = 0
LOADER_KWARGS = dict(num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())


# ----------------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def build_loaders(
    data_dir: str,
    batch_size: int,
    seed: int,
    num_clients: int = 3,
):
    """Returns (per_client_train_loaders, global_test_loader, sizes,
                client_datasets, test_dataset)."""
    # "label_skew" hardcodes 5-class hospital weight rows (GBCU-specific) and
    # would IndexError on any other class count. It only matters for
    # federated methods anyway -- the "centralized" method immediately
    # re-pools all client partitions via ConcatDataset, so which partition
    # scheme is used has no effect on that method's result. "dirichlet"
    # supports any number of classes.
    client_datasets, test_dataset = get_client_datasets(
        data_dir, num_clients=num_clients, partition="dirichlet",
        seed=seed,
    )
    client_loaders = []
    sizes = []
    for cdata in client_datasets:
        sizes.append(len(cdata))
        client_loaders.append(
            DataLoader(cdata, batch_size=batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(seed),
                       **LOADER_KWARGS)
        )
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, **LOADER_KWARGS)
    return client_loaders, test_loader, sizes, client_datasets, test_dataset


# ----------------------------------------------------------------------------
# Method 1: Centralised training (pool all client data)
# ----------------------------------------------------------------------------
def run_centralized(
    client_datasets, test_loader, *,
    seed: int, rounds: int, local_epochs: int, batch_size: int,
    checkpoint_path: str | None = None,
) -> Dict:
    set_global_seed(seed)
    pooled = ConcatDataset(client_datasets)
    loader = DataLoader(pooled, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        **LOADER_KWARGS)
    model = build_model(pretrained=True).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    # Centralised matches FL wall-clock budget: rounds * local_epochs epochs.
    total_epochs = rounds * local_epochs

    # Resume support: this process has been killed mid-run by session/host
    # interruptions more than once, always losing all progress because the
    # only save happened after the full epoch loop. Persist a per-epoch
    # checkpoint (model + optimizer + curves so far) and resume from it if
    # present, instead of restarting from epoch 1 every time.
    resume_path = (checkpoint_path + ".inprogress") if checkpoint_path else None
    start_ep = 1
    acc_curve, loss_curve = [], []
    if resume_path is not None and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        acc_curve = ckpt["acc_curve"]
        loss_curve = ckpt["loss_curve"]
        start_ep = ckpt["epoch"] + 1
        print(f"  [centralized seed={seed}] resuming from epoch {start_ep}/{total_epochs} "
              f"(found {resume_path})")

    for ep in range(start_ep, total_epochs + 1):
        train_client(model, loader, optim, crit,
                     epochs=1, proximal_mu=0.0,
                     desc=f"  [centralized seed={seed}] epoch {ep}/{total_epochs}")
        vl, va = eval_model(model, test_loader, crit)
        acc_curve.append(round(float(va), 4))
        loss_curve.append(round(float(vl), 4))
        if resume_path is not None:
            os.makedirs(os.path.dirname(resume_path), exist_ok=True)
            torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                       "epoch": ep, "acc_curve": acc_curve, "loss_curve": loss_curve},
                      resume_path)
    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        if resume_path is not None and os.path.exists(resume_path):
            os.remove(resume_path)  # done -- drop the in-progress shard
    return _final_eval(model, test_loader,
                       extra={"accuracy_curve": acc_curve,
                              "loss_curve":     loss_curve})


# ----------------------------------------------------------------------------
# Method 2: Local-only baseline (one model per hospital, eval on global test)
# ----------------------------------------------------------------------------
def run_local_only(
    client_datasets, test_loader, *,
    seed: int, rounds: int, local_epochs: int, batch_size: int,
    checkpoint_dir: str | None = None,
) -> Dict:
    set_global_seed(seed)
    total_epochs = rounds * local_epochs
    crit = nn.CrossEntropyLoss()
    per_client_metrics = []
    per_client_predictions = []
    per_client_probabilities = []
    for cid, cdata in enumerate(client_datasets):
        loader = DataLoader(cdata, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed + cid),
                            **LOADER_KWARGS)
        model = build_model(pretrained=True).to(DEVICE)
        optim = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

        # Resume support (same reasoning as run_centralized): if this client
        # was already fully trained in a prior, interrupted run, its final
        # checkpoint exists -- load it and skip straight to (cheap)
        # inference rather than retraining. If it was only partially
        # trained, resume from its last saved epoch instead of scratch.
        final_ckpt = (os.path.join(checkpoint_dir, f"local_only_client{cid}.pt")
                     if checkpoint_dir else None)
        resume_ckpt = (final_ckpt + ".inprogress") if final_ckpt else None
        start_ep = 1
        if final_ckpt is not None and os.path.exists(final_ckpt):
            model.load_state_dict(torch.load(final_ckpt, map_location=DEVICE))
            start_ep = total_epochs + 1  # already fully trained -- skip loop
            print(f"  [local_only seed={seed} client={cid}] already trained "
                  f"(found {final_ckpt}), skipping to inference")
        elif resume_ckpt is not None and os.path.exists(resume_ckpt):
            ck = torch.load(resume_ckpt, map_location=DEVICE)
            model.load_state_dict(ck["model"])
            optim.load_state_dict(ck["optim"])
            start_ep = ck["epoch"] + 1
            print(f"  [local_only seed={seed} client={cid}] resuming from "
                  f"epoch {start_ep}/{total_epochs} (found {resume_ckpt})")

        for ep in range(start_ep, total_epochs + 1):
            train_client(model, loader, optim, crit,
                         epochs=1, proximal_mu=0.0,
                         desc=f"  [local_only seed={seed} client={cid}] "
                              f"epoch {ep}/{total_epochs}")
            if resume_ckpt is not None:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                           "epoch": ep}, resume_ckpt)
        y_true, y_pred, y_probs = run_inference(model, test_loader)
        m = compute_metrics(y_true, y_pred, y_probs)
        per_client_metrics.append({
            "client_id": cid,
            "accuracy":  float(m["accuracy"]),
            "f1_macro":  float(m["f1_macro"]),
            "auc_macro": float(m["auc_macro"]),
            "f1_per_class":  [float(v) for v in m["f1_per_class"]],
            "auc_per_class": [float(v) for v in m["auc_per_class"]],
            "sensitivity":   [float(v) for v in m["sensitivity"]],
            "specificity":   [float(v) for v in m["specificity"]],
        })
        per_client_predictions.append([int(v) for v in y_pred.tolist()])
        per_client_probabilities.append([[float(p) for p in row]
                                         for row in y_probs.tolist()])
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), final_ckpt)
            if resume_ckpt is not None and os.path.exists(resume_ckpt):
                os.remove(resume_ckpt)  # done -- drop the in-progress shard

    # Macro-aggregate per-image: average softmax over the 3 client models,
    # then argmax. This is the "averaged-local" reference reported in the paper.
    probs_stack = np.mean(np.array(per_client_probabilities), axis=0)
    y_true = _ground_truth(test_loader)
    y_pred_avg = probs_stack.argmax(axis=1)
    m_avg = compute_metrics(y_true, y_pred_avg, probs_stack)
    return {
        "per_client":     per_client_metrics,
        "averaged":       {
            "accuracy":  float(m_avg["accuracy"]),
            "f1_macro":  float(m_avg["f1_macro"]),
            "auc_macro": float(m_avg["auc_macro"]),
            "f1_per_class":  [float(v) for v in m_avg["f1_per_class"]],
            "auc_per_class": [float(v) for v in m_avg["auc_per_class"]],
            "sensitivity":   [float(v) for v in m_avg["sensitivity"]],
            "specificity":   [float(v) for v in m_avg["specificity"]],
        },
        "y_true":         [int(v) for v in y_true.tolist()],
        "y_pred":         [int(v) for v in y_pred_avg.tolist()],
        "y_probs":        probs_stack.tolist(),
        "per_client_y_pred":  per_client_predictions,
        "per_client_y_probs": per_client_probabilities,
    }


# ----------------------------------------------------------------------------
# Method 3 / 4: FedAvg and FedProx
# ----------------------------------------------------------------------------
def run_federated(
    client_datasets, test_loader, *,
    seed: int, rounds: int, local_epochs: int, batch_size: int,
    strategy: str,           # "fedavg" or "fedprox"
    proximal_mu: float = 0.1,
    checkpoint_path: str | None = None,
) -> Dict:
    set_global_seed(seed)
    client_loaders = [
        DataLoader(cdata, batch_size=batch_size, shuffle=True,
                   generator=torch.Generator().manual_seed(seed + cid),
                   **LOADER_KWARGS)
        for cid, cdata in enumerate(client_datasets)
    ]
    global_model = build_model(pretrained=True).to(DEVICE)
    crit = nn.CrossEntropyLoss()
    acc_curve, loss_curve = [], []
    for rnd in range(1, rounds + 1):
        snapshot = [p.detach().clone() for p in global_model.parameters()]
        results = []
        for cid, ldr in enumerate(client_loaders):
            local = copy.deepcopy(global_model)
            optim = torch.optim.Adam(local.parameters(),
                                     lr=1e-4, weight_decay=1e-4)
            train_client(local, ldr, optim, crit,
                         epochs=local_epochs,
                         proximal_mu=(proximal_mu if strategy == "fedprox"
                                      else 0.0),
                         global_params=snapshot,
                         desc=f"  [{strategy} seed={seed}] R{rnd} C{cid}")
            results.append((get_model_parameters(local), len(ldr.dataset)))
        set_model_parameters(global_model, fedavg_aggregate(results))
        vl, va = eval_model(global_model, test_loader, crit)
        acc_curve.append(round(float(va), 4))
        loss_curve.append(round(float(vl), 4))
    # Persist the final global model so the multi-seed XAI cosine-overlap
    # analysis can reload the per-seed global model later.
    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(global_model.state_dict(), checkpoint_path)
    return _final_eval(global_model, test_loader,
                       extra={"accuracy_curve": acc_curve,
                              "loss_curve":     loss_curve,
                              "proximal_mu":    proximal_mu
                                                if strategy == "fedprox" else 0.0})


# ----------------------------------------------------------------------------
# Common evaluation helpers
# ----------------------------------------------------------------------------
def _ground_truth(loader: DataLoader) -> np.ndarray:
    ys = []
    for _, y in loader:
        ys.extend(y.numpy().tolist())
    return np.array(ys)


def _final_eval(model, test_loader, extra: Dict | None = None) -> Dict:
    y_true, y_pred, y_probs = run_inference(model, test_loader)
    m = compute_metrics(y_true, y_pred, y_probs)
    out = {
        "accuracy":      float(m["accuracy"]),
        "f1_macro":      float(m["f1_macro"]),
        "auc_macro":     float(m["auc_macro"]),
        "f1_per_class":  [float(v) for v in m["f1_per_class"]],
        "auc_per_class": [float(v) for v in m["auc_per_class"]],
        "sensitivity":   [float(v) for v in m["sensitivity"]],
        "specificity":   [float(v) for v in m["specificity"]],
        "y_true":        [int(v) for v in y_true.tolist()],
        "y_pred":        [int(v) for v in y_pred.tolist()],
        "y_probs":       [[float(p) for p in row] for row in y_probs.tolist()],
    }
    if extra:
        out.update(extra)
    return out


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str, default="./data/data")
    parser.add_argument("--seeds",       type=str, default="7,42,123",
                        help="Comma-separated list of integer seeds.")
    parser.add_argument("--methods",     type=str,
                        default="centralized,local_only,fedavg,fedprox",
                        help="Comma-separated subset of "
                             "{centralized, local_only, fedavg, fedprox}.")
    parser.add_argument("--rounds",      type=int, default=10)
    parser.add_argument("--local_epochs",type=int, default=3)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--proximal_mu", type=float, default=0.1)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"\n{'='*60}")
    print(f"  FedGB multi-seed driver")
    print(f"  Device  : {DEVICE}")
    print(f"  Seeds   : {seeds}")
    print(f"  Methods : {methods}")
    print(f"  Rounds  : {args.rounds}  (local_epochs={args.local_epochs})")
    print(f"{'='*60}\n")

    overall_t0 = time.time()
    for seed in seeds:
        seed_dir = os.path.join(OUT_ROOT, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        # Build loaders once per seed (same partition for fairness across methods)
        _, test_loader, sizes, client_datasets, _ = build_loaders(
            args.data_dir, args.batch_size, seed=seed)
        print(f"\n[seed {seed}] client sizes = {sizes}, "
              f"|test| = {len(test_loader.dataset)}")
        for method in methods:
            out_path = os.path.join(seed_dir, f"{method}.json")
            if os.path.exists(out_path):
                print(f"  [skip] {out_path} already exists.")
                continue
            t0 = time.time()
            print(f"\n  >>> {method.upper()} (seed={seed}) starting...")
            if method == "centralized":
                ckpt_path = os.path.join(seed_dir, "centralized.pt")
                result = run_centralized(
                    client_datasets, test_loader,
                    seed=seed, rounds=args.rounds,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    checkpoint_path=ckpt_path)
            elif method == "local_only":
                result = run_local_only(
                    client_datasets, test_loader,
                    seed=seed, rounds=args.rounds,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    checkpoint_dir=seed_dir)
            elif method in ("fedavg", "fedprox"):
                ckpt_path = os.path.join(seed_dir, f"{method}_global.pt")
                result = run_federated(
                    client_datasets, test_loader,
                    seed=seed, rounds=args.rounds,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    strategy=method,
                    proximal_mu=args.proximal_mu,
                    checkpoint_path=ckpt_path)
            else:
                raise ValueError(f"Unknown method: {method}")
            result["meta"] = {
                "seed":         seed,
                "method":       method,
                "rounds":       args.rounds,
                "local_epochs": args.local_epochs,
                "batch_size":   args.batch_size,
                "device":       str(DEVICE),
                "wall_seconds": round(time.time() - t0, 1),
                "client_sizes": sizes,
                "class_names":  CLASS_NAMES,
            }
            with open(out_path, "w") as f:
                json.dump(result, f)
            print(f"  <<< wrote {out_path} "
                  f"({result['meta']['wall_seconds']}s)")
    print(f"\n[done] total wall-time {round(time.time()-overall_t0,1)}s")


if __name__ == "__main__":
    main()
