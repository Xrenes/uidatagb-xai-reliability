"""
compute_stats.py
----------------
Aggregates the per-seed JSON files written by run_seeds.py and produces:

  - Mean +/- standard deviation for accuracy, macro-F1, macro-AUC per method.
  - Paired McNemar tests on per-image correctness (accuracy comparison).
  - Paired bootstrap 95 % CIs on macro-F1 differences.

Inputs:
  outputs/seeds/seed_<n>/<method>.json

Output:
  outputs/seeds/aggregate.json   -- summary numbers consumed by the paper.

Usage (run from fedgb/):
  python compute_stats.py \
         --seeds_dir ./outputs/seeds \
         --pairs fedprox:centralized,fedprox:fedavg,fedprox:local_only

The aggregate.json is structured so the HTML paper Table IV (mean +/- std)
and Table IX (paired statistics) can be populated mechanically.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np

CLASS_NAMES = ["nml", "bmt", "stn", "abn", "malg"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_seed_runs(seeds_dir: str) -> Dict[str, List[dict]]:
    """Return {method: [per-seed-result, ...]} sorted by seed value."""
    runs: Dict[str, List[Tuple[int, dict]]] = {}
    for seed_dir in sorted(glob.glob(os.path.join(seeds_dir, "seed_*"))):
        seed = int(os.path.basename(seed_dir).split("_", 1)[1])
        for jp in glob.glob(os.path.join(seed_dir, "*.json")):
            method = os.path.splitext(os.path.basename(jp))[0]
            if method.startswith("local_only_client"):
                continue
            with open(jp) as f:
                data = json.load(f)
            runs.setdefault(method, []).append((seed, data))
    return {m: [d for _, d in sorted(lst)] for m, lst in runs.items()}


# ---------------------------------------------------------------------------
# Mean +/- std summary
# ---------------------------------------------------------------------------
def _scalar_for(data: dict, key: str) -> float:
    """Local-only stores its averaged metrics under 'averaged'."""
    if "averaged" in data and key in data["averaged"]:
        return float(data["averaged"][key])
    return float(data[key])


def _per_class_for(data: dict, key: str) -> List[float]:
    if "averaged" in data and key in data["averaged"]:
        return [float(v) for v in data["averaged"][key]]
    return [float(v) for v in data[key]]


def summarise(runs: Dict[str, List[dict]]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}
    for method, results in runs.items():
        accs = np.array([_scalar_for(r, "accuracy")  for r in results])
        f1s  = np.array([_scalar_for(r, "f1_macro")  for r in results])
        aucs = np.array([_scalar_for(r, "auc_macro") for r in results])
        f1_pc  = np.array([_per_class_for(r, "f1_per_class")  for r in results])
        auc_pc = np.array([_per_class_for(r, "auc_per_class") for r in results])
        summary[method] = {
            "n_seeds": int(len(results)),
            "seeds":   [int(r.get("meta", {}).get("seed", -1)) for r in results],
            "accuracy":  {"mean": float(accs.mean()), "std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0},
            "f1_macro":  {"mean": float(f1s.mean()),  "std": float(f1s.std(ddof=1))  if len(f1s)  > 1 else 0.0},
            "auc_macro": {"mean": float(aucs.mean()), "std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0},
            "f1_per_class": {
                "mean": [float(x) for x in f1_pc.mean(axis=0)],
                "std":  [float(x) for x in (f1_pc.std(axis=0, ddof=1) if len(f1_pc) > 1 else np.zeros(f1_pc.shape[1]))],
            },
            "auc_per_class": {
                "mean": [float(x) for x in auc_pc.mean(axis=0)],
                "std":  [float(x) for x in (auc_pc.std(axis=0, ddof=1) if len(auc_pc) > 1 else np.zeros(auc_pc.shape[1]))],
            },
        }
    return summary


# ---------------------------------------------------------------------------
# Paired McNemar (accuracy) on shared test set, averaged across seeds
# ---------------------------------------------------------------------------
def _mcnemar_b_c(y_true: np.ndarray,
                 y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray) -> Tuple[int, int]:
    """Return (b, c) where b=A correct & B wrong, c=A wrong & B correct."""
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    return b, c


def _mcnemar_pvalue(b: int, c: int) -> float:
    """Exact mid-p McNemar test using the binomial distribution."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_two_sided = 0.0
    for i in range(0, k + 1):
        p_two_sided += comb(n, i) * (0.5 ** n)
    p_two_sided = min(1.0, 2.0 * p_two_sided)
    return float(p_two_sided)


def paired_mcnemar(runs: Dict[str, List[dict]],
                   method_a: str, method_b: str) -> dict:
    if method_a not in runs or method_b not in runs:
        return {"error": f"missing method ({method_a} or {method_b})"}
    pairs = []
    # Align by seed
    by_seed_a = {r["meta"]["seed"]: r for r in runs[method_a]}
    by_seed_b = {r["meta"]["seed"]: r for r in runs[method_b]}
    common_seeds = sorted(set(by_seed_a) & set(by_seed_b))
    for s in common_seeds:
        ra, rb = by_seed_a[s], by_seed_b[s]
        y_true   = np.array(ra["y_true"])
        y_true_b = np.array(rb["y_true"])
        if not np.array_equal(y_true, y_true_b):
            # Should not happen if test loader order is deterministic,
            # but fall back to length-aligned comparison.
            n = min(len(y_true), len(y_true_b))
            y_true   = y_true[:n]
            y_pred_a = np.array(ra["y_pred"])[:n]
            y_pred_b = np.array(rb["y_pred"])[:n]
        else:
            y_pred_a = np.array(ra["y_pred"])
            y_pred_b = np.array(rb["y_pred"])
        b, c = _mcnemar_b_c(y_true, y_pred_a, y_pred_b)
        pairs.append({
            "seed": int(s),
            "b_AonlyCorrect": b,
            "c_BonlyCorrect": c,
            "p_mcnemar":      _mcnemar_pvalue(b, c),
        })
    return {
        "method_a":     method_a,
        "method_b":     method_b,
        "common_seeds": [int(s) for s in common_seeds],
        "per_seed":     pairs,
        "median_pvalue": float(np.median([p["p_mcnemar"] for p in pairs]))
                         if pairs else None,
    }


# ---------------------------------------------------------------------------
# Paired bootstrap for macro-F1 difference (per seed, then averaged)
# ---------------------------------------------------------------------------
def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall    = tp / (tp + fn)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1s))


def paired_bootstrap_f1(runs: Dict[str, List[dict]],
                        method_a: str, method_b: str,
                        n_resamples: int = 2000,
                        rng_seed: int = 0) -> dict:
    if method_a not in runs or method_b not in runs:
        return {"error": "missing method"}
    rng = np.random.default_rng(rng_seed)
    by_seed_a = {r["meta"]["seed"]: r for r in runs[method_a]}
    by_seed_b = {r["meta"]["seed"]: r for r in runs[method_b]}
    common_seeds = sorted(set(by_seed_a) & set(by_seed_b))
    per_seed = []
    n_classes = len(CLASS_NAMES)
    for s in common_seeds:
        ra, rb = by_seed_a[s], by_seed_b[s]
        y_true   = np.array(ra["y_true"])
        y_pred_a = np.array(ra["y_pred"])
        y_pred_b = np.array(rb["y_pred"])
        n = len(y_true)
        diffs = np.empty(n_resamples, dtype=np.float64)
        for k in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            f1a = _macro_f1(y_true[idx], y_pred_a[idx], n_classes)
            f1b = _macro_f1(y_true[idx], y_pred_b[idx], n_classes)
            diffs[k] = f1a - f1b
        per_seed.append({
            "seed":           int(s),
            "diff_mean":      float(diffs.mean()),
            "diff_ci95_low":  float(np.percentile(diffs,  2.5)),
            "diff_ci95_high": float(np.percentile(diffs, 97.5)),
            "p_two_sided":    float(2 * min((diffs <= 0).mean(),
                                            (diffs >= 0).mean())),
        })
    return {
        "method_a":      method_a,
        "method_b":      method_b,
        "common_seeds":  [int(s) for s in common_seeds],
        "n_resamples":   int(n_resamples),
        "per_seed":      per_seed,
        "median_diff":   float(np.median([p["diff_mean"] for p in per_seed]))
                          if per_seed else None,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds_dir", type=str,
                        default="./outputs/seeds")
    parser.add_argument("--pairs", type=str,
                        default="fedprox:centralized,fedprox:fedavg,"
                                "fedprox:local_only,fedavg:local_only,"
                                "centralized:local_only",
                        help="Comma-separated A:B pairs for paired testing.")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Bootstrap resamples per seed (set 0 to skip).")
    parser.add_argument("--out", type=str,
                        default="./outputs/seeds/aggregate.json")
    args = parser.parse_args()

    runs = load_seed_runs(args.seeds_dir)
    if not runs:
        print(f"[error] no seed runs found under {args.seeds_dir}")
        return
    summary = summarise(runs)
    print("\n=== Mean +/- std (per method) ===")
    for m, s in summary.items():
        print(f"  {m:12s} n={s['n_seeds']}  "
              f"acc={s['accuracy']['mean']:.4f}+/-{s['accuracy']['std']:.4f}  "
              f"f1={s['f1_macro']['mean']:.4f}+/-{s['f1_macro']['std']:.4f}  "
              f"auc={s['auc_macro']['mean']:.4f}+/-{s['auc_macro']['std']:.4f}")

    mcnemar_results = {}
    bootstrap_results = {}
    for pair in args.pairs.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        a, b = [x.strip() for x in pair.split(":", 1)]
        mcnemar_results[f"{a}_vs_{b}"] = paired_mcnemar(runs, a, b)
        if args.bootstrap > 0:
            bootstrap_results[f"{a}_vs_{b}"] = paired_bootstrap_f1(
                runs, a, b, n_resamples=args.bootstrap)

    out = {
        "summary":   summary,
        "mcnemar":   mcnemar_results,
        "bootstrap": bootstrap_results,
        "class_names": CLASS_NAMES,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
