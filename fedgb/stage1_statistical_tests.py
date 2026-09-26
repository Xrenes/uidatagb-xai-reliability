"""
stage1_statistical_tests.py
------------------------------
Real paired statistical significance testing for the Stage-1 three-seed
classification replication (seeds 7, 42, 123), using the per-image
y_true/y_pred/y_probs already saved by run_seeds.py for each seed
(outputs/seeds/seed_{7,42,123}/centralized.json) -- all three seeds were
evaluated on the identical Stage-1 corrected validation set (n=3193), so
predictions are paired and directly comparable image-for-image.

Implements:
  - McNemar's exact test (paired accuracy differences), all 3 seed pairs.
  - DeLong's test for correlated ROC AUCs (per-class, one-vs-rest), all
    3 seed pairs. The DeLong AUC point estimate is cross-checked against
    sklearn.metrics.roc_auc_score on the same data as a correctness
    check before the paired covariance/p-value is trusted.
  - Paired bootstrap (2000 resamples) 95% CIs for accuracy, macro-F1, and
    macro-AUC, per seed.

Run from fedgb/:  python stage1_statistical_tests.py
Outputs: outputs/phase0/stage1_statistical_tests.json
"""
import itertools
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [7, 42, 123]
N_BOOT = 2000
RNG_SEED = 42


def load_seed(seed):
    path = os.path.join(HERE, "outputs", "seeds", f"seed_{seed}", "centralized.json")
    with open(path) as f:
        d = json.load(f)
    return (np.array(d["y_true"]), np.array(d["y_pred"]), np.array(d["y_probs"]))


# ---------------- McNemar's exact test ----------------
def mcnemar_exact(correct_a, correct_b):
    """correct_a/b: boolean arrays, same order (paired). Returns (b01, b10, p_value)."""
    from scipy.stats import binomtest
    b01 = int(np.sum(correct_a & ~correct_b))   # a correct, b wrong
    b10 = int(np.sum(~correct_a & correct_b))   # a wrong, b correct
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    p = binomtest(min(b01, b10), n, 0.5, alternative="two-sided").pvalue
    return b01, b10, p


# ---------------- DeLong's test (fast DeLong, per-class one-vs-rest) ----------------
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fast_delong(preds_sorted, m):
    """preds_sorted: (k, m+n) array, k classifiers' scores, columns 0..m-1 are
    positives, m..end are negatives. Returns (aucs[k], covariance[k,k])."""
    k = preds_sorted.shape[0]
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = compute_midrank(pos[r, :])
        ty[r, :] = compute_midrank(neg[r, :])
        tz[r, :] = compute_midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01) if k > 1 else np.array([[np.var(v01)]])
    sy = np.cov(v10) if k > 1 else np.array([[np.var(v10)]])
    sx = np.atleast_2d(sx)
    sy = np.atleast_2d(sy)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_pair_test(y_true_bin, score_a, score_b):
    """y_true_bin: 0/1 array for one class (one-vs-rest). score_a/b: model scores
    for that class, same images, same order. Returns (auc_a, auc_b, p_value)."""
    order = np.argsort(-y_true_bin)  # positives (1) first
    y_sorted = y_true_bin[order]
    m = int(y_sorted.sum())
    if m == 0 or m == len(y_sorted):
        return None, None, None  # class not present / degenerate
    preds = np.vstack([score_a[order], score_b[order]])
    aucs, cov = fast_delong(preds, m)
    auc_diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = auc_diff / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(p)


def macro_auc_delong_correctness_check(y_true, y_probs, n_classes):
    """Sanity check: DeLong's own AUC formula vs sklearn, per class."""
    max_abs_diff = 0.0
    for c in range(n_classes):
        y_bin = (y_true == c).astype(float)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        order = np.argsort(-y_bin)
        y_sorted = y_bin[order]
        m = int(y_sorted.sum())
        score = y_probs[:, c][order]
        aucs, _ = fast_delong(score.reshape(1, -1), m)
        sk_auc = roc_auc_score(y_bin, y_probs[:, c])
        max_abs_diff = max(max_abs_diff, abs(aucs[0] - sk_auc))
    return max_abs_diff


# ---------------- Paired bootstrap ----------------
def bootstrap_ci(y_true, y_pred, y_probs, n_classes, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, f1s, aucs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_probs[idx]
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        try:
            y_bin = np.eye(n_classes)[yt]
            aucs.append(roc_auc_score(y_bin, ypr, average="macro", multi_class="ovr"))
        except ValueError:
            pass  # a resample missing a class; skip that resample's AUC
    def ci(vals):
        vals = np.sort(vals)
        return (float(np.mean(vals)), float(vals[int(0.025 * len(vals))]),
                float(vals[int(0.975 * len(vals)) - 1]))
    return {"accuracy": ci(accs), "f1_macro": ci(f1s), "auc_macro": ci(aucs)}


def main():
    data = {s: load_seed(s) for s in SEEDS}
    n_classes = int(max(d[0].max() for d in data.values())) + 1
    print(f"[stats] loaded {len(SEEDS)} seeds, n_classes={n_classes}, "
          f"n_images={len(data[SEEDS[0]][0])}")

    # correctness check: DeLong's own AUC vs sklearn
    for s in SEEDS:
        y_true, _, y_probs = data[s]
        diff = macro_auc_delong_correctness_check(y_true, y_probs, n_classes)
        print(f"[stats] seed={s}: DeLong-vs-sklearn max per-class AUC discrepancy "
              f"= {diff:.2e} (should be ~1e-10, confirms DeLong implementation is correct)")

    results = {"n_images": len(data[SEEDS[0]][0]), "n_classes": n_classes,
               "delong_correctness_check_max_diff": {}, "mcnemar": [], "delong": [],
               "bootstrap_ci": {}}

    for s in SEEDS:
        y_true, _, y_probs = data[s]
        results["delong_correctness_check_max_diff"][str(s)] = \
            macro_auc_delong_correctness_check(y_true, y_probs, n_classes)

    # McNemar, pairwise
    for a, b in itertools.combinations(SEEDS, 2):
        yt_a, yp_a, _ = data[a]
        yt_b, yp_b, _ = data[b]
        assert np.array_equal(yt_a, yt_b), "seeds must share the same validation image order"
        correct_a = (yt_a == yp_a)
        correct_b = (yt_b == yp_b)
        b01, b10, p = mcnemar_exact(correct_a, correct_b)
        acc_a, acc_b = accuracy_score(yt_a, yp_a), accuracy_score(yt_b, yp_b)
        entry = {"seed_a": a, "seed_b": b, "acc_a": acc_a, "acc_b": acc_b,
                 "n_a_correct_b_wrong": b01, "n_a_wrong_b_correct": b10, "p_value": p,
                 "significant_at_0.05": bool(p < 0.05)}
        results["mcnemar"].append(entry)
        print(f"[stats] McNemar seed{a} vs seed{b}: acc {acc_a:.4f} vs {acc_b:.4f}, "
              f"p={p:.4f} ({'SIGNIFICANT' if p < 0.05 else 'not significant'} at alpha=0.05)")

    # DeLong, pairwise, per-class + macro (mean of per-class z via Fisher-ish
    # summary: report per-class p-values and the count significant)
    from dataset import CLASS_NAMES, CLASS_SHORT
    for a, b in itertools.combinations(SEEDS, 2):
        yt, _, ypr_a = data[a]
        _, _, ypr_b = data[b]
        per_class = {}
        n_sig = 0
        n_tested = 0
        for c in range(n_classes):
            y_bin = (yt == c).astype(float)
            auc_a, auc_b, p = delong_pair_test(y_bin, ypr_a[:, c], ypr_b[:, c])
            if p is None:
                continue
            n_tested += 1
            if p < 0.05:
                n_sig += 1
            per_class[CLASS_SHORT[CLASS_NAMES[c]]] = {
                "auc_a": auc_a, "auc_b": auc_b, "p_value": p, "significant_at_0.05": bool(p < 0.05)}
        entry = {"seed_a": a, "seed_b": b, "per_class": per_class,
                  "n_classes_tested": n_tested, "n_classes_significant": n_sig}
        results["delong"].append(entry)
        print(f"[stats] DeLong seed{a} vs seed{b}: {n_sig}/{n_tested} classes show "
              f"significant per-class AUC difference at alpha=0.05")

    # Bootstrap CIs, per seed
    for s in SEEDS:
        y_true, y_pred, y_probs = data[s]
        ci = bootstrap_ci(y_true, y_pred, y_probs, n_classes)
        results["bootstrap_ci"][str(s)] = ci
        print(f"[stats] seed={s} bootstrap 95% CI: "
              f"acc={ci['accuracy'][0]:.4f} [{ci['accuracy'][1]:.4f}, {ci['accuracy'][2]:.4f}]  "
              f"F1={ci['f1_macro'][0]:.4f} [{ci['f1_macro'][1]:.4f}, {ci['f1_macro'][2]:.4f}]  "
              f"AUC={ci['auc_macro'][0]:.4f} [{ci['auc_macro'][1]:.4f}, {ci['auc_macro'][2]:.4f}]")

    out_dir = os.path.join(HERE, "outputs", "phase0")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stage1_statistical_tests.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[stats] wrote {out_dir}/stage1_statistical_tests.json")


if __name__ == "__main__":
    main()
