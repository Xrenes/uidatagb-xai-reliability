"""
run_full_queue.py
------------------
Orchestrates the full remaining Phase 1/2 compute queue as ONE process, so
it survives Claude-session interruptions without needing each step to be
manually re-launched. Steps run strictly sequentially (single GPU):

  1. Wait for the already-running multi-seed sweep (run_seeds.py, seeds
     7/42/123 x {centralized, local_only, fedavg, fedprox}) to finish.
  2. compute_stats.py -> outputs/seeds/aggregate.json (mean+-std, McNemar,
     paired bootstrap).
  3. FedProx mu-ablation: mu in {0.01, 0.05, 0.5} (mu=0.1 already covered
     by the seed=42 fedprox run). --skip_gradcam to keep this fast.
  4. FedBN, Ditto, DP-FedAvg (seed 42, single run each). --skip_gradcam.
  5. Multi-seed XAI cosine-overlap: re-run compute_overlap_all_methods.py
     for seeds 7 and 123 (42 already exists), all 5 methods, against each
     seed's own fedprox_global.pt + local_only_client*.pt.
  6. theta-sensitivity: sweep theta in {0.60,0.65,0.70,0.75,0.80} over the
     existing seed-42 overlap results (no retraining needed).

Writes outputs/full_queue.log (this script's own log) plus each sub-step's
normal output. Safe to re-run: every step checks for its own output file
and skips if already present.

Run detached (see how Claude launched this):
  python run_full_queue.py
"""
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(REPO, "outputs")
SEEDS_DIR = os.path.join(OUT, "seeds")
PY = sys.executable
LOG_PATH = os.path.join(OUT, "full_queue.log")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run(cmd, cwd=REPO):
    log(f"RUN: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    dt = time.time() - t0
    log(f"{'OK' if result.returncode == 0 else 'FAILED'} "
        f"(exit={result.returncode}, {dt/60:.1f} min): {' '.join(cmd)}")
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Step 1: wait for the multi-seed sweep
# ---------------------------------------------------------------------------
def wait_for_sweep():
    expected = [
        os.path.join(SEEDS_DIR, f"seed_{s}", f"{m}.json")
        for s in (7, 42, 123)
        for m in ("centralized", "local_only", "fedavg", "fedprox")
    ]
    log(f"Waiting for {len(expected)} sweep output files...")
    while True:
        missing = [p for p in expected if not os.path.exists(p)]
        if not missing:
            log("Sweep complete.")
            return
        log(f"  still waiting on {len(missing)}/{len(expected)}: "
            f"{os.path.relpath(missing[0], SEEDS_DIR)} ...")
        time.sleep(120)


# ---------------------------------------------------------------------------
# Step 2: aggregate stats
# ---------------------------------------------------------------------------
def step_stats():
    out_path = os.path.join(SEEDS_DIR, "aggregate.json")
    if os.path.exists(out_path):
        log(f"[skip] {out_path} exists")
        return
    run([PY, "compute_stats.py",
         "--seeds_dir", "./outputs/seeds",
         "--pairs", "fedprox:centralized,fedprox:fedavg,fedprox:local_only,"
                     "fedavg:local_only,centralized:local_only",
         "--bootstrap", "2000"])


# ---------------------------------------------------------------------------
# Step 3: FedProx mu ablation
# ---------------------------------------------------------------------------
def step_mu_ablation():
    for mu in (0.01, 0.05, 0.5):
        tag = f"mu{str(mu).replace('.', '')}"
        out_path = os.path.join(OUT, f"results_{tag}.json")
        if os.path.exists(out_path):
            log(f"[skip] {out_path} exists")
            continue
        run([PY, "run_simulation.py",
             "--data_dir", "./data/data", "--rounds", "10",
             "--strategy", "fedprox", "--proximal_mu", str(mu),
             "--results_tag", tag, "--skip_gradcam"])


# ---------------------------------------------------------------------------
# Step 4: FedBN, Ditto, DP-FedAvg
# ---------------------------------------------------------------------------
def step_fl_breadth():
    for strategy, tag, extra in [
        ("fedbn", "fedbn", []),
        ("ditto", "ditto", []),
        ("dp_fedavg", "dpfedavg", ["--dp_noise_multiplier", "1.0",
                                    "--dp_max_grad_norm", "1.0"]),
    ]:
        out_path = os.path.join(OUT, f"results_{tag}.json")
        if os.path.exists(out_path):
            log(f"[skip] {out_path} exists")
            continue
        run([PY, "run_simulation.py",
             "--data_dir", "./data/data", "--rounds", "10",
             "--strategy", strategy, "--results_tag", tag,
             "--skip_gradcam"] + extra)


# ---------------------------------------------------------------------------
# Step 5: multi-seed XAI cosine-overlap
# ---------------------------------------------------------------------------
def step_multiseed_overlap():
    for seed in (7, 123):
        out_path = os.path.join(OUT, f"overlap_seed_{seed}.json")
        if os.path.exists(out_path):
            log(f"[skip] {out_path} exists")
            continue
        seed_dir = os.path.join(SEEDS_DIR, f"seed_{seed}")
        global_ckpt = os.path.join(seed_dir, "fedprox_global.pt")
        if not os.path.exists(global_ckpt):
            log(f"[ERROR] {global_ckpt} missing, skipping seed {seed}")
            continue
        run([PY, "compute_overlap_all_methods.py",
             "--global_ckpt", global_ckpt,
             "--local_dir", seed_dir,
             "--data_dir", "./data/data",
             "--theta", "0.70",
             "--methods", "gradcam,gradcam_pp,saliency,eigencam,scorecam",
             "--device", "cuda",
             "--out", f"./outputs/overlap_seed_{seed}.json"])


# ---------------------------------------------------------------------------
# Step 6: theta sensitivity (exact per-image recount, Grad-CAM++, seed 42 —
# script already exists from an earlier pass; just invoke it correctly)
# ---------------------------------------------------------------------------
def step_theta_sensitivity():
    out_path = os.path.join(OUT, "theta_sensitivity.json")
    if os.path.exists(out_path):
        log(f"[skip] {out_path} exists")
        return
    run([PY, "compute_theta_sensitivity.py",
         "--global_ckpt", "./outputs/checkpoints/round_010.pt",
         "--local_dir", "./outputs/seeds/seed_42"])


def main():
    log("=== full_queue started ===")
    wait_for_sweep()
    step_stats()
    step_mu_ablation()
    step_fl_breadth()
    step_multiseed_overlap()
    step_theta_sensitivity()
    log("=== full_queue DONE ===")


if __name__ == "__main__":
    main()
