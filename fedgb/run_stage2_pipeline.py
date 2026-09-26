"""
run_stage2_pipeline.py
------------------------
Master orchestrator for finishing "Tier 1" of the Stage-2 (batch-corrected)
validation: waits for the already-running primary+3-replicate training
(session_replicate_train.py) to finish, then runs, in order:

  1. session_extra_seeds.py       -- seeds 7, 123 (3-seed replication)
  2. batch_id_shortcut_diagnostic.py -- direct shortcut-mechanism test
  3. generate_uidatagb_figures.py    -- Stage-2 classification figures/tables
  4. analyze_failure_cases_uidatagb.py -- Stage-2 failure-case analysis
  5. compute_consistency.py           -- Stage-2 cross-replicate consistency
  6. compute_faithfulness.py          -- Stage-2 faithfulness
  7. compute_stability.py             -- Stage-2 stability
  8. sanity_checks.py                 -- Stage-2 sanity check
  9. compute_calibration.py           -- Stage-2 calibration
  10. shortcut_audit.py               -- Stage-2 margin-shortcut audit

All of steps 3-10 run with P0_* environment variables pointed at the
Stage-2 checkpoints/data (outputs/session_corrected/, data/uidatagb_sessioncorrected/)
so that none of the underlying scripts need per-run edits (see the P0_*
overrides added to _p0common.py, compute_consistency.py,
analyze_failure_cases_uidatagb.py, and generate_uidatagb_figures.py).

Each step's stdout/stderr is captured to its own log file AND appended to
run_stage2_pipeline.log; a step failing does not stop the remaining steps
(so partial results are still available), but is recorded clearly.

Run from fedgb/ (intended to be left running unattended for several
hours):  python -u run_stage2_pipeline.py
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.join(HERE, "outputs", "session_corrected")
LOG_PATH = os.path.join(HERE, "run_stage2_pipeline.log")

PRIMARY_REPLICATE_CKPTS = [
    os.path.join(SC, "seed_42", name)
    for name in ("primary.pt", "replicate_1.pt", "replicate_2.pt", "replicate_3.pt")
]

STAGE2_ENV = {
    "P0_CKPT": os.path.join(SC, "seed_42", "primary.pt"),
    "P0_VAL_DIR": os.path.join(HERE, "data", "uidatagb_sessioncorrected", "validation"),
    "P0_DATA_DIR": os.path.join(HERE, "data", "uidatagb_sessioncorrected"),
    "P0_OUT_DIR": os.path.join(SC, "phase0"),
    "P0_FIG_DIR": os.path.join(SC, "figures"),
    "P0_FAILURE_JSON": os.path.join(SC, "phase0", "failure_cases.json"),
    "P0_REPLICATE_CKPTS": os.pathsep.join([
        os.path.join(SC, "seed_42", "replicate_1.pt"),
        os.path.join(SC, "seed_42", "replicate_2.pt"),
        os.path.join(SC, "seed_42", "replicate_3.pt"),
    ]),
}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def wait_for_checkpoints(paths, poll_seconds=60):
    log(f"Waiting for {len(paths)} checkpoint(s) from session_replicate_train.py...")
    while True:
        missing = [p for p in paths if not os.path.exists(p)]
        if not missing:
            log("All expected checkpoints found.")
            return
        log(f"  still waiting on {len(missing)}: {[os.path.basename(m) for m in missing]}")
        time.sleep(poll_seconds)


def run_step(name, cmd, extra_env=None, cwd=HERE):
    log(f"=== START: {name} ===  ({' '.join(cmd)})")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    step_log = os.path.join(HERE, f"stage2_step_{name}.log")
    t0 = time.time()
    with open(step_log, "w") as f:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    log(f"=== END: {name}  {status}  ({dt:.0f}s)  -> {step_log} ===")
    return proc.returncode == 0


def main():
    log("run_stage2_pipeline.py started.")
    py = sys.executable

    # 0. wait for the already-running primary+3-replicate job
    wait_for_checkpoints(PRIMARY_REPLICATE_CKPTS)

    results = {}

    # 1. multi-seed replication (seeds 7, 123; seed 42 already done)
    results["extra_seeds"] = run_step("extra_seeds", [py, "-u", "session_extra_seeds.py"])

    # 2. batch-ID shortcut diagnostic (uses raw data/uidatagb/, not Stage-2 env)
    results["batch_id_diagnostic"] = run_step(
        "batch_id_diagnostic", [py, "-u", "batch_id_shortcut_diagnostic.py"])

    # 3. Stage-2 classification figures/tables
    results["classification_figures"] = run_step(
        "classification_figures", [py, "-u", "generate_uidatagb_figures.py"], STAGE2_ENV)

    # 4. Stage-2 failure-case analysis
    results["failure_cases"] = run_step(
        "failure_cases", [py, "-u", "analyze_failure_cases_uidatagb.py"], STAGE2_ENV)

    # 5-10. Stage-2 quantitative XAI reliability battery
    for script in ["compute_consistency.py", "compute_faithfulness.py",
                   "compute_stability.py", "sanity_checks.py",
                   "compute_calibration.py", "shortcut_audit.py"]:
        key = script.replace(".py", "")
        results[key] = run_step(key, [py, "-u", script], STAGE2_ENV)

    log("=== PIPELINE COMPLETE ===")
    log(f"Step results: {json.dumps(results, indent=2)}")
    with open(os.path.join(HERE, "stage2_pipeline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    failed = [k for k, v in results.items() if not v]
    if failed:
        log(f"WARNING: {len(failed)} step(s) failed: {failed}. "
            f"Check the corresponding stage2_step_<name>.log files.")
    else:
        log("All steps completed successfully.")


if __name__ == "__main__":
    main()
