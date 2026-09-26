#!/usr/bin/env bash
# run_gpu_queue.sh
# ----------------
# Autonomous post-sweep GPU job queue for the paper-improvement plan.
# Waits for the multi-seed training sweep (run_seeds.py) to finish, then runs
# every remaining GPU/compute job sequentially (GPU is 4GB, so NO concurrency).
#
# Launch AFTER run_seeds.py is already running:
#   nohup bash run_gpu_queue.sh > outputs/gpu_queue.log 2>&1 &
set -u
cd "$(dirname "$0")"
LOG(){ echo "[queue $(date +%H:%M:%S)] $*"; }

SEEDS="7 42 123"

# ---- 1. Wait for the multi-seed sweep to produce all required files --------
need_files=()
for s in $SEEDS; do
  for m in centralized local_only fedavg fedprox; do
    need_files+=("outputs/seeds/seed_${s}/${m}.json")
  done
  need_files+=("outputs/seeds/seed_${s}/fedprox_global.pt")
done

LOG "waiting for multi-seed sweep to complete (${#need_files[@]} files)..."
while true; do
  missing=0
  for f in "${need_files[@]}"; do [ -f "$f" ] || missing=$((missing+1)); done
  if [ "$missing" -eq 0 ]; then LOG "all sweep files present."; break; fi
  sleep 60
done

# ---- 2. Multi-seed aggregate statistics ------------------------------------
LOG "running compute_stats.py ..."
python compute_stats.py \
  --seeds_dir ./outputs/seeds \
  --pairs fedprox:centralized,fedprox:fedavg,fedprox:local_only,fedavg:local_only,centralized:local_only \
  --bootstrap 2000 && LOG "compute_stats done" || LOG "compute_stats FAILED"

# ---- 3. FedProx mu sensitivity ablation (seed-0 internal, tagged) ----------
for mu in 0.01 0.05 0.10 0.50; do
  tag="mu_${mu/./p}"
  out="outputs/results_${tag}.json"
  if [ -f "$out" ]; then LOG "skip mu=$mu (exists)"; continue; fi
  LOG "mu ablation: fedprox mu=$mu -> $tag"
  python run_simulation.py --strategy fedprox --proximal_mu "$mu" \
    --rounds 10 --local_epochs 3 --skip_gradcam --results_tag "$tag" \
    && LOG "mu=$mu done" || LOG "mu=$mu FAILED"
done

# ---- 4. Additional FL strategies: FedBN, Ditto, DP-FedAvg ------------------
run_strategy(){ # $1=strategy $2=tag $3..=extra args
  local strat="$1" tag="$2"; shift 2
  local out="outputs/results_${tag}.json"
  if [ -f "$out" ]; then LOG "skip $tag (exists)"; return; fi
  LOG "strategy $strat -> $tag  extra:[$*]"
  python run_simulation.py --strategy "$strat" --rounds 10 --local_epochs 3 \
    --skip_gradcam --results_tag "$tag" "$@" \
    && LOG "$tag done" || LOG "$tag FAILED"
}
run_strategy fedbn     fedbn
run_strategy ditto     ditto     --ditto_lambda 0.1 --ditto_personal_epochs 2
run_strategy dp_fedavg dp_fedavg --dp_noise_multiplier 1.0 --dp_max_grad_norm 1.0

# ---- 5. Multi-seed XAI cosine-overlap (5 methods per seed) ------------------
for s in $SEEDS; do
  gck="outputs/seeds/seed_${s}/fedprox_global.pt"
  out="outputs/overlap_seed_${s}.json"
  if [ -f "$out" ]; then LOG "skip overlap seed $s (exists)"; continue; fi
  if [ ! -f "$gck" ]; then LOG "overlap seed $s: missing $gck, SKIP"; continue; fi
  LOG "XAI overlap seed $s (5 methods) ..."
  python compute_overlap_all_methods.py \
    --global_ckpt "$gck" \
    --local_dir "outputs/seeds/seed_${s}" \
    --methods gradcam,gradcam_pp,saliency,eigencam,scorecam \
    --device cuda --out "$out" \
    && LOG "overlap seed $s done" || LOG "overlap seed $s FAILED"
done

# ---- 6. Theta sensitivity (raw per-image Grad-CAM++ overlaps) ---------------
LOG "theta sensitivity ..."
python compute_theta_sensitivity.py \
  --global_ckpt ./outputs/checkpoints/round_010.pt \
  --local_dir ./outputs/seeds/seed_42 \
  && LOG "theta sensitivity done" || LOG "theta sensitivity FAILED"

LOG "ALL GPU QUEUE PHASES COMPLETE."
