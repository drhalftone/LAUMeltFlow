#!/usr/bin/env bash
# Controlled sweep on n24 (the reliable dataset) targeting the surface-undershoot
# bias. 3 recipes x 3 seeds. A vs B isolates loss-weighting; B vs C isolates
# capacity. Reports overall mean, hot-band mean, and signed hot-band bias.
set -u
cd "$(dirname "$0")/.."
SEEDS="0 1 2"
DATA=heat_python/data/aw2_forcing_dataset_n24.npz
HOLDOUT=heat_python/data/aw2_holdout_traj.npz
mkdir -p heat_python/models/scratch
BASE="--data $DATA --flux --no-pred-gas --noise 0.03 --stayput 1.0"

run () {  # $1=tag  $2=extra-train-args
  local tag=$1 extra=$2
  for s in $SEEDS; do
    local m=heat_python/models/scratch/bias_${tag}_s${s}.pt
    python -m heat_python.train_gnn $BASE $extra --seed "$s" --out "$m" \
      >/dev/null 2>&1
    local out
    out=$(python -m heat_python.eval_rollout_traj --traj "$HOLDOUT" --model "$m" 2>/dev/null)
    local mean hot bias
    mean=$(echo "$out" | grep "interior T:" | sed -E 's/.*mean \|err\| ([0-9.]+) K.*/\1/')
    hot=$(echo "$out"  | grep "hot band"   | sed -E 's/.*mean \|err\| ([0-9.]+) K.*/\1/')
    bias=$(echo "$out" | grep "hot band"   | sed -E 's/.*signed bias ([+-][0-9.]+) K.*/\1/')
    echo "RESULT ${tag} seed=${s} mean=${mean} hot=${hot} bias=${bias}"
  done
}

echo "=== A: baseline (lambda=0, h64) ==="
run A ""
echo "=== B: loss-weight (lambda=8, h64) ==="
run B "--loss-weight 8"
echo "=== C: loss-weight + capacity (lambda=8, h128) ==="
run C "--loss-weight 8 --hidden 128"
echo "=== DONE ==="
