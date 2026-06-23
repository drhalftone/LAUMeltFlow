#!/usr/bin/env bash
# Seeded held-out spread for single-step flux: 8 seeds x {8-traj, n24}.
# Reports mean interior |err| per run so we can quote a distribution, not anecdote.
set -u
cd "$(dirname "$0")/.."
SEEDS="0 1 2 3 4 5 6 7"
HOLDOUT=heat_python/data/aw2_holdout_traj.npz
mkdir -p heat_python/models/scratch

run_one () {  # $1=data  $2=tag
  local data=$1 tag=$2
  for s in $SEEDS; do
    local m=heat_python/models/scratch/${tag}_s${s}.pt
    python -m heat_python.train_gnn --data "$data" --flux --no-pred-gas \
      --noise 0.03 --stayput 1.0 --seed "$s" --out "$m" >/dev/null 2>&1
    local err
    err=$(python -m heat_python.eval_rollout_traj --traj "$HOLDOUT" --model "$m" 2>/dev/null \
          | grep "mean |err|" | sed -E 's/.*mean \|err\| ([0-9.]+) K.*/\1/')
    echo "RESULT ${tag} seed=${s} mean_err=${err}"
  done
}

echo "=== 8-traj (aw2_forcing_dataset.npz) ==="
run_one heat_python/data/aw2_forcing_dataset.npz n8
echo "=== n24 (aw2_forcing_dataset_n24.npz) ==="
run_one heat_python/data/aw2_forcing_dataset_n24.npz n24
echo "=== DONE ==="
