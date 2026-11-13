#!/usr/bin/env bash
# Train all 16 seeds (8 n8 + 8 n24), roll out on the held-out forcing, and render
# one COMPLETED 3-panel still (Fortran solver vs GNN, space-time fully filled) per
# seed into figs/aw2_diverged/{n8,n24}/seedN.png. Solver-pinned color scale, so a
# diverged run shows neon-green off-scale cells. No gifs.
set -u
cd "$(dirname "$0")/.."
HOLDOUT=heat_python/data/aw2_holdout_traj.npz
CD=heat_2026-04-11_1837/examples/aw2_tc21
mkdir -p heat_python/models/scratch heat_python/data/scratch

run () {  # $1=tag(n8|n24)  $2=dataset
  local tag=$1 data=$2
  for s in 0 1 2 3 4 5 6 7; do
    local m=heat_python/models/scratch/${tag}_s${s}.pt
    local npz=heat_python/data/scratch/roll_${tag}_s${s}.npz
    python -m heat_python.gnn.train_gnn --data "$data" --flux --no-pred-gas \
      --noise 0.03 --stayput 1.0 --seed "$s" --out "$m" >/dev/null 2>&1
    local err
    err=$(python -m heat_python.gnn.eval_rollout_traj --traj "$HOLDOUT" --model "$m" \
          --save "$npz" 2>/dev/null | grep "interior T:" \
          | sed -E 's/.*mean \|err\| ([0-9.]+) K.*/\1/')
    python -m heat_python.viz_rollout_anim --rollout "$npz" --case-dir "$CD" \
      --with-solver --still \
      --label "aw2 flux-form, ${tag} seed ${s} (held-out mean |dT| ${err} K)" \
      --out "heat_python/figs/aw2_diverged/${tag}/seed${s}.png" >/dev/null 2>&1
    echo "RESULT ${tag} seed=${s} mean_err=${err}"
  done
}

echo "=== n8 (8 forcings) ==="
run n8  heat_python/data/aw2_forcing_dataset.npz
echo "=== n24 (24 forcings) ==="
run n24 heat_python/data/aw2_forcing_dataset_n24.npz
echo "=== DONE ==="
