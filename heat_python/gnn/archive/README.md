# Superseded pipeline scripts

These are not maintained and are not on the path to the headline result. They are
kept because several of them **are** results: the negative findings written up in
`docs/heat_TPS/Goodwin_TPS.tex` came from these runs, and deleting the code would
leave the claims unsupported.

Run them as `python -m heat_python.gnn.archive.<module>`.

| File | What it was | Why it is here |
|---|---|---|
| `make_volume_dataset.py` | Volume / state-space sampling instead of trajectory sampling | **Failed.** Rollout diverged (8000 K, then 25000 K). Diagnosed to independent sampling of temperature and char state, which puts cells off the coupled ablator manifold. This is the sampling-methodology result in the paper. |
| `eval_ensemble_traj.py` | Averaging several models' per-step predictions | **Failed.** Diverged to ~840 K. Closed-loop averaging follows a trajectory none of the members would take. |
| `train_adaptive_rollout.py` | Multi-step training with a time-step-aware model | Superseded. Gave 925 K, slightly worse than single-step plus noise. |
| `eval_adaptive.py` | Rollout on an adaptive cadence schedule | Superseded. Bounded the in-sample rollout (784 K) but not the held-out one. |
| `make_aw2_adaptive_dataset.py` | Mixed-cadence dataset with a dt node feature | Superseded along with the rest of the adaptive-cadence line. |
| `make_aw2_dataset.py` | Coarse-cadence aw2 dataset (record every 2000 steps) | Superseded by `../make_aw2_fine.py` (record every 200), which the working models train on. |
| `eval_rollout.py` | aw1-only rollout evaluation | Superseded by the case-agnostic `../eval_rollout_traj.py`. |
| `measure_speedup.py` | Early aw1 wall-time comparison (~4x vs the Python solver) | Superseded. The reported speedup now comes from the n24 aw2 timing runs. |

## The live pipeline

Everything still in `heat_python/gnn/` is current:

```
data_gen.py  ->  make_aw2_fine.py  ->  graph.py  ->  train_gnn.py
                                                  -> train_rollout.py  ->  eval_rollout_traj.py
```

with `gnn_model.py` and `gnn_flux_model.py` holding the networks themselves, and
`make_aw2_holdout_traj.py` building the held-out test case.
