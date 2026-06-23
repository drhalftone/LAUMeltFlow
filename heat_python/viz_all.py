"""Regenerate ALL rollout visualizations into one folder-per-recipe layout.

Every training recipe gets its own figs/rollouts/<name>/ folder with a space-time
heatmap and profile animation. Each figure is TITLED with the case (aw1 gas-off
vs aw2 gas) and the model version (direct vs conservative flux-form, training
recipe), so a rollout figure is self-identifying on its own.

    python -m heat_python.viz_all
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from .viz_rollout import static_heatmap, profile_animation, _depth_axis

AW1 = "heat_2026-04-11_1837/examples/aw1"
AW2 = "heat_2026-04-11_1837/examples/aw2_tc21"

# (rollout npz, folder name, title label, case dir for the depth axis)
# Label format: "<case>, <model>, <training recipe>" -- no em dashes.
RECIPES = [
    # aw1 (gas-off), direct per-cell-delta model
    ("rollout_aw1.npz",           "01_onestep",            "aw1 gas-off, direct GNN, single-step", AW1),
    ("rollout_aw1_noise.npz",     "02_noise",              "aw1 gas-off, direct GNN, single-step + noise", AW1),
    ("rollout_aw1_ms.npz",        "03_multistep_m4_BEST",  "aw1 gas-off, direct GNN, multi-step M=4 (best direct)", AW1),
    ("rollout_aw1_ms8.npz",       "04_multistep_m8",       "aw1 gas-off, direct GNN, multi-step M=8", AW1),
    ("rollout_aw1_k2.npz",        "05_k2",                 "aw1 gas-off, direct GNN, K=2 stencil", AW1),
    ("rollout_aw1_volume.npz",    "06_volume_v1_diverged", "aw1 gas-off, direct GNN, volume-sampled v1 (diverged)", AW1),
    ("rollout_aw1_volume_v2.npz", "07_volume_v2_diverged", "aw1 gas-off, direct GNN, volume-sampled v2 (diverged)", AW1),
    ("rollout_aw1_volume_v3.npz", "08_volume_v3_diverged", "aw1 gas-off, direct GNN, volume-sampled v3 (diverged)", AW1),
    ("rollout_flux_aw1.npz",      "09_flux_aw1",           "aw1 gas-off, conservative flux-form, single-step", AW1),
    # aw2 (gas + aerothermal)
    ("rollout_aw2_noise.npz",     "10_aw2_noise",          "aw2 gas, direct GNN, single-step + noise", AW2),
    ("rollout_aw2_ms.npz",        "11_aw2_multistep",      "aw2 gas, direct GNN, multi-step", AW2),
    ("rollout_aw2sweep_ms.npz",   "12_aw2_sweep_multistep","aw2 gas forcing-sweep, direct GNN, multi-step", AW2),
    ("rollout_aw2_fine_ms.npz",   "13_aw2_fine_multistep", "aw2 gas fine mesh, direct GNN, multi-step", AW2),
    ("rollout_aw2_coarse_noise.npz","14_aw2_coarse_noise", "aw2 gas coarse mesh, direct GNN, single-step + noise", AW2),
    ("rollout_aw2_k2_ms.npz",     "15_aw2_k2_multistep",   "aw2 gas, direct GNN, K=2 multi-step", AW2),
    ("rollout_aw2_nogas_ms.npz",  "16_aw2_nogas_multistep","aw2 (gas held from truth), direct GNN, multi-step", AW2),
    ("rollout_aw2_adaptive.npz",  "17_aw2_adaptive",       "aw2 gas, direct GNN, adaptive time-step", AW2),
    ("rollout_direct_aw2.npz",    "18_aw2_direct_holdout", "aw2 gas held-out, direct per-cell delta (diverges)", AW2),
    ("rollout_flux_aw2.npz",      "19_flux_aw2",           "aw2 gas held-out, conservative flux-form", AW2),
]


def main():
    repo = Path(__file__).resolve().parents[1]
    data = repo / "heat_python" / "data"
    figs = repo / "heat_python" / "figs"
    for npz, folder, label, case in RECIPES:
        p = data / npz
        if not p.exists():
            print(f"  skip {npz} (missing)")
            continue
        d = dict(np.load(p))
        n = d["T_gt"].shape[1] - 2                    # interior cells from the data
        x_mm = _depth_axis(case, n)
        out = figs / "rollouts" / folder
        out.mkdir(parents=True, exist_ok=True)
        err = np.abs(d["T_pred"][:, 1:-1] - d["T_gt"][:, 1:-1])
        static_heatmap(d["time"], d["T_gt"], d["T_pred"], x_mm,
                       out / "rollout_spacetime.png", label=label)
        profile_animation(d["time"], d["T_gt"], d["T_pred"], x_mm,
                          out / "rollout_profile.gif", label=label)
        print(f"  -> {folder}  (mean |err| {err.mean():.1f} K)  [{label}]")


if __name__ == "__main__":
    main()
