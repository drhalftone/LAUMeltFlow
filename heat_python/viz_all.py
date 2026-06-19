"""Regenerate ALL rollout visualizations into one folder-per-recipe layout.

Every training recipe gets its own figs/rollout_NN_<name>/ folder with its
space-time heatmap and profile animation, so the inconsistent earlier layout
(root = one-step, noise/, multistep/) is replaced by a uniform structure.
Divergent recipes (volume v1-v3) are included; their blow-up is visible.

    python -m heat_python.viz_all
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

from .viz_rollout import static_heatmap, profile_animation, _depth_axis

# (rollout npz, folder name)  -- ordered by the project narrative
RECIPES = [
    ("rollout_aw1.npz",           "rollout_01_onestep"),
    ("rollout_aw1_noise.npz",     "rollout_02_noise"),
    ("rollout_aw1_ms.npz",        "rollout_03_multistep_m4_BEST"),
    ("rollout_aw1_ms8.npz",       "rollout_04_multistep_m8"),
    ("rollout_aw1_k2.npz",        "rollout_05_k2"),
    ("rollout_aw1_volume.npz",    "rollout_06_volume_v1_diverged"),
    ("rollout_aw1_volume_v2.npz", "rollout_07_volume_v2_diverged"),
    ("rollout_aw1_volume_v3.npz", "rollout_08_volume_v3_diverged"),
]


def main():
    repo = Path(__file__).resolve().parents[1]
    data = repo / "heat_python" / "data"
    figs = repo / "heat_python" / "figs"
    x_mm = _depth_axis()
    for npz, folder in RECIPES:
        p = data / npz
        if not p.exists():
            print(f"  skip {npz} (missing)")
            continue
        out = figs / folder
        out.mkdir(parents=True, exist_ok=True)
        d = dict(np.load(p))
        err = np.abs(d["T_pred"][:, 1:-1] - d["T_gt"][:, 1:-1])
        static_heatmap(d["time"], d["T_gt"], d["T_pred"], x_mm,
                       out / "rollout_spacetime.png")
        profile_animation(d["time"], d["T_gt"], d["T_pred"], x_mm,
                          out / "rollout_profile.gif")
        print(f"  -> {folder}  (rollout mean |err| {err.mean():.1f} K)")


if __name__ == "__main__":
    main()
