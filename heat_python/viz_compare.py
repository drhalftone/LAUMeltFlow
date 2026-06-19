"""New rollout visualizations (kept separate from viz_rollout.py outputs):

  1. Thermocouple traces: temperature vs time at a few fixed depths, solver vs
     the multi-step GNN. The classic TPS diagnostic / white-paper figure.
  2. Model-progression comparison: mean rollout error vs time for the three
     training recipes (no-noise -> noise -> multi-step), showing the blow-up
     get tamed then sharpened.

    python -m heat_python.viz_compare
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .case import load_case


def _depth_axis():
    repo = Path(__file__).resolve().parents[1]
    case = load_case(repo / "heat_2026-04-11_1837" / "examples" / "aw1" / "heat.case")
    n = case.nbrn
    return (np.arange(n) + 0.5) * (case.l0 / n) * 1e3, n


def thermocouple_traces(ms_path, out_png):
    d = dict(np.load(ms_path))
    time = d["time"]
    gt = d["T_gt"][:, 1:-1]; pr = d["T_pred"][:, 1:-1]
    x_mm, n = _depth_axis()
    # probes at a few depths (outer/hot at right)
    targets_mm = [48, 42, 34, 24, 10]
    cells = [int(np.argmin(np.abs(x_mm - t))) for t in targets_mm]
    colors = plt.cm.plasma(np.linspace(0.05, 0.85, len(cells)))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for c, col in zip(cells, colors):
        ax.plot(time, gt[:, c], "-", color=col, lw=2,
                label=f"x={x_mm[c]:.0f} mm")
        ax.plot(time, pr[:, c], "--", color=col, lw=1.6)
    ax.plot([], [], "k-", label="solver (truth)")
    ax.plot([], [], "k--", label="GNN surrogate")
    ax.set_xlabel("time [s]"); ax.set_ylabel("temperature [K]")
    ax.set_title("Thermocouple traces: GNN vs solver (multi-step model)\n"
                 "outer/hot face on the right (x=48 mm), inner on the left")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"  saved {out_png}")


def error_progression(paths_labels, out_png):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for path, label, style in paths_labels:
        if not Path(path).exists():
            continue
        d = dict(np.load(path))
        time = d["time"]
        err = np.abs(d["T_pred"][:, 1:-1] - d["T_gt"][:, 1:-1])
        ax.plot(time, err.mean(1), style, lw=2, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("time [s]"); ax.set_ylabel("mean interior |ΔT| [K]  (log)")
    ax.set_title("Rollout error accumulation, by training recipe")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"  saved {out_png}")


def char_front_compare(rollout_path, out_png, virgin_frac=0.98):
    """Overlay the solver and GNN char fronts on the temperature-error field,
    plus a side panel of char-front depth vs time. Needs rho_gt/rho_pred in the
    rollout npz (eval_rollout saves them). The char front is the deepest cell
    (from the hot/outer face) with density below virgin_frac of virgin."""
    d = dict(np.load(rollout_path))
    if "rho_pred" not in d:
        print(f"  {rollout_path} has no density (re-run eval_rollout); skipping")
        return
    time = d["time"]
    x_mm, n = _depth_axis()
    Terr = np.abs(d["T_pred"][:, 1:-1] - d["T_gt"][:, 1:-1])
    virgin = float(d["rhov_bulk"])

    def front(rho):
        rho = rho[:, 1:-1]
        f = []
        for s in range(len(time)):
            c = np.where(rho[s] < virgin_frac * virgin)[0]
            f.append(x_mm[c.min()] if c.size else x_mm[-1])
        return np.array(f)

    f_gt, f_pred = front(d["rho_gt"]), front(d["rho_pred"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ext = [x_mm[0], x_mm[-1], time[-1], time[0]]
    im = ax[0].imshow(Terr, aspect="auto", extent=ext, cmap="viridis")
    ax[0].plot(f_gt, time, "w-", lw=2.2, label="char front: solver")
    ax[0].plot(f_pred, time, "r--", lw=1.8, label="char front: GNN")
    ax[0].set_title("Temperature |error| with char fronts")
    ax[0].set_xlabel("x [mm]  (hot/outer at right)"); ax[0].set_ylabel("time [s]")
    ax[0].legend(loc="lower left", fontsize=8)
    fig.colorbar(im, ax=ax[0], label="|ΔT| [K]")

    ax[1].plot(time, x_mm[-1] - f_gt, "k-", lw=2.2, label="solver")
    ax[1].plot(time, x_mm[-1] - f_pred, "r--", lw=1.8, label="GNN")
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("char depth from outer face [mm]")
    ax[1].set_title("Char-front penetration vs time")
    ax[1].legend(loc="upper left")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"  saved {out_png}")


def main():
    repo = Path(__file__).resolve().parents[1]
    data = repo / "heat_python" / "data"
    out = repo / "heat_python" / "figs" / "compare"
    out.mkdir(parents=True, exist_ok=True)
    thermocouple_traces(data / "rollout_aw1_ms.npz",
                        out / "thermocouple_traces.png")
    char_front_compare(data / "rollout_aw1_ms.npz",
                       out / "char_front_compare.png")
    error_progression([
        (data / "rollout_aw1.npz", "one-step (no noise)", "-"),
        (data / "rollout_aw1_noise.npz", "+ noise injection", "-"),
        (data / "rollout_aw1_ms.npz", "+ multi-step rollout", "-"),
    ], out / "error_progression.png")


if __name__ == "__main__":
    main()
