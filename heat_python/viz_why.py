"""Why the aw1 surrogate works and the aw2 surrogate fails -- the REAL cause.

The cause is temporal, not spatial. Per cell, aw2's field is not steeper than
aw1's (its finer mesh spreads the gradient over more cells). What differs is the
per-step change the model must predict: aw2's surface jumps ~18x more per
super-step than aw1's, because aw2's surface heats explosively while the
surrogate steps at a fixed coarse cadence.

    python -m heat_python.viz_why
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load(rollout, n, l0):
    d = np.load(Path(__file__).resolve().parents[1] / rollout)
    return d["time"], d["T_gt"][:, 1:-1], d["err"]


def main():
    t1, gt1, err1 = load("heat_python/data/rollout_aw1_n100.npz", 100, 0.05)
    t2, gt2, err2 = load("heat_python/data/rollout_aw2sweep_ms.npz", 500, 0.05)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    blue, red = "#1f6fb2", "#c0392b"

    # (a) SPATIAL per-cell |dT| near the surface -- NOT the cause (aw2 gentler).
    a = ax[0]
    sp1 = np.abs(np.diff(gt1, axis=1)).max(0)            # per-cell max over time
    sp2 = np.abs(np.diff(gt2, axis=1)).max(0)
    a.plot(np.linspace(0, 1, len(sp1)), sp1, color=blue, lw=2, label="aw1 (100 cells)")
    a.plot(np.linspace(0, 1, len(sp2)), sp2, color=red, lw=2, label="aw2 (500 cells)")
    a.set_title("(a) Spatial: per-cell |ΔT| vs depth\n(what the stencil sees) — NOT the cause",
                fontsize=10)
    a.set_xlabel("normalized depth  (hot wall = 1)")
    a.set_ylabel("max per-cell |ΔT| [K]")
    a.text(0.5, 0.06, "per cell, aw2 is no steeper than aw1", transform=a.transAxes,
           ha="center", fontsize=9, color="#555", style="italic")
    a.legend(fontsize=8)

    # (b) TEMPORAL per-step |dT| at the surface -- the REAL cause.
    b = ax[1]
    tp1 = np.abs(np.diff(gt1[:, -1]))                    # surface per-step jump
    tp2 = np.abs(np.diff(gt2[:, -1]))
    b.plot(t1[1:], tp1, color=blue, lw=2, label=f"aw1 (max {tp1.max():.0f} K/step)")
    b.plot(t2[1:], tp2, color=red, lw=2, label=f"aw2 (max {tp2.max():.0f} K/step)")
    b.set_title("(b) Temporal: surface |ΔT| per super-step\n(what the model must predict) — THE CAUSE",
                fontsize=10)
    b.set_xlabel("time [s]"); b.set_ylabel("surface |ΔT| per step [K]")
    b.text(0.5, 0.78, "aw2 surface leaps ~18x more per step\n(stiff, fast transient)",
           transform=b.transAxes, ha="center", fontsize=9, color="#a8421f", style="italic")
    b.legend(fontsize=8)

    # (c) consequence: rollout error vs time.
    c = ax[2]
    c.semilogy(t1, err1.mean(1), color=blue, lw=2, label="aw1 GNN (~7.6 K)")
    c.semilogy(t2, err2.mean(1), color=red, lw=2, label="aw2 GNN (diverges)")
    c.set_title("(c) Consequence: rollout error vs time", fontsize=10)
    c.set_xlabel("time [s]"); c.set_ylabel("mean |error| [K]  (log)")
    c.legend(fontsize=8)

    fig.suptitle("Why aw1 works and aw2 fails: a fast surface transient the coarse super-step can't track "
                 "(temporal, not spatial)", fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = Path(__file__).resolve().parents[1] / "heat_python" / "figs" / "why_aw1_aw2.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
