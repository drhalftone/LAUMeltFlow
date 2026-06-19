"""aw2 surrogate: error-vs-time across recipes, showing what adaptive cadence
fixed and what residual remains.

    python -m heat_python.viz_aw2_progress
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load(f):
    d = np.load(Path(__file__).resolve().parents[1] / f)
    return d["time"], d["err"].mean(1)


def main():
    runs = [
        ("heat_python/data/rollout_aw2sweep_ms.npz", "fixed coarse (652 steps)", "#7f7f7f"),
        ("heat_python/data/rollout_aw2_fine_ms.npz", "fixed fine (6515 steps)", "#d39a2d"),
        ("heat_python/data/rollout_aw2_adaptive.npz", "ADAPTIVE (945 steps)", "#1f6fb2"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for f, lbl, c in runs:
        try:
            t, e = load(f)
            ax.semilogy(t, e, lw=2.2, color=c, label=lbl)
        except FileNotFoundError:
            pass
    ax.axhline(7.6, color="#2e7d32", ls=":", lw=1.5, label="aw1 (works, 7.6 K)")
    ax.axvline(60, color="0.8", ls="--", lw=1, alpha=0.7)
    ax.text(60, ax.get_ylim()[1] * 0.5, " flux off", fontsize=8, color="0.5")
    ax.set_xlabel("time [s]"); ax.set_ylabel("mean rollout |error| [K]  (log)")
    ax.set_title("aw2 surrogate: adaptive cadence tames the surface transient,\n"
                 "but a residual bulk-conduction drift remains over the plateau")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "heat_python" / "figs" / "aw2_progress.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
