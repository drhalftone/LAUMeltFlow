"""Flux-form vs direct on the held-out aw2 forcing: error-vs-time (log scale).

The figure for white-paper Sec 4.4: the direct per-cell-delta model diverges to
tens of thousands of kelvin while the conservative flux-form stays bounded.

    python -m heat_python.viz_flux_compare
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(f):
    d = np.load(Path(__file__).resolve().parents[1] / f)
    return d["time"], d["err"].mean(1), d["err"].max(1)


def main():
    repo = Path(__file__).resolve().parents[1]
    runs = [
        ("heat_python/data/rollout_direct_aw2.npz", "direct per-cell delta", "#c0392b"),
        ("heat_python/data/rollout_flux_aw2.npz", "conservative flux-form", "#1f6fb2"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for f, lbl, c in runs:
        t, em, ex = load(f)
        ax.semilogy(t, em, lw=2.4, color=c, label=f"{lbl} (mean)")
        ax.semilogy(t, ex, lw=1.0, color=c, ls="--", alpha=0.6, label=f"{lbl} (max)")
    ax.axhline(7.6, color="#2e7d32", ls=":", lw=1.5, label="aw1 gas-off best (7.6 K)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("interior T rollout |error| [K]  (log)")
    ax.set_title("Held-out aw2 (gas) rollout: the conservative flux-form removes the divergence\n"
                 "direct predictor runs to ~25,000 K; flux-form stays bounded near 100 K")
    ax.legend(fontsize=9, loc="center right")
    fig.tight_layout()
    out = repo / "heat_python" / "figs" / "flux_aw2_compare.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
