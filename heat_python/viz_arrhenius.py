"""How long the TACOT components take to react, versus how long reentry lasts.

Reads the real kinetics from an example case's solid.dat and plots the
reaction timescale 1/k against temperature, with the aero pulse duration drawn
across it. Where a curve crosses that line is where material stops surviving
the event. Study aid, not a paper figure.

    python -m heat_python.viz_arrhenius
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOLID = ROOT / "heat_2026-04-11_1837/examples/aw2_tc21/solid.dat"
OUT = ROOT / "heat_python/figs/method/arrhenius.png"

PULSE = 60.0          # aw2 aero pulse duration [s]
BLUE, ORANGE = "#3B82C4", "#E07A28"
INK, MUTED, GRID = "#22262b", "#6b7280", "#dfe1e5"


def read_species(path: Path):
    rows = [l.split() for l in path.read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    return [dict(rhov=float(r[0]), rhoc=float(r[1]), A=float(r[2]),
                 psi=float(r[3]), EoR=float(r[4]), gam=float(r[5]),
                 Tmin=float(r[6])) for r in rows[2:]]


def T_at(s, t):
    """Temperature where this component's reaction timescale equals t seconds."""
    return s["EoR"] / np.log(s["A"] * t)


def main():
    sp = [s for s in read_species(SOLID) if s["A"] > 0]
    T = np.linspace(298.0, 1900.0, 3000)

    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ylo, yhi = 1e-4, 1e10
    ax.axhspan(PULSE, yhi, color="#3B82C4", alpha=0.055, zorder=0)
    ax.axhspan(ylo, PULSE, color="#E07A28", alpha=0.055, zorder=0)
    ax.text(316, 1.2e4, "SURVIVES", fontsize=10, color="#2f6ea8",
            fontweight="bold", va="center")
    ax.text(1880, 4e-4, "CONSUMED", fontsize=10, color="#b35f18",
            fontweight="bold", va="center", ha="right")

    for s, c, name in zip(sp, (BLUE, ORANGE), ("Component 1", "Component 2")):
        tau = 1.0 / (s["A"] * np.exp(-s["EoR"] / T))
        tau = np.where(T >= s["Tmin"], tau, np.nan)
        ax.plot(T, tau, color=c, lw=2.2, zorder=4, solid_capstyle="round")

    main_sp = sp[1]
    T60, T1 = T_at(main_sp, PULSE), T_at(main_sp, 1.0)
    ax.axvspan(T60, T1, color=ORANGE, alpha=0.13, zorder=1)
    ax.plot([T60, T1], [PULSE, 1.0], "o", color=ORANGE, ms=7,
            mec="#fcfcfb", mew=1.6, zorder=6)

    ax.axhline(PULSE, color=INK, lw=1.3, ls=(0, (5, 3)), zorder=3)
    ax.text(1890, PULSE * 1.6, "60 s  aero pulse", ha="right", fontsize=9.5,
            color=INK, fontweight="medium")

    mid = (T60 + T1) / 2
    ax.annotate(f"{T1 - T60:.0f} K wide", xy=(mid, 6e8),
                ha="center", fontsize=14, color="#b35f18", fontweight="bold")
    ax.annotate("cooler than this band, it lasts the whole flight\n"
                "hotter than it, gone in a second",
                xy=(mid, 6e7), ha="center", fontsize=9.2,
                color="#b35f18", linespacing=1.4)
    ax.annotate("", xy=(T60, 4e6), xytext=(T1, 4e6),
                arrowprops=dict(arrowstyle="<->", color="#b35f18", lw=1.2))

    ax.annotate("Component 2\nmain resin, leaves the char",
                xy=(1180, 1.0 / (main_sp["A"] * np.exp(-main_sp["EoR"] / 1180))),
                xytext=(26, 30), textcoords="offset points", fontsize=9.5,
                color=ORANGE, fontweight="medium", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.0,
                                shrinkA=0, shrinkB=4))
    s1 = sp[0]
    ax.annotate("Component 1\nlight volatiles",
                xy=(430, 1.0 / (s1["A"] * np.exp(-s1["EoR"] / 430))),
                xytext=(26, -46), textcoords="offset points", fontsize=9.5,
                color=BLUE, fontweight="medium", linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.0,
                                shrinkA=0, shrinkB=4))

    for tval, lab in ((1e-3, "1 ms"), (1.0, "1 s"), (3.6e3, "1 hour"),
                      (8.64e4, "1 day"), (3.15e7, "1 year")):
        ax.text(1893, tval, lab, ha="right", va="center", fontsize=7.6,
                color=MUTED, alpha=0.85)

    ax.set_yscale("log")
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(298, 1900)
    ax.set_xlabel("Material temperature   T   [K]", fontsize=10.5, color=INK)
    ax.set_ylabel("Time to react   1/k   [s]", fontsize=10.5, color=INK)
    ax.set_title("Pyrolysis is a switch, not a dial",
                 fontsize=13.5, color=INK, pad=13, loc="left")
    ax.text(0.0, 1.018, "TACOT 3.0 kinetics, aw2 case",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.grid(True, which="major", axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, facecolor=fig.get_facecolor())
    print(f"wrote {OUT}")
    for s, name in zip(sp, ("Component 1", "Component 2")):
        print(f"{name}: survives-60s below {T_at(s, PULSE):.0f} K, "
              f"gone-in-1s above {T_at(s, 1.0):.0f} K, "
              f"window {T_at(s, 1.0) - T_at(s, PULSE):.0f} K")


if __name__ == "__main__":
    main()
