"""Regenerate the param-sweep plot as bar charts (val loss + rollout drift).

Reads outputs/sweep/results.json, writes outputs/sweep/param_sweep.png.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fmt_params(n):
    if n >= 1000:
        return f"{n/1000:.1f}K"
    return str(n)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    res_path = os.path.join(here, "outputs", "sweep", "results.json")
    with open(res_path) as f:
        results = json.load(f)

    # Sort by hidden_dim for consistent x ordering
    results.sort(key=lambda r: r["hidden_dim"])

    h_dims = [r["hidden_dim"] for r in results]
    params = [r["n_params"] for r in results]
    val_losses = [r["best_val_loss"] for r in results]
    # Get rollout pos_err_final at N=16
    rollouts = [r["chain_results"]["16"]["pos_err_final"] for r in results]

    labels = [f"h={h}\n({fmt_params(p)})" for h, p in zip(h_dims, params)]

    # Highlight h=32 (the sweet-spot config)
    sweet_h = 32
    colors = ["tab:green" if h == sweet_h else "tab:blue" for h in h_dims]

    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 16,
        "axes.labelsize": 14, "xtick.labelsize": 12,
        "ytick.labelsize": 12, "legend.fontsize": 12,
    })

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax = axes[0]
    bars = ax.bar(range(len(h_dims)), val_losses, color=colors,
                  edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(range(len(h_dims)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Best validation MSE (normalized)")
    ax.set_title("Training accuracy vs. model size")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    for bar, v in zip(bars, val_losses):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v * 1.4, f"{v:.1e}",
                ha="center", va="bottom", fontsize=10)

    ax = axes[1]
    bars = ax.bar(range(len(h_dims)), rollouts, color=colors,
                  edgecolor="black", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(range(len(h_dims)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Final rollout position error (m)")
    ax.set_title("Rollout drift vs. model size (N=16, 1 s)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    for bar, v in zip(bars, rollouts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v * 1.4,
                f"{v*100:.1f} cm" if v < 1 else f"{v:.2f} m",
                ha="center", va="bottom", fontsize=10)

    # Legend marker for highlighted bar
    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(facecolor="tab:green", edgecolor="black",
                       label=f"h={sweet_h} (sweet spot)"),
                 Patch(facecolor="tab:blue", edgecolor="black",
                       label="other configs")],
        loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(here, "outputs", "sweep", "param_sweep.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
