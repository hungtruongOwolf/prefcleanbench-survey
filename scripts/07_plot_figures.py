"""
Step 7: Generate all figures for the survey report.
Usage: python scripts/07_plot_figures.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import os

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


def fig1_remove_vs_flip():
    """Grouped bar chart: Remove vs Flip across method families."""
    fig, ax = plt.subplots(figsize=(3.3, 2.4))

    methods = ["LLM-Judge", "RwGap", "VoteAll", "VoteMaj", "IFD-Gap"]
    remove_vals = [0.648, 0.623, 0.630, 0.668, 0.617]
    flip_vals = [0.610, 0.570, 0.590, 0.635, 0.583]

    x = np.arange(len(methods))
    w = 0.32
    ax.bar(x - w / 2, remove_vals, w, label="Remove", color="#2E86AB", edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, flip_vals, w, label="Flip", color="#E8505B", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Avg. Win-Tie Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=7.5)
    ax.set_ylim(0.54, 0.70)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.axhline(y=0.500, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.text(4.3, 0.503, "baseline", fontsize=6.5, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i in range(len(methods)):
        delta = remove_vals[i] - flip_vals[i]
        mid_y = max(remove_vals[i], flip_vals[i]) + 0.004
        ax.text(x[i], mid_y, f"+{delta:.3f}", ha="center", fontsize=6, color="#333", fontstyle="italic")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_remove_vs_flip.pdf"), bbox_inches="tight")
    plt.close()
    print("Figure 1: Remove vs Flip — done")


def fig2_filtering_rate():
    """Line chart: filtering rate effect on two datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.5))
    rates = [10, 20, 30, 40]

    # AnthropicHH
    ax1.plot(rates, [6.75, 6.88, 6.80, 6.58], "o-", color="#2E86AB", label="RwGap-R", markersize=5, linewidth=1.5)
    ax1.plot(rates, [6.52, 6.60, 6.62, 6.45], "s--", color="#E8505B", label="IFD-R", markersize=5, linewidth=1.5)
    ax1.plot(rates, [6.68, 6.78, 6.72, 6.50], "^-.", color="#6B9F36", label="IFD-Gap-R", markersize=5, linewidth=1.5)
    ax1.set_xlabel("Filtering Rate (%)")
    ax1.set_ylabel("Avg. Gold Reward")
    ax1.set_title("AnthropicHH (20K)")
    ax1.set_xticks(rates)
    ax1.legend(fontsize=7, loc="lower left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_ylim(6.3, 7.0)

    # UltraFeedback
    ax2.plot(rates, [6.10, 6.18, 6.12, 5.92], "o-", color="#2E86AB", label="RwGap-R", markersize=5, linewidth=1.5)
    ax2.plot(rates, [5.88, 5.95, 5.98, 5.78], "s--", color="#E8505B", label="IFD-R", markersize=5, linewidth=1.5)
    ax2.plot(rates, [6.02, 6.12, 6.08, 5.85], "^-.", color="#6B9F36", label="IFD-Gap-R", markersize=5, linewidth=1.5)
    ax2.set_xlabel("Filtering Rate (%)")
    ax2.set_ylabel("Avg. Gold Reward")
    ax2.set_title("UltraFeedback (20K)")
    ax2.set_xticks(rates)
    ax2.legend(fontsize=7, loc="lower left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_ylim(5.65, 6.35)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_filtering_rate.pdf"), bbox_inches="tight")
    plt.close()
    print("Figure 2: Filtering Rate — done")


def fig3_heatmap():
    """Heatmap: cross-algorithm WinTie averaged across datasets."""
    fig, ax = plt.subplots(figsize=(3.3, 2.2))

    data = np.array([
        [0.668, 0.654, 0.626, 0.660],
        [0.648, 0.635, 0.624, 0.639],
        [0.668, 0.651, 0.640, 0.638],
    ])
    methods_short = ["VoteMaj-R", "LLM-Judge-R", "Tag-Cmp"]
    algos = ["DPO", "CPO", "KTO", "ORPO"]

    sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu",
                xticklabels=algos, yticklabels=methods_short, ax=ax,
                cbar_kws={"label": "Avg. WinTie", "shrink": 0.8},
                linewidths=0.8, linecolor="white",
                annot_kws={"size": 8.5, "fontweight": "bold"},
                vmin=0.61, vmax=0.68)

    for j in range(data.shape[1]):
        max_i = np.argmax(data[:, j])
        ax.add_patch(plt.Rectangle((j, max_i), 1, 1, fill=False, edgecolor="#E8505B", linewidth=2))

    ax.set_title("Win-Tie Rate (avg. across datasets)", fontsize=9, pad=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_heatmap.pdf"), bbox_inches="tight")
    plt.close()
    print("Figure 3: Heatmap — done")


def fig4_cost_performance():
    """Scatter: cost vs performance for removal methods."""
    fig, ax = plt.subplots(figsize=(3.3, 2.6))

    names = ["Tag-Cmp", "Tag-Div", "IFD-Gap-R", "IFD-R", "RwGap-R", "VoteAll-R", "VoteMaj-R", "LLM-Judge-R"]
    perf = [0.668, 0.595, 0.617, 0.585, 0.623, 0.630, 0.668, 0.648]
    cost = [0.5, 0.5, 1.0, 1.0, 4.0, 2.0, 2.5, 8.0]

    colors = {
        "Tag-Cmp": "#6B9F36", "Tag-Div": "#6B9F36", "IFD-Gap-R": "#6B9F36", "IFD-R": "#6B9F36",
        "RwGap-R": "#2E86AB", "VoteAll-R": "#2E86AB", "VoteMaj-R": "#2E86AB",
        "LLM-Judge-R": "#E8505B",
    }
    markers = {
        "Tag-Cmp": "D", "Tag-Div": "D", "IFD-Gap-R": "D", "IFD-R": "D",
        "RwGap-R": "s", "VoteAll-R": "s", "VoteMaj-R": "s",
        "LLM-Judge-R": "^",
    }

    for i, m in enumerate(names):
        ax.scatter(cost[i], perf[i], c=colors[m], marker=markers[m], s=60, zorder=5, edgecolors="white", linewidth=0.5)
        ox, oy = 0.15, 0.003
        if m == "VoteMaj-R": oy = 0.006
        elif m == "Tag-Cmp": oy = -0.008
        elif m == "Tag-Div": oy = -0.005
        elif m == "IFD-R": oy = -0.007
        ax.annotate(m, (cost[i], perf[i]), fontsize=6, xytext=(cost[i] + ox, perf[i] + oy))

    p1 = mpatches.Patch(color="#6B9F36", label="Heuristic")
    p2 = mpatches.Patch(color="#2E86AB", label="Reward Model")
    p3 = mpatches.Patch(color="#E8505B", label="LLM Judge")
    ax.legend(handles=[p1, p2, p3], fontsize=7, loc="lower right")

    ax.set_xlabel("Relative Compute Cost (GPU-hours equiv.)")
    ax.set_ylabel("Avg. Win-Tie Rate")
    ax.set_xscale("log")
    ax.set_xlim(0.3, 12)
    ax.set_ylim(0.57, 0.69)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_cost_perf.pdf"), bbox_inches="tight")
    plt.close()
    print("Figure 4: Cost-Performance — done")


if __name__ == "__main__":
    fig1_remove_vs_flip()
    fig2_filtering_rate()
    fig3_heatmap()
    fig4_cost_performance()
    print(f"\nAll figures saved to {OUT_DIR}/")
