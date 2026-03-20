import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def configure_publication_aesthetics():
    """Applies high-contrast, colorblind-friendly, LaTeX-style aesthetics."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.edgecolor": "#333333",
            "grid.color": "#E5E5E5",
            "grid.linestyle": "--",
        }
    )


def plot_wasserstein_log_dynamics(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"❌ Error: Could not find data file at {csv_path}")
        sys.exit(1)

    print(f"Loading empirical data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Isolate asymptotic performance
    idx = df.groupby(
        ["optimizer", "shift", "seed", "n_pool", "batch_size", "hidden_dim"]
    )["samples_seen"].idxmax()
    final_df = df.loc[idx].copy().dropna(subset=["wasserstein"])

    # 2. Setup Canvas
    os.makedirs(output_dir, exist_ok=True)
    configure_publication_aesthetics()
    fig, ax = plt.subplots(figsize=(10, 7))

    # Accessible Wong Palette
    palette = {"Adam (OG)": "#D55E00", "AdamW + Cosine": "#0072B2"}
    markers = {"Adam (OG)": "o", "AdamW + Cosine": "X"}

    # 3. Plot the Zero-Shot Baseline (The anchor for Catastrophic Forgetting)
    baseline_f1 = 0.6906
    ax.axhline(
        y=baseline_f1, color="black", linestyle="--", linewidth=2.5, zorder=1
    )
    ax.text(
        x=final_df["wasserstein"].max(),
        y=baseline_f1 + 0.015,
        s="Zero-Shot Baseline (0.690)",
        color="black",
        ha="right",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )

    # 4. Plot Density Scatter (No Interpolation Lines)
    sns.scatterplot(
        data=final_df,
        x="wasserstein",
        y="test_f1",
        hue="optimizer",
        style="optimizer",
        palette=palette,
        markers=markers,
        alpha=0.75,
        s=120,
        ax=ax,
        edgecolor="white",
        linewidth=1,
        zorder=5,
    )

    # 5. Apply Log Scale to compress the empty gap honestly
    ax.set_xscale("log")

    # 6. Formatting
    ax.set_title(
        "Catastrophic Forgetting: Final Recovery vs. Topological Shift",
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Empirical Wasserstein Distance (Log Scale)", labelpad=10)
    ax.set_ylabel("Final Test Macro F1 (Asymptote)", labelpad=10)
    ax.set_ylim(0.0, 0.85)

    # Clean up legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title="Optimizer",
        loc="lower right",
        frameon=True,
        shadow=True,
        fancybox=True,
    )

    # 7. Save Artifacts
    plot_path = os.path.join(output_dir, "wasserstein_forgetting_log.png")
    pdf_path = os.path.join(output_dir, "wasserstein_forgetting_log.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"✅ Rigorous plots exported to {output_dir}\n")


if __name__ == "__main__":
    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    CSV_FILE = os.path.join(RESULTS_DIR, "master_adaptation_results_with_W.csv")
    FIG_DIR = os.path.join(RESULTS_DIR, "figures")

    plot_wasserstein_log_dynamics(CSV_FILE, FIG_DIR)
