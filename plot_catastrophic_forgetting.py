import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_catastrophic_forgetting_diagnostic(df, output_dir):
    plt.figure(figsize=(12, 6))

    # Isolate a single, representative run to cleanly show the phenomenon
    # e.g., Seed 43, Shift 1.0, Batch 64, Hidden 512
    diagnostic_run = df[
        (df["seed"] == 43)
        & (df["shift"] == 1.0)
        & (df["batch_size"] == 64)
        & (df["hidden_dim"] == 512)
    ].copy()

    # We want to plot both losses on the same graph, so we melt the dataframe
    melted_df = pd.melt(
        diagnostic_run,
        id_vars=["samples_seen"],
        value_vars=["train_loss", "test_ce"],
        var_name="Metric",
        value_name="Cross-Entropy Loss",
    )

    # Rename for cleaner legend
    melted_df["Metric"] = melted_df["Metric"].map(
        {
            "train_loss": "Adaptation Batch Loss (Train)",
            "test_ce": "Generalization Loss (Test)",
        }
    )

    sns.lineplot(
        data=melted_df,
        x="samples_seen",
        y="Cross-Entropy Loss",
        hue="Metric",
        linewidth=3,
        palette=["#2ca02c", "#d62728"],
    )

    plt.title("Diagnostic: Catastrophic Forgetting via Aggressive Fine-Tuning")
    plt.xlabel("Target Samples Evaluated")
    plt.ylabel("Cross-Entropy Loss")

    # Add an annotation pointing out the divergence
    plt.annotate(
        "Model overfits to adaptation batches\nwhile test generalization collapses.",
        xy=(32000, 8.0),
        xytext=(50000, 6.0),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    sns.despine()
    plt.legend(title="Evaluation Metric")

    output_path = os.path.join(output_dir, "fig8_diagnostic_forgetting.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams.update({"font.family": "serif"})

    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    FILE_PATH = os.path.join(RESULTS_DIR, "master_adaptation_results.csv")
    OUTPUT_DIR = os.path.join(RESULTS_DIR, "figures")

    df = pd.read_csv(FILE_PATH)
    plot_catastrophic_forgetting_diagnostic(df, OUTPUT_DIR)
    print("Saved Diagnostic Figure 8.")
