import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_hyperparameter_effects(csv_path, output_dir):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. Filter to just the winning optimizer and the hardest shift
    df_hard = df[(df['optimizer'] == 'AdamW + Cosine') & (df['shift'] == 5.0)].copy()

    # Convert numeric categories to strings for better legend formatting
    df_hard['hidden_dim'] = df_hard['hidden_dim'].astype(str) + " Neurons"
    df_hard['n_pool'] = (df_hard['n_pool'] / 1000).astype(int).astype(str) + "k Samples"
    df_hard['batch_size'] = "Batch " + df_hard['batch_size'].astype(str)

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Create a 1x3 grid
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharey=True)
    fig.suptitle("AdamW Recovery Dynamics under Extreme Shift (k=5.0)", fontsize=16, fontweight='bold', y=1.05)

    # --- PLOT 1: Network Capacity ---
    sns.lineplot(
        data=df_hard, x="samples_seen", y="test_f1", hue="hidden_dim", 
        palette="viridis", errorbar=('ci', 95), ax=axes[0], linewidth=2
    )
    axes[0].set_title("Effect of Network Capacity", fontweight='bold')
    axes[0].set_ylabel("Test Macro F1")
    axes[0].set_xlabel("Target Samples Seen")

    # --- PLOT 2: Unlabeled Data Volume ---
    sns.lineplot(
        data=df_hard, x="samples_seen", y="test_f1", hue="n_pool", 
        palette="magma", errorbar=('ci', 95), ax=axes[1], linewidth=2
    )
    axes[1].set_title("Effect of Target Pool Size", fontweight='bold')
    axes[1].set_xlabel("Target Samples Seen")

    # --- PLOT 3: Gradient Stability ---
    sns.lineplot(
        data=df_hard, x="samples_seen", y="test_f1", hue="batch_size", 
        palette="cubehelix", errorbar=('ci', 95), ax=axes[2], linewidth=2
    )
    axes[2].set_title("Effect of Batch Size", fontweight='bold')
    axes[2].set_xlabel("Target Samples Seen")

    plt.tight_layout()
    
    png_path = os.path.join(output_dir, "hyperparameter_main_effects.png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {png_path}")

if __name__ == "__main__":
    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    CSV_FILE = os.path.join(RESULTS_DIR, "master_adaptation_results.csv")
    FIG_DIR = os.path.join(RESULTS_DIR, "figures")
    plot_hyperparameter_effects(CSV_FILE, FIG_DIR)
