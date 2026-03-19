import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_adaptation_dynamics(csv_path, output_dir):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set publication-ready aesthetics
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # We want to plot Test F1 and Test CE Loss.
    # We will create a 2 (Metrics) x 3 (Shifts) grid.
    shifts = sorted(df['shift'].unique())
    
    fig, axes = plt.subplots(nrows=2, ncols=len(shifts), figsize=(18, 10), sharex=True)
    
    # Custom color palette for the optimizers
    palette = {"Adam (OG)": "#e74c3c", "AdamW + Cosine": "#2980b9"}

    print("Generating plots (this may take a moment as seaborn calculates confidence intervals)...")

    for col_idx, shift_val in enumerate(shifts):
        shift_data = df[df['shift'] == shift_val]
        
        # --- ROW 1: Macro F1 Score ---
        ax_f1 = axes[0, col_idx]
        sns.lineplot(
            data=shift_data, 
            x="samples_seen", 
            y="test_f1", 
            hue="optimizer", 
            palette=palette,
            errorbar=('ci', 95), # 95% Confidence Interval across seeds/configs
            ax=ax_f1,
            linewidth=2
        )
        ax_f1.set_title(f"Distribution Shift (k = {shift_val})", fontsize=16, fontweight='bold')
        ax_f1.set_ylabel("Test Macro F1" if col_idx == 0 else "")
        ax_f1.set_xlabel("")
        
        # Remove individual legends, we'll add one global legend later
        if ax_f1.get_legend(): ax_f1.get_legend().remove()

        # --- ROW 2: Cross Entropy Loss ---
        ax_ce = axes[1, col_idx]
        sns.lineplot(
            data=shift_data, 
            x="samples_seen", 
            y="test_ce", 
            hue="optimizer", 
            palette=palette,
            errorbar=('ci', 95),
            ax=ax_ce,
            linewidth=2
        )
        ax_ce.set_ylabel("Test CE Loss" if col_idx == 0 else "")
        ax_ce.set_xlabel("Target Samples Seen")
        
        if ax_ce.get_legend(): ax_ce.get_legend().remove()

    # Create a single global legend at the top
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=14, title="Fine-Tuning Strategy", bbox_to_anchor=(0.5, 1.05))

    # Adjust layout
    plt.tight_layout()
    
    # Save high-res PNG and vector PDF
    png_path = os.path.join(output_dir, "forgetting_dynamics_grid.png")
    pdf_path = os.path.join(output_dir, "forgetting_dynamics_grid.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"\nPlots saved successfully to:")
    print(f" -> {png_path}")
    print(f" -> {pdf_path}")

if __name__ == "__main__":
    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    CSV_FILE = os.path.join(RESULTS_DIR, "master_adaptation_results.csv")
    FIG_DIR = os.path.join(RESULTS_DIR, "figures")
    
    plot_adaptation_dynamics(CSV_FILE, FIG_DIR)
