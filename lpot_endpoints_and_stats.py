import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def cohens_d(group1, group2):
    """Calculate Cohen's d for effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def plot_and_test_endpoints(csv_path, output_dir):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Isolate the final performance (the end of the adaptation loop)
    idx = df.groupby(['optimizer', 'shift', 'seed', 'n_pool', 'batch_size', 'hidden_dim'])['samples_seen'].idxmax()
    final_df = df.loc[idx]

    # 2. VISUALIZATION: The Split Violin Plot
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))
    
    palette = {"Adam (OG)": "#e74c3c", "AdamW + Cosine": "#2980b9"}
    
    sns.violinplot(
        data=final_df, x="shift", y="test_f1", hue="optimizer", 
        split=True, inner="quart", palette=palette, linewidth=1.5
    )
    
    plt.title("Final Target Recovery by Optimizer and Shift Level", fontsize=16, fontweight='bold')
    plt.xlabel("Distribution Shift Severity (k)", fontsize=14)
    plt.ylabel("Final Test Macro F1", fontsize=14)
    
    # Clean up legend
    plt.legend(title="Optimization Strategy", loc="lower right", frameon=True)
    
    plot_path = os.path.join(output_dir, "endpoint_distribution_violin.png")
    pdf_path = os.path.join(output_dir, "endpoint_distribution_violin.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"✅ Endpoint plots saved to {output_dir}\n")

    # 3. STATISTICAL SIGNIFICANCE REPORT
    print(f"{'='*70}")
    print("📊 STATISTICAL SIGNIFICANCE REPORT (AdamW + Cosine vs Adam OG)")
    print("Test: Welch's t-test (unequal variances) | Metric: Final Test Macro F1")
    print(f"{'='*70}")
    
    # FIX: Defined headers as variables to avoid f-string escape character errors
    col_shift = "Shift"
    col_pval = "p-value"
    col_cohen = "Cohen's d"
    col_sig = "Significance"
    print(f"{col_shift:<10} | {col_pval:<15} | {col_cohen:<15} | {col_sig}")
    print("-" * 70)
    
    shifts = sorted(final_df['shift'].unique())
    for shift_val in shifts:
        # Extract the arrays of F1 scores
        adamw_data = final_df[(final_df['shift'] == shift_val) & (final_df['optimizer'] == 'AdamW + Cosine')]['test_f1'].values
        adam_data = final_df[(final_df['shift'] == shift_val) & (final_df['optimizer'] == 'Adam (OG)')]['test_f1'].values
        
        # Run Welch's t-test (we use Welch's because the variances are clearly unequal)
        t_stat, p_val = stats.ttest_ind(adamw_data, adam_data, equal_var=False)
        
        # Calculate Effect Size
        d = cohens_d(adamw_data, adam_data)
        
        # Determine significance stars
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        print(f"{shift_val:<10.1f} | {p_val:<15.2e} | {d:<15.2f} | {sig}")
    print(f"{'='*70}")
    print("* Note: Cohen's d > 0.8 is considered a 'large' effect size.")

if __name__ == "__main__":
    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    CSV_FILE = os.path.join(RESULTS_DIR, "master_adaptation_results.csv")
    FIG_DIR = os.path.join(RESULTS_DIR, "figures")
    
    plot_and_test_endpoints(CSV_FILE, FIG_DIR)
