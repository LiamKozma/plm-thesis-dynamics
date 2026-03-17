import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def setup_plotting_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams.update({
        'font.family': 'serif',
        'figure.autolayout': True,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight'
    })

def plot_clean_capacity_vs_shift(df, output_dir):
    plt.figure(figsize=(10, 6))
    
    # 1. Get the max F1 for each individual run
    adapt_df = df[df['samples_seen'] > 0]
    run_level_df = adapt_df.groupby(['seed', 'n_pool', 'shift', 'batch_size', 'hidden_dim'])['test_f1'].max().reset_index()
    run_level_df.rename(columns={'test_f1': 'max_achieved_f1'}, inplace=True)
    
    # Create the Architecture label
    run_level_df['Architecture'] = run_level_df.apply(
        lambda row: f"H:{int(row['hidden_dim'])} | B:{int(row['batch_size'])}", axis=1
    )

    # 2. Plotting using Seaborn's pointplot
    # Pointplot automatically calculates means and standard deviations, 
    # and neatly dodges the categories so they NEVER overlap!
    ax = sns.pointplot(
        data=run_level_df,
        x='shift',
        y='max_achieved_f1',
        hue='Architecture',
        dodge=0.4,       # Shifts the points slightly to avoid overlap
        errorbar='sd',   # Standard deviation for error bars
        markers=['o', 's', '^', 'D'],
        capsize=0.1,
        palette='Set1',
        linestyles='--'
    )

    plt.title('Adaptation Capacity vs. Controlled Distribution Shift')
    plt.xlabel('Distribution Shift Severity (k)')
    plt.ylabel('Max Recovered Test Macro F1 (± 1 Std Dev)')
    
    # Move legend outside the plot area
    plt.legend(title='Architecture', bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()
    
    output_path = os.path.join(output_dir, 'fig6_clean_capacity.pdf')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    setup_plotting_style()
    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    FILE_PATH = os.path.join(RESULTS_DIR, "master_adaptation_results.csv")
    OUTPUT_DIR = os.path.join(RESULTS_DIR, "figures")
    
    df = pd.read_csv(FILE_PATH)
    plot_clean_capacity_vs_shift(df, OUTPUT_DIR)
    print("Saved beautiful, readable Figure 6!")
