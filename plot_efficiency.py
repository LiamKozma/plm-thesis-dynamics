import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def setup_plotting_style():
    sns.set_theme(style="ticks", context="paper", font_scale=1.5)
    plt.rcParams.update({
        'font.family': 'serif',
        'figure.autolayout': True,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight'
    })

def plot_capacity_vs_wasserstein(df, output_dir):
    plt.figure(figsize=(10, 6))
    
    # Calculate the max F1 achieved during adaptation (samples_seen > 0)
    adapt_df = df[df['samples_seen'] > 0]
    max_f1_df = adapt_df.groupby(['seed', 'n_pool', 'shift', 'batch_size', 'hidden_dim', 'wasserstein_distance'])['test_f1'].max().reset_index()
    max_f1_df.rename(columns={'test_f1': 'max_achieved_f1'}, inplace=True)
    
    # Create a clean categorical label for Architecture
    max_f1_df['Architecture'] = max_f1_df.apply(
        lambda row: f"H:{row['hidden_dim']} | B:{row['batch_size']}", axis=1
    )

    # Plot using a scatterplot with a robust regression line (lmplot/regplot)
    ax = sns.scatterplot(
        data=max_f1_df,
        x='wasserstein_distance',
        y='max_achieved_f1',
        hue='Architecture',
        palette='Set1',
        s=150,
        alpha=0.8,
        edgecolor='k'
    )
    
    # Add a trendline to show the general degradation
    sns.regplot(
        data=max_f1_df,
        x='wasserstein_distance',
        y='max_achieved_f1',
        scatter=False,
        ax=ax,
        color='black',
        line_kws={'linestyle': '--', 'alpha': 0.5}
    )

    plt.title('Adaptation Capacity vs. Wasserstein Distance')
    plt.xlabel('Initial Wasserstein Distance from Source')
    plt.ylabel('Maximum Recovered Test Macro F1')
    
    # Move legend
    plt.legend(title='Architecture (Hidden | Batch)', bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine()
    
    output_path = os.path.join(output_dir, 'fig4_capacity_vs_wasserstein.pdf')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    setup_plotting_style()
    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    FILE_PATH = os.path.join(RESULTS_DIR, "master_adaptation_results.csv")
    OUTPUT_DIR = os.path.join(RESULTS_DIR, "figures")
    
    df = pd.read_csv(FILE_PATH)
    plot_capacity_vs_wasserstein(df, OUTPUT_DIR)
    print("Saved improved Figure 4!")
