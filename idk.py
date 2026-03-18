import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_capacity_interaction(df, output_dir):
    # Filter for Shift 1.0 and Seed 43 to keep the data clean
    base_df = df[(df['seed'] == 43) & (df['shift'] == 1.0)].copy()
    
    # Isolate our two opposing architectures
    fragile_df = base_df[(base_df['batch_size'] == 64) & (base_df['hidden_dim'] == 512)].copy()
    anchor_df = base_df[(base_df['batch_size'] == 256) & (base_df['hidden_dim'] == 1024)].copy()

    # Melt both for seaborn plotting
    fragile_melt = pd.melt(fragile_df, id_vars=['samples_seen'], value_vars=['train_loss', 'test_ce'], var_name='Metric', value_name='Loss')
    anchor_melt = pd.melt(anchor_df, id_vars=['samples_seen'], value_vars=['train_loss', 'test_ce'], var_name='Metric', value_name='Loss')
    
    # Map friendly names
    metric_map = {'train_loss': 'Adaptation Loss (Train)', 'test_ce': 'Generalization Loss (Test)'}
    fragile_melt['Metric'] = fragile_melt['Metric'].map(metric_map)
    anchor_melt['Metric'] = anchor_melt['Metric'].map(metric_map)

    # Set up a 1x2 side-by-side subplot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle('Architectural Capacity vs. Catastrophic Forgetting (Shift 1.0)', fontsize=16, y=1.05)

    # Plot 1: The Fragile Model (The Cliff)
    sns.lineplot(ax=axes[0], data=fragile_melt, x='samples_seen', y='Loss', hue='Metric', linewidth=3, palette=['#2ca02c', '#d62728'])
    axes[0].set_title('Fragile Architecture\n(Batch: 64 | Hidden Dim: 512)')
    axes[0].set_xlabel('Target Samples Evaluated')
    axes[0].set_ylabel('Cross-Entropy Loss')
    
    # Plot 2: The Anchor Model (The Flatline)
    sns.lineplot(ax=axes[1], data=anchor_melt, x='samples_seen', y='Loss', hue='Metric', linewidth=3, palette=['#2ca02c', '#d62728'], legend=False)
    axes[1].set_title('Anchor Architecture\n(Batch: 256 | Hidden Dim: 1024)')
    axes[1].set_xlabel('Target Samples Evaluated')

    sns.despine()
    
    output_path = os.path.join(output_dir, 'fig9_capacity_interaction.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams.update({'font.family': 'serif'})
    
    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    FILE_PATH = os.path.join(RESULTS_DIR, "master_adaptation_results.csv")
    OUTPUT_DIR = os.path.join(RESULTS_DIR, "figures")
    
    df = pd.read_csv(FILE_PATH)
    plot_capacity_interaction(df, OUTPUT_DIR)
    print("Saved Capacity Interaction Figure 9.")
