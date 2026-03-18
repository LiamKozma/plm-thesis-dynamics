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
        'lines.linewidth': 3,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight'
    })

def plot_true_recovery_dynamics(df, output_dir):
    plt.figure(figsize=(12, 7))
    
    # 1. Isolate the "Ground Zero" setup: 
    # - n_pool = 500k (so the model isn't starved)
    # - Highest capacity model (H=1024, B=256) to see optimal biological recovery
    ground_zero_df = df[(df['n_pool'] == 500000) & 
                        (df['hidden_dim'] == 1024) & 
                        (df['batch_size'] == 256)].copy()
    
    # 2. Map the Shift parameters to their actual average Wasserstein Distances
    # This directly answers your question of "If data is X distance away..."
    shift_to_w = ground_zero_df.groupby('shift')['wasserstein_distance'].mean().to_dict()
    
    # Create a new cleaner label for the legend
    ground_zero_df['Distribution'] = ground_zero_df['shift'].apply(
        lambda s: f"Shift: {s} (Wasserstein: ~{shift_to_w[s]:.2f})"
    )

    # 3. Calculate the baseline F1 (samples_seen == 0) and the 95% threshold
    baseline_f1 = ground_zero_df[ground_zero_df['samples_seen'] == 0]['test_f1'].mean()
    recovery_threshold = baseline_f1 * 0.95

    # 4. Plot the time-series learning curves
    ax = sns.lineplot(
        data=ground_zero_df,
        x='samples_seen',
        y='test_f1',
        hue='Distribution',
        palette=['#2ca02c', '#ff7f0e', '#d62728'], # Green (Easy), Orange (Medium), Red (Hard)
        errorbar=('ci', 95), # Still shows variance across your 3 seeds
        alpha=0.9
    )

    # 5. Draw the Baseline and Recovery Threshold lines
    plt.axhline(baseline_f1, color='black', linestyle='-', linewidth=2, label=f'Original Baseline ({baseline_f1:.2f})')
    plt.axhline(recovery_threshold, color='black', linestyle='--', linewidth=2, alpha=0.7, label=f'95% Recovery Threshold ({recovery_threshold:.2f})')

    plt.title('Protein Model Recovery Dynamics in Biological "Sweet Spot"')
    plt.xlabel('Target Samples Evaluated (Active Learning)')
    plt.ylabel('Test Macro F1')
    
    # Format X-axis in thousands (k)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x > 0 else '0'))
    

    # Legend formatting
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Dataset Shift Severity")
    
    # Save it!
    output_path = os.path.join(output_dir, 'fig7_ground_zero_dynamics.pdf')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    setup_plotting_style()
    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    FILE_PATH = os.path.join(RESULTS_DIR, "master_adaptation_results.csv")
    OUTPUT_DIR = os.path.join(RESULTS_DIR, "figures")
    
    df = pd.read_csv(FILE_PATH)
    plot_true_recovery_dynamics(df, OUTPUT_DIR)
    print("Saved the Ground Zero Dynamics plot (Figure 7).")
