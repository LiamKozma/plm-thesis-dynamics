import os
import glob
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

# ==========================================
# ACADEMIC PLOTTING AESTHETICS
# ==========================================
sns.set_theme(style="ticks", context="paper", font_scale=1.3)
plt.rcParams.update({
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.autolayout': True
})

def parse_wasserstein_from_log(filepath):
    """Extracts strictly the Wasserstein distance from the summary log."""
    with open(filepath, 'r') as f:
        for line in f:
            if "Wasserstein" in line:
                return float(line.split('|')[1].strip())
    return None

def calculate_recovery_cost(batch_csv_path, threshold):
    """
    Finds the first instance where the model achieves a Macro F1 >= threshold.
    Returns the number of samples required, or None if it never recovered.
    """
    df = pd.read_csv(batch_csv_path)
    # Filter for batches where F1 meets or exceeds the threshold
    recovered_df = df[df['test_f1'] >= threshold]
    
    if not recovered_df.empty:
        # Return the samples_seen from the FIRST batch that crossed the threshold
        return recovered_df.iloc[0]['samples_seen'], True
    else:
        # If it never crossed the threshold, return the max samples seen and a False flag
        return df['samples_seen'].max(), False

def plot_recovery_landscape(df, output_path, threshold):
    """
    Generates a scatter plot with a regression curve mapping the data cost 
    of recovery against the magnitude of covariate shift.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Split data into successful recoveries and failures
    recovered = df[df['Recovered'] == True]
    failed = df[df['Recovered'] == False]
    
    # 1. Plot the Successful Recoveries with a trendline
    if not recovered.empty:
        sns.regplot(
            data=recovered, x='Wasserstein', y='Samples_to_Recover', 
            scatter=False, color='#2ca02c', line_kws={'linestyle': '--', 'linewidth': 2}, ax=ax
        )
        sns.scatterplot(
            data=recovered, x='Wasserstein', y='Samples_to_Recover', 
            hue='N_Pool', palette='viridis', s=80, edgecolor='w', alpha=0.8, ax=ax
        )
        
    # 2. Plot the Failures (Models that hit the data ceiling without recovering)
    if not failed.empty:
        sns.scatterplot(
            data=failed, x='Wasserstein', y='Samples_to_Recover', 
            color='red', marker='X', s=100, label='Failed to Recover', ax=ax
        )

    ax.set_title(f'Data Requirements for PLM Recovery (Target F1 $\geq$ {threshold})', fontweight='bold', pad=15)
    ax.set_xlabel('Wasserstein Distance (Evolutionary Drift Magnitude)', fontweight='bold')
    ax.set_ylabel('Target Samples Required for Recovery', fontweight='bold')
    
    # Format Y-axis to show commas for large sample numbers
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    
    # Clean up legend
    handles, labels = ax.get_legend_handles_labels()
    # Filter out the regplot legend artifacts if they exist
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    if unique:
        ax.legend(*zip(*unique), title="Target Pool / Status", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    sns.despine()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"-> Recovery Threshold Plot saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="phylogenetic_gmm")
    parser.add_argument("--threshold", type=float, default=0.65, help="Target F1 score to define 'recovery'")
    args = parser.parse_args()

    base_dir = f"results/{args.data}/experiments/adapt"
    output_dir = f"results/{args.data}/adapt"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Calculating recovery costs (Threshold: F1 >= {args.threshold})...")
    
    data = []
    regex = r'S(\d+)_NP(\d+)_Shf([0-9.]+)_B(\d+)_H(\d+)'
    
    # Iterate through all batch logs
    batch_files = glob.glob(os.path.join(base_dir, "**/*_batch_log.csv"), recursive=True)
    
    for b_file in batch_files:
        filename = os.path.basename(b_file)
        match = re.search(regex, filename)
        if not match: continue
        
        seed = int(match.group(1))
        n_pool = int(match.group(2))
        shift_val = float(match.group(3))
        batch_size = int(match.group(4))
        h_dim = int(match.group(5))
        
        # Find the corresponding summary log to get the Wasserstein distance
        log_file = b_file.replace('_batch_log.csv', '.log').replace('adapted_model', 'adapt_log')
        w_dist = parse_wasserstein_from_log(log_file) if os.path.exists(log_file) else shift_val
        
        # Calculate how many samples it took to hit the F1 threshold
        samples_req, recovered = calculate_recovery_cost(b_file, args.threshold)
        
        data.append({
            "Wasserstein": w_dist, "Shift": shift_val, "N_Pool": f"{n_pool:,}",
            "Batch": batch_size, "Hidden": h_dim, "Seed": seed,
            "Samples_to_Recover": samples_req, "Recovered": recovered
        })

    if data:
        df = pd.DataFrame(data).sort_values(by=["Wasserstein"])
        
        # Print a quick summary to the terminal
        print("\n" + "="*60)
        print(f" RECOVERY SUMMARY (F1 >= {args.threshold})")
        print("="*60)
        print(df[['Wasserstein', 'N_Pool', 'Recovered', 'Samples_to_Recover']].to_string(index=False))
        print("="*60 + "\n")
        
        plot_recovery_landscape(df, os.path.join(output_dir, f"recovery_threshold_F1_{args.threshold}.png"), args.threshold)
    else:
        print("No valid batch logs found to calculate recovery.")

if __name__ == "__main__":
    main()
