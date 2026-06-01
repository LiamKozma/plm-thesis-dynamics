import os
import glob
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import re

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

def parse_adapt_log(filepath):
    last_ce, last_f1, w_dist = None, None, None
    with open(filepath, 'r') as f:
        for line in f:
            if "Wasserstein" in line:
                w_dist = float(line.split('|')[1].strip())
            elif "Final Test CE" in line:
                last_ce = float(line.split('|')[1].strip())
            elif "Final Test F1" in line:
                last_f1 = float(line.split('|')[1].strip())
    return last_ce, last_f1, w_dist

def generate_drift_crash_plot(df, output_path):
    """
    Publication-ready Boxplot + Stripplot showing how predictive power 
    degrades across discrete magnitudes of evolutionary drift.
    """
    if df.empty: return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Map Shift to string for categorical plotting
    df['Shift_Cat'] = 'Shift ' + df['Shift'].astype(str)
    
    # Draw boxplots to show quartiles/variance
    sns.boxplot(
        data=df, x='Shift_Cat', y='Macro F1', hue='N_Pool', 
        palette='Set2', boxprops={'alpha': 0.7}, ax=ax
    )
    # Overlay actual data points to show distribution density
    sns.stripplot(
        data=df, x='Shift_Cat', y='Macro F1', hue='N_Pool', 
        palette='dark:k', dodge=True, alpha=0.5, size=5, ax=ax
    )
    
    # Formatting
    ax.set_title('Impact of Evolutionary Drift on PLM Predictive Power', fontweight='bold', pad=15)
    ax.set_xlabel('Covariate Shift Magnitude', fontweight='bold')
    ax.set_ylabel('Validation Macro F1 Score', fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    # Clean up legend (remove duplicate stripplot entries)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], [f"Pool: {l}" for l in labels[:2]], title='Target Data Availability', loc='lower left')
    
    sns.despine()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"-> Generated Drift Crash Plot: {output_path}")
    plt.close()

def generate_normalized_recovery_curves(batch_df, output_dir):
    """
    Plots adaptation trajectories normalized by Samples Seen.
    Facets by Pool Size and visualizes the variance across batch sizes.
    """
    if batch_df.empty: return

    for n_pool, pool_df in batch_df.groupby('N_Pool'):
        # --- F1 Recovery Threshold Plot ---
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use a high-contrast palette so extreme shifts are highly visible
        palette = sns.color_palette("rocket", n_colors=pool_df['Shift'].nunique())
        
        sns.lineplot(
            data=pool_df, 
            x='samples_seen', 
            y='test_f1', 
            hue='Shift', 
            style='batch_size', # Distinguish batch size impacts by line style
            palette=palette,
            linewidth=2.5,
            errorbar='sd',
            ax=ax
        )
        
        ax.set_title(f'Adaptation Trajectory (Target Pool Size: {n_pool:,})', fontweight='bold', pad=15)
        ax.set_xlabel('Number of Target Samples Seen', fontweight='bold')
        ax.set_ylabel('Validation Macro F1 Score', fontweight='bold')
        ax.set_ylim(0, 1.05)
        
        # Format X-axis with commas
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
        
        plt.legend(title='Shift & Batch Size', bbox_to_anchor=(1.05, 1), loc='upper left')
        sns.despine()
        
        out_path = os.path.join(output_dir, f"recovery_trajectory_NP{n_pool}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"-> Generated Normalized Recovery Curve: {out_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="phylogenetic_gmm")
    parser.add_argument("--mode", type=str, required=True, choices=['adapt'])
    args = parser.parse_args()

    base_dir = f"results/{args.data}/experiments/{args.mode}"
    output_dir = f"results/{args.data}/{args.mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Parsing logs from: {base_dir}...\n")
    
    # 1. Parse Summary Metrics (The End-State)
    data = []
    log_files = glob.glob(os.path.join(base_dir, "**/*.log"), recursive=True)
    
    # Updated Regex to capture the swept hyperparameters
    regex = r'S(\d+)_NP(\d+)_Shf([0-9.]+)_B(\d+)_H(\d+)'

    for log in log_files:
        filename = os.path.basename(log)
        match = re.search(regex, filename)
        if not match: continue
            
        seed = int(match.group(1))
        n_pool = int(match.group(2))
        shift_val = float(match.group(3))
        batch_size = int(match.group(4))
        h_dim = int(match.group(5))

        ce, f1, w_dist = parse_adapt_log(log)
            
        if ce is not None and f1 is not None:
            data.append({
                "Wasserstein": w_dist if w_dist is not None else shift_val,
                "Shift": shift_val, "N_Pool": n_pool, "Batch": batch_size, "Hidden": h_dim,
                "Seed": seed, "CE Loss": ce, "Macro F1": f1
            })

    if data:
        df = pd.DataFrame(data).sort_values(by=["Shift", "N_Pool"])
        generate_drift_crash_plot(df, os.path.join(output_dir, "drift_crash_distribution.png"))
    
    # 2. Parse Batch Trajectories (The Journey)
    batch_data = []
    batch_files = glob.glob(os.path.join(base_dir, "**/*_batch_log.csv"), recursive=True)
    
    for b_file in batch_files:
        filename = os.path.basename(b_file)
        match = re.search(regex, filename)
        if not match: continue
        
        df_b = pd.read_csv(b_file)
        df_b['Seed'] = int(match.group(1))
        df_b['N_Pool'] = int(match.group(2))
        df_b['Shift'] = float(match.group(3))
        df_b['batch_size'] = int(match.group(4))
        df_b['h_dim'] = int(match.group(5))
        batch_data.append(df_b)
        
    if batch_data:
        batch_df = pd.concat(batch_data, ignore_index=True)
        generate_normalized_recovery_curves(batch_df, output_dir)

if __name__ == "__main__":
    main()
