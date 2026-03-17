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
    with open(filepath, 'r') as f:
        for line in f:
            if "Wasserstein" in line:
                return float(line.split('|')[1].strip())
    return None

def extract_f1_scores(batch_csv_path):
    df = pd.read_csv(batch_csv_path)
    # Batch 0 is the initial zero-shot state
    initial_f1 = df[df['batch_number'] == 0]['test_f1'].values[0]
    # The last row is the final adapted state
    final_f1 = df.iloc[-1]['test_f1']
    return initial_f1, final_f1

def plot_degradation_dumbbell(df, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add a tiny bit of horizontal jitter so overlapping runs are visible
    np.random.seed(42)
    df['W_Jitter'] = df['Wasserstein'] + np.random.normal(0, 0.015, size=len(df))
    
    # Draw the connecting lines showing the drop
    for _, row in df.iterrows():
        ax.plot([row['W_Jitter'], row['W_Jitter']], 
                [row['Initial_F1'], row['Final_F1']], 
                color='gray', alpha=0.5, linewidth=1.5, zorder=1)
        
    # Plot the Initial (Zero-Shot) points
    sns.scatterplot(data=df, x='W_Jitter', y='Initial_F1', 
                    color='#1f77b4', s=70, label='Pre-Adaptation (Zero-Shot)', zorder=2, ax=ax)
    
    # Plot the Final points
    sns.scatterplot(data=df, x='W_Jitter', y='Final_F1', 
                    color='#d62728', marker='v', s=80, label='Post-Adaptation (Degraded)', zorder=3, ax=ax)

    ax.set_title('Catastrophic Forgetting: Pre vs. Post Adaptation F1 Scores', fontweight='bold', pad=15)
    ax.set_xlabel('Wasserstein Distance (Evolutionary Drift Magnitude)', fontweight='bold')
    ax.set_ylabel('Validation Macro F1 Score', fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    # Clean legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title='Model State', loc='lower left')
    
    sns.despine()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"-> Saved Degradation Plot to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="phylogenetic_gmm")
    args = parser.parse_args()

    base_dir = f"results/{args.data}/experiments/adapt"
    output_dir = f"results/{args.data}/adapt"
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    regex = r'S(\d+)_NP(\d+)_Shf([0-9.]+)_B(\d+)_H(\d+)'
    batch_files = glob.glob(os.path.join(base_dir, "**/*_batch_log.csv"), recursive=True)
    
    for b_file in batch_files:
        filename = os.path.basename(b_file)
        match = re.search(regex, filename)
        if not match: continue
        
        log_file = b_file.replace('_batch_log.csv', '.log').replace('adapted_model', 'adapt_log')
        w_dist = parse_wasserstein_from_log(log_file)
        if w_dist is None: continue
            
        initial_f1, final_f1 = extract_f1_scores(b_file)
        
        data.append({
            "Wasserstein": w_dist, 
            "Initial_F1": initial_f1, 
            "Final_F1": final_f1
        })

    if data:
        df = pd.DataFrame(data)
        plot_degradation_dumbbell(df, os.path.join(output_dir, "adaptation_degradation_drop.png"))

if __name__ == "__main__":
    main()
