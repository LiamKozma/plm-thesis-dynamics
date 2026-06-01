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
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
plt.rcParams.update({
    'font.family': 'serif',
    'figure.autolayout': True
})

def extract_f1_degradation(batch_csv_path):
    """Calculates how much the F1 score dropped from zero-shot to final adaptation."""
    df = pd.read_csv(batch_csv_path)
    initial_f1 = df[df['batch_number'] == 0]['test_f1'].values[0]
    final_f1 = df.iloc[-1]['test_f1']
    # Positive value means the model lost predictive power (Forgetting)
    return initial_f1 - final_f1

def generate_main_effects_plot(df, output_path):
    """
    Creates a faceted pointplot showing the marginal effect of each hyperparameter 
    on the severity of catastrophic forgetting.
    """
    # 1. Melt the dataframe to isolate the hyperparameters
    df_melt = df.melt(
        id_vars=['F1_Degradation'], 
        value_vars=['Covariate Shift', 'Target Pool Size', 'Batch Size', 'MLP Hidden Dim'],
        var_name='Hyperparameter', 
        value_name='Setting'
    )
    
    # Ensure categorical sorting makes sense
    df_melt['Setting'] = df_melt['Setting'].astype(str)

    # 2. Setup the Facet Grid
    g = sns.FacetGrid(
        df_melt, 
        col="Hyperparameter", 
        col_wrap=4, 
        sharex=False, 
        sharey=True, 
        height=5, 
        aspect=0.8
    )
    
    # 3. Map a pointplot to show the Mean and 95% Confidence Interval
    g.map_dataframe(
        sns.pointplot, 
        x="Setting", 
        y="F1_Degradation", 
        color="#d62728",
        capsize=.1,
        markers="D",
        linestyles="--"
    )

    # 4. Formatting
    g.set_axis_labels("", "Drop in Macro F1 (Forgetting)")
    g.set_titles(col_template="{col_name}", fontweight='bold')
    
    # Add a global title
    plt.subplots_adjust(top=0.85)
    g.figure.suptitle('Main Effects of Hyperparameters on Catastrophic Forgetting', fontsize=16, fontweight='bold')
    
    # Add a baseline reference line at 0 (No degradation)
    for ax in g.axes.flat:
        ax.axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
        ax.tick_params(axis='x', rotation=45)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"-> Saved Main Effects Plot to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="phylogenetic_gmm")
    args = parser.parse_args()

    base_dir = f"results/{args.data}/experiments/adapt"
    output_dir = f"results/{args.data}/adapt"
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    # Regex designed to capture your current sweep naming convention
    regex = r'S(\d+)_NP(\d+)_Shf([0-9.]+)_B(\d+)_H(\d+)'
    batch_files = glob.glob(os.path.join(base_dir, "**/*_batch_log.csv"), recursive=True)
    
    for b_file in batch_files:
        filename = os.path.basename(b_file)
        match = re.search(regex, filename)
        if not match: continue
        
        n_pool = int(match.group(2))
        shift_val = float(match.group(3))
        batch_size = int(match.group(4))
        h_dim = int(match.group(5))
            
        f1_drop = extract_f1_degradation(b_file)
        
        data.append({
            "Covariate Shift": f"Shift {shift_val}", 
            "Target Pool Size": f"{n_pool:,}",
            "Batch Size": f"{batch_size}",
            "MLP Hidden Dim": f"{h_dim}",
            "F1_Degradation": f1_drop
        })

    if data:
        df = pd.DataFrame(data)
        generate_main_effects_plot(df, os.path.join(output_dir, "hyperparameter_main_effects.png"))
    else:
        print("No valid batch logs found.")

if __name__ == "__main__":
    main()
