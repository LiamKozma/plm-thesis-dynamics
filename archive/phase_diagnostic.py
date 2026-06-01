import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def extract_phase_data(csv_file):
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # FILTER: Only keep Experiment 1 and 2
    df = df[df['Experiment'].isin(['phylogenetic_gmm_exp1', 'phylogenetic_gmm_exp2'])]
    
    group_cols = ['Experiment', 'Seed', 'Pool_Size', 'Shift', 'Wasserstein_Distance']
    extracted = []
    
    for name, group in df.groupby(group_cols):
        group = group.sort_values('samples_seen')
        
        baseline_data = group[group['samples_seen'] == 0]
        if baseline_data.empty: continue
        baseline_f1 = baseline_data['test_f1'].iloc[0]
        
        adaptation_data = group[group['samples_seen'] > 0]
        
        # Did it EVER recover? (True/False) -> 1/0
        recovered = (adaptation_data['test_f1'] > baseline_f1).any()
        
        record = {
            'Optimizer': 'Standard Adam' if name[0] == 'phylogenetic_gmm_exp1' else 'AdamW',
            'Pool_Size': name[2],
            'Shift': name[3],
            'Wasserstein': name[4],
            'Recovered': int(recovered)
        }
        extracted.append(record)
        
    return pd.DataFrame(extracted)

def run_phase_diagnostics(df_t):
    # N_trains for Exp 1 & 2 was fixed at 1,000,000 based on your YAMLs
    N_TRAIN = 1000000 
    
    # Aggregate the data
    agg_df = df_t.groupby(['Optimizer', 'Shift', 'Pool_Size']).agg(
        Total_Runs=('Recovered', 'count'),
        Success_Rate=('Recovered', 'mean'),
        Mean_Wasserstein=('Wasserstein', 'mean')
    ).reset_index()
    
    # Calculate the crucial ratio you hypothesized
    agg_df['Train_to_Pool_Ratio'] = N_TRAIN / agg_df['Pool_Size']
    agg_df['Success_Rate_%'] = agg_df['Success_Rate'] * 100

    print("\n" + "="*80)
    print("PHASE TRANSITION DIAGNOSTIC: SHIFT vs. POOL SIZE (PASTE TO LLM)")
    print("="*80)
    print(agg_df.to_string(index=False, float_format="%.4f"))
    print("="*80 + "\n")
    
    return agg_df

def plot_phase_heatmap(agg_df):
    sns.set_theme(style="white", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    optimizers = ['Standard Adam', 'AdamW']
    
    for i, opt in enumerate(optimizers):
        # Filter for the specific optimizer
        subset = agg_df[agg_df['Optimizer'] == opt]
        
        # Pivot the data to create a grid: Rows=Shift, Cols=Pool_Size, Values=Success_Rate
        pivot_table = subset.pivot(index='Shift', columns='Pool_Size', values='Success_Rate')
        
        # Sort index descending so highest shift is at the top
        pivot_table = pivot_table.sort_index(ascending=False)
        
        # Plot the heatmap
        ax = axes[i]
        sns.heatmap(pivot_table, annot=True, fmt=".0%", cmap="mako", 
                    vmin=0, vmax=1, cbar=(i==1), ax=ax, linewidths=2, linecolor='white')
        
        ax.set_title(f"{opt}: Recovery Phase Space", fontweight='bold', pad=15)
        ax.set_ylabel("Theoretical Shift Parameter", fontweight='bold')
        ax.set_xlabel("Available Data Pool (n_pools)", fontweight='bold')

    plt.suptitle("The Data Starvation Boundary: Shift Severity vs. Data Availability", 
                 fontsize=18, fontweight='black', y=1.05)
    
    plt.tight_layout()
    plt.savefig('phase_transition_heatmap.png', dpi=300, bbox_inches='tight')
    print("Masterpiece Heatmap saved as 'phase_transition_heatmap.png'")

if __name__ == "__main__":
    csv_path = 'plm_timeseries_exp3.csv'
    df_extracted = extract_phase_data(csv_path)
    agg_data = run_phase_diagnostics(df_extracted)
    plot_phase_heatmap(agg_data)
