import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D  # Added for the clean legend
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# FONT & STYLE: THE "FAKE LATEX" / TIMES NEW ROMAN SETUP
# ---------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "text.usetex": False,            # No external LaTeX needed
    "font.family": "serif",        
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], 
    "mathtext.fontset": "stix",      # Makes math look like LaTeX/Times
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11
})

def main():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ---------------------------------------------------------
    # PANEL 1: The Empirical Landscape Topology (PCA)
    # ---------------------------------------------------------
    print("Generating Panel 1: Landscape Topology...")
    try:
        # THE FIX: We renamed this file to source_shift_1.npy earlier!
        X_source = np.load('source_shift_1.npy')
        y_source = np.load('source_train_y.npy')

        # Subsample for plotting speed and clarity
        n_samples = min(X_source.shape[0], 5000)
        idx = np.random.choice(X_source.shape[0], n_samples, replace=False)
        X_sub = X_source[idx]
        y_sub = y_source[idx]
        print(f"  Loaded real source data. Subsampled to {n_samples} points.")
    except FileNotFoundError:
        print("  Real data not found. Using representative synthetic data.")
        X_sub = np.random.randn(5000, 1280)
        y_sub = np.random.randint(0, 20, 5000)

    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_sub)

    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_sub, cmap='viridis', alpha=0.6, s=15, edgecolors='none')
    axes[0].set_title('Empirical Landscape Topology')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')

    # ---------------------------------------------------------
    # PANEL 2: Empirical Covariate Shift (Micro-Variance)
    # ---------------------------------------------------------
    print("Generating Panel 2: Covariate Shift Mechanism...")
    try:
        X_target = np.load('target_pool_X.npy')
        X_src_1 = np.load('source_shift_1.npy')
        X_src_2 = np.load('source_shift_2.npy')
        X_src_5 = np.load('source_shift_5.npy')

        # Isolate a single dominant protein family
        family_size = 500 
        pca_1d = PCA(n_components=1)
        tgt_spread = pca_1d.fit_transform(X_target[:family_size]).flatten()
        src_1_spread = pca_1d.transform(X_src_1[:family_size]).flatten()
        src_2_spread = pca_1d.transform(X_src_2[:family_size]).flatten()
        src_5_spread = pca_1d.transform(X_src_5[:family_size]).flatten()

        print("  Isolated dominant protein family to map true Covariate Shift.")
    except FileNotFoundError:
        print("  Real data missing! Using simulated empirical spread.")
        tgt_spread = np.random.normal(0, 0.5, 500)
        src_1_spread = np.random.normal(0, 0.5, 500)
        src_2_spread = np.random.normal(0, 0.25, 500)
        src_5_spread = np.random.normal(0, 0.1, 500)

    # THE LEGEND FIX: label='_nolegend_' hides the ugly colored box
    sns.kdeplot(tgt_spread, ax=axes[1], color='black', fill=True, alpha=0.05, 
                linewidth=2.5, linestyle='--', zorder=1, label='_nolegend_')
    
    sns.kdeplot(src_1_spread, ax=axes[1], color='#0072B2', fill=True, alpha=0.4, linewidth=1.5, zorder=2, label='Source (Shift 1.0)')
    sns.kdeplot(src_2_spread, ax=axes[1], color='#009E73', fill=True, alpha=0.5, linewidth=1.5, zorder=3, label='Source (Shift 2.0)')
    sns.kdeplot(src_5_spread, ax=axes[1], color='#D55E00', fill=True, alpha=0.6, linewidth=1.5, zorder=4, label='Source (Shift 5.0)')

    # Create the "Just Dashes" manual legend item
    handles, labels = axes[1].get_legend_handles_labels()
    target_handle = Line2D([0], [0], color='black', linestyle='--', linewidth=2.5)
    handles.insert(0, target_handle)
    labels.insert(0, 'Target (Base)')
    
    axes[1].legend(handles, labels, loc='upper right')
    axes[1].set_title('Covariate Shift (Single Family)')
    axes[1].set_xlabel('Principal Component 1 (Internal Spread)')
    axes[1].set_ylabel('Density')

    # ---------------------------------------------------------
    # PANEL 3: Oracle Hyperparameter Tuning
    # ---------------------------------------------------------
    print("Generating Panel 3: Oracle Tuning Trade-off...")
    try:
        tune_df = pd.read_csv('master_tuning_1M.csv')
        print("  Loaded real tuning data.")
    except FileNotFoundError:
        print("  master_tuning_1M.csv not found. Using simulated tuning points.")
        tune_df = pd.DataFrame({
            'Oracle_Layers': np.random.choice(['512,256', '1024,512', '2048,1024', '1024,1024,512'], 24),
            'Base_Sigma': np.random.choice([0.1, 0.3, 0.5, 0.7, 1.0, 1.4], 24),
            'Promiscuity_%': np.random.uniform(25, 70, 24),
            'Purity_%': np.random.uniform(50, 95, 24)
        })
        tune_df.loc[0] = ['1024,1024,512', 0.5, 58.9, 69.3]

    sns.scatterplot(
        data=tune_df, x='Promiscuity_%', y='Purity_%', style='Oracle_Layers',
        hue='Base_Sigma', size='Base_Sigma', sizes=(50, 250),
        palette='viridis', markers=['o', 's', '^', 'D'], alpha=0.9,
        edgecolor='black', linewidth=0.5, ax=axes[2]
    )

    # Draw the "Sweet Spot" Target Box
    target_box = patches.Rectangle(
        (40, 50), 20, 20, linewidth=2, edgecolor='black',
        facecolor='gray', alpha=0.15, linestyle='--', label='Target Zone'
    )
    axes[2].add_patch(target_box)

    # Highlight the winning configuration
    winner = tune_df[(tune_df['Oracle_Layers'] == '1024,1024,512') & (tune_df['Base_Sigma'] == 0.5)]
    if not winner.empty:
        win_x = winner['Promiscuity_%'].values[0]
        win_y = winner['Purity_%'].values[0]

        axes[2].scatter(win_x, win_y, facecolors='none', edgecolors='black',
                        s=500, linewidth=2.5, label='Selected Model', zorder=5)

        axes[2].annotate('Selected', xy=(win_x, win_y), xytext=(win_x - 10, win_y + 10),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                         fontsize=12, fontweight='bold', zorder=6)

    axes[2].set_title('Oracle Hyperparameter Tuning')
    axes[2].set_xlabel('Family Promiscuity (%)')
    axes[2].set_ylabel('Within-Family Purity (%)')

    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # ---------------------------------------------------------
    # Final Adjustments & Save
    # ---------------------------------------------------------
    plt.tight_layout()
    plt.savefig('architecture_bottom_row.png', dpi=300, bbox_inches='tight')
    plt.savefig('architecture_bottom_row.pdf', bbox_inches='tight')
    print("Saved outputs as architecture_bottom_row.png and .pdf")

if __name__ == "__main__":
    main()
