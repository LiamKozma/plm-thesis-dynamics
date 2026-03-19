import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Set style for academic paper
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

def main():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # ---------------------------------------------------------
    # PANEL A: The Immutable Logos (Biological Landscape)
    # ---------------------------------------------------------
    print("Loading data for Panel A...")
    X = np.load('source_train_X.npy')
    y = np.load('source_train_y.npy')
    
    # Subsample for plotting speed and clarity (10,000 points)
    idx = np.random.choice(X.shape[0], 10000, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]
    
    print("Running PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sub)
    
    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_sub, cmap='tab20', alpha=0.6, s=15, edgecolors='none')
    axes[0].set_title('A. The Biological Landscape (Zipfian Topology)')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].text(0.05, 0.95, 'Frozen Oracle Maps $\mathbb{R}^{1280}$ to Classes', 
                 transform=axes[0].transAxes, fontsize=12, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ---------------------------------------------------------
    # PANEL B: The Epistemic Shock (Covariate Shift)
    # ---------------------------------------------------------
    print("Generating Panel B...")
    # Simulate the densities based on your parameters
    source_sigma = 0.5 / 5.0 # k=5.0
    target_sigma = 0.5       # base_sigma
    
    source_dist = np.random.normal(0, source_sigma, 10000)
    target_dist = np.random.normal(0, target_sigma, 10000)
    
    sns.kdeplot(source_dist, ax=axes[1], color='#1f77b4', fill=True, alpha=0.5, label=f'Source Domain ($\sigma={source_sigma}$)')
    sns.kdeplot(target_dist, ax=axes[1], color='#d62728', fill=True, alpha=0.3, label=f'Target Domain ($\sigma={target_sigma}$)')
    
    axes[1].set_title('B. Covariate Shift Mechanism ($k=5.0$)')
    axes[1].set_xlabel('Latent Feature Variance')
    axes[1].set_ylabel('Density')
    axes[1].legend(loc='upper right')

    # ---------------------------------------------------------
    # PANEL C: Metric Decoupling & Recovery (The Student)
    # ---------------------------------------------------------
    print("Generating Panel C...")
    # Generate stylized conceptual data demonstrating the theoretical curve
    batches = np.linspace(0, 100, 500)
    
    # CE Loss: Deceivingly goes down continuously
    ce_loss = 2.5 * np.exp(-0.05 * batches) + 0.5
    
    # F1 Score: Plummets due to collapse, then recovers after n_B
    f1_score = 0.8 - 0.6 * np.exp(-0.1 * batches) + 0.5 / (1 + np.exp(-0.2 * (batches - 60)))
    
    ax3_twin = axes[2].twinx()
    
    l1 = axes[2].plot(batches, ce_loss, color='#2ca02c', linewidth=3, label='Target CE Loss (Decreasing)')
    l2 = ax3_twin.plot(batches, f1_score, color='#9467bd', linewidth=3, linestyle='--', label='Target Macro F1 (Collapsing)')
    
    # Mark the Recovery Threshold (n_B)
    axes[2].axvline(x=60, color='black', linestyle=':', linewidth=2)
    axes[2].text(62, 2.0, 'Recovery Threshold ($n_B$)', fontsize=12, fontweight='bold')
    
    axes[2].set_title('C. Adaptation Dynamics & Metric Decoupling')
    axes[2].set_xlabel('Target Data Volume (Batches)')
    axes[2].set_ylabel('Cross-Entropy Loss', color='#2ca02c')
    ax3_twin.set_ylabel('Macro F1 Score', color='#9467bd')
    
    # Combine legends
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    axes[2].legend(lns, labs, loc='center right')

    # ---------------------------------------------------------
    # Final Adjustments & Save
    # ---------------------------------------------------------
    plt.tight_layout()
    plt.savefig('architecture_figure.png', dpi=300, bbox_inches='tight')
    plt.savefig('architecture_figure.pdf', bbox_inches='tight') # Save as PDF for your LaTeX thesis!
    print("Saved as architecture_figure.png and architecture_figure.pdf")

if __name__ == "__main__":
    main()
