import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster import hierarchy
import argparse
import os

def load_data(data_dir, n_train, n_pool, shift, seed):
    """Loads the specific source and target arrays based on the Nextflow naming convention."""
    source_x = np.load(os.path.join(data_dir, f"source/source_X_{n_train}_s{seed}.npy"))
    source_y = np.load(os.path.join(data_dir, f"source/source_y_{n_train}_s{seed}.npy"))
    
    target_x = np.load(os.path.join(data_dir, f"target/tgt_pool_X_{n_pool}_{shift}_s{seed}.npy"))
    target_y = np.load(os.path.join(data_dir, f"target/tgt_pool_y_{n_pool}_{shift}_s{seed}.npy"))
    
    return source_x, source_y, target_x, target_y

def plot_phylogenetic_tree(source_x, source_y, output_path):
    """Calculates class centroids and plots the hierarchical branching structure."""
    n_classes = len(np.unique(source_y))
    centroids = np.array([source_x[source_y == k].mean(axis=0) for k in range(n_classes)])
    
    plt.figure(figsize=(10, 6))
    sns.set_style("white")
    
    # Calculate pairwise distances and linkage
    Z = hierarchy.linkage(centroids, method='ward', metric='euclidean')
    
    hierarchy.dendrogram(Z, labels=[f"Family {k}" for k in range(n_classes)], leaf_rotation=45)
    plt.title("Recovered Phylogenetic Tree (Source Data Centroids)", fontsize=14, fontweight='bold')
    plt.ylabel("Euclidean Distance in 1280D Space", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Saved Phylogenetic Tree to {output_path}")

def plot_latent_space(source_x, source_y, target_x, target_y, output_path, method='tsne'):
    """Projects 1280D data into 2D to visualize the clusters and the distribution shift."""
    # Subsample if the dataset is massive to speed up t-SNE
    max_samples = 2000
    if len(source_x) > max_samples:
        idx = np.random.choice(len(source_x), max_samples, replace=False)
        source_x, source_y = source_x[idx], source_y[idx]
    if len(target_x) > max_samples:
        idx = np.random.choice(len(target_x), max_samples, replace=False)
        target_x, target_y = target_x[idx], target_y[idx]

    # Combine data for a unified 2D projection
    combined_x = np.vstack((source_x, target_x))
    domains = np.array(['Source (Train)'] * len(source_x) + ['Target (Shifted)'] * len(target_x))
    combined_classes = np.concatenate((source_y, target_y))

    print(f"Running {method.upper()} on 1280D data. This might take a few seconds...")
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        # t-SNE is much better at preserving local neighborhood clusters (the families)
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        
    embedding = reducer.fit_transform(combined_x)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Use different markers for Source vs Target, and colors for Families
    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1],
        hue=combined_classes, style=domains,
        palette="tab20", alpha=0.7, s=60, edgecolor=None
    )
    
    plt.title(f"2D Latent Space Projection ({method.upper()}) | Visualizing Evolutionary Drift", fontsize=14, fontweight='bold')
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Protein Families & Domains")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Saved Latent Space Projection to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="results/phylogenetic_gmm/data")
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--n_pool", type=int, default=100)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results/phylogenetic_gmm/visualizations")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading data from {args.data_dir}...")
    try:
        source_x, source_y, target_x, target_y = load_data(
            args.data_dir, args.n_train, args.n_pool, args.shift, args.seed
        )
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. Did you run the Nextflow pipeline first?\n{e}")
        exit(1)

    # 1. Plot the Tree
    tree_path = os.path.join(args.out_dir, f"tree_s{args.seed}.png")
    plot_phylogenetic_tree(source_x, source_y, tree_path)
    
    # 2. Plot the 2D Latent Space (t-SNE)
    tsne_path = os.path.join(args.out_dir, f"latent_space_tsne_shift{args.shift}_s{args.seed}.png")
    plot_latent_space(source_x, source_y, target_x, target_y, tsne_path, method='tsne')
