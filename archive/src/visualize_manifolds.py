import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
import os

def load_data(data_dir, n_train, n_pool, shift, seed):
    """Loads the specific source and target arrays based on the Nextflow naming convention."""
    source_x = np.load(os.path.join(data_dir, f"source/source_X_{n_train}_s{seed}.npy"))
    source_y = np.load(os.path.join(data_dir, f"source/source_y_{n_train}_s{seed}.npy"))
    
    target_x = np.load(os.path.join(data_dir, f"target/tgt_pool_X_{n_pool}_{shift}_s{seed}.npy"))
    target_y = np.load(os.path.join(data_dir, f"target/tgt_pool_y_{n_pool}_{shift}_s{seed}.npy"))
    
    return source_x, source_y, target_x, target_y

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
    
    # We create clear, descriptive labels for our domains
    domains = np.array(['Source (Narrow/Biased)'] * len(source_x) + ['Target (Broad/True Pool)'] * len(target_x))

    print(f"Running {method.upper()} on 1280D data. This might take a few seconds...")
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        
    embedding = reducer.fit_transform(combined_x)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    # THE FIX: Color entirely by Domain. 
    # Use smaller point size (s=30), add alpha for transparency, and a white edge to distinguish overlapping points.
    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1],
        hue=domains, 
        palette={"Source (Narrow/Biased)": "#1f77b4", "Target (Broad/True Pool)": "#ff7f0e"},
        alpha=0.6, s=30, edgecolor="w", linewidth=0.5
    )
    
    plt.title(f"2D Latent Space Projection ({method.upper()})\nVisualizing Dispersion-Based Covariate Shift", fontsize=15, fontweight='bold')
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    
    # Move the legend cleanly inside the plot space, no more squishing!
    plt.legend(title="Data Distribution", loc='best', frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Saved Latent Space Projection to {output_path}")

def plot_class_distribution(target_y, output_path):
    """Plots the rank-abundance of functional classes to show biological imbalance."""
    classes, counts = np.unique(target_y, return_counts=True)
    
    # Sort by abundance (Standard bioinformatics rank-abundance curve)
    sorted_indices = np.argsort(-counts)
    sorted_classes = classes[sorted_indices]
    sorted_counts = counts[sorted_indices]
    
    plt.figure(figsize=(12, 6))
    # Using a bar plot to clearly show the long-tail distribution
    sns.barplot(x=[str(c) for c in sorted_classes], y=sorted_counts, palette="viridis")
    
    plt.title("Functional Class Rank-Abundance (Target Pool)\nDemonstrating Label Imbalance from the Frozen Oracle", fontsize=14, fontweight='bold')
    plt.xlabel("Functional Class ID (Sorted by Abundance)", fontsize=12)
    plt.ylabel("Number of Sequences", fontsize=12)
    
    # Gracefully handle the x-axis labels if there are a ton of classes
    if len(classes) > 40:
        plt.xticks(rotation=90, fontsize=8)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(40))
    else:
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Saved Class Distribution Plot to {output_path}")

def plot_latent_space_by_class(target_x, target_y, output_path, method='tsne'):
    """Projects data into 2D, colored strictly by Functional Class to map the landscape."""
    max_samples = 2000
    if len(target_x) > max_samples:
        idx = np.random.choice(len(target_x), max_samples, replace=False)
        target_x, target_y = target_x[idx], target_y[idx]

    print(f"Running {method.upper()} to map the Functional Landscape...")
    reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding = reducer.fit_transform(target_x)
    
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    # Use a highly distinct categorical palette for different classes
    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1],
        hue=target_y, 
        palette="Spectral", 
        alpha=0.8, s=40, edgecolor="w", linewidth=0.5
    )
    
    plt.title(f"2D Functional Landscape ({method.upper()})\nTarget Pool Colored by Frozen NN Class Assignment", fontsize=15, fontweight='bold')
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    
    # Smart Legend: Hide it if there are 100 classes so it doesn't crush the plot
    if len(np.unique(target_y)) <= 20:
        plt.legend(title="Functional Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend([],[], frameon=False) 
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"-> Saved Class Landscape Projection to {output_path}")

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

    # 1. Plot the Covariate Shift (Domain coloring)
    tsne_domain_path = os.path.join(args.out_dir, f"latent_space_domain_shift{args.shift}_s{args.seed}.png")
    plot_latent_space(source_x, source_y, target_x, target_y, tsne_domain_path, method='tsne')

    # 2. Plot the Class Rank-Abundance Distribution
    class_dist_path = os.path.join(args.out_dir, f"class_distribution_s{args.seed}.png")
    plot_class_distribution(target_y, class_dist_path)

    # 3. Plot the Functional Landscape (Class coloring)
    tsne_class_path = os.path.join(args.out_dir, f"latent_space_classes_s{args.seed}.png")
    plot_latent_space_by_class(target_x, target_y, tsne_class_path, method='tsne')
