import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def load_data(data_dir, n_train, seed):
    x_path = os.path.join(data_dir, f"source/source_X_{n_train}_s{seed}.npy")
    y_path = os.path.join(data_dir, f"source/source_y_{n_train}_s{seed}.npy")
    return np.load(x_path), np.load(y_path)

def get_relative_families(X, y, anchor_family):
    """Calculates true 1280D distances to find the closest and furthest families."""
    n_families = len(np.unique(y))
    # Calculate the exact centroid for every family
    centroids = np.array([X[y == k].mean(axis=0) for k in range(n_families)])
    
    anchor_centroid = centroids[anchor_family]
    
    # Calculate Euclidean distances from the anchor to all other centroids
    distances = np.linalg.norm(centroids - anchor_centroid, axis=1)
    
    # Mask the anchor itself with NaN so it isn't chosen as the "closest"
    distances[anchor_family] = np.nan
    
    closest_family = np.nanargmin(distances)
    furthest_family = np.nanargmax(distances)
    
    return anchor_family, closest_family, furthest_family

def plot_targeted_heatmap(X, y, target_families, labels, output_path, features_to_show=100):
    """
    Plots a heatmap specifically ordered by: Anchor -> Closest -> Furthest.
    """
    plt.figure(figsize=(12, 10))
    
    # Extract and order the data exactly how we want it
    X_ordered = []
    y_boundaries = []
    current_idx = 0
    
    for family in target_families:
        mask = (y == family)
        family_data = X[mask]
        X_ordered.append(family_data)
        
        current_idx += len(family_data)
        y_boundaries.append(current_idx)
        
    X_plot = np.vstack(X_ordered)[:, :features_to_show]
    
    # Plot the heatmap
    ax = sns.heatmap(
        X_plot, 
        cmap="viridis", 
        cbar_kws={'label': 'Embedding Value'},
        yticklabels=False
    )
    
    # Draw horizontal lines to separate the families
    for boundary in y_boundaries[:-1]:
        ax.axhline(boundary, color='white', linewidth=3)
        
    # Annotate the Y-axis with our custom narrative labels
    y_ticks = [0] + y_boundaries
    y_centers = [(y_ticks[i] + y_ticks[i+1])/2 for i in range(len(y_ticks)-1)]
    ax.set_yticks(y_centers)
    ax.set_yticklabels(labels, rotation=0, fontweight='bold', fontsize=12)

    plt.title(f"Motif Conservation Matrix (First {features_to_show} Dimensions)\nTracking Evolutionary Drift from Anchor Family", fontsize=15, fontweight='bold')
    plt.xlabel("Embedding Dimensions (Sequence Positions)", fontsize=12)
    plt.ylabel("Protein Sequences", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"-> Saved Targeted Motif Heatmap to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="results/phylogenetic_gmm/data")
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results/phylogenetic_gmm/visualizations")
    
    # You can now specify which family to anchor the story around
    parser.add_argument("--anchor", type=int, default=1, help="The base family to compare against")
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Loading data...")
    X, y = load_data(args.data_dir, args.n_train, args.seed)
    
    # 1. Mathematically find the relatives
    anchor, closest, furthest = get_relative_families(X, y, args.anchor)
    print(f"Anchor Family: {anchor}")
    print(f"Closest Relative: {closest}")
    print(f"Furthest Outgroup: {furthest}")
    
    # 2. Plot them in order
    target_families = [anchor, closest, furthest]
    labels = [
        f"Anchor (Fam {anchor})", 
        f"Closest Relative (Fam {closest})\nShould share many motifs", 
        f"Furthest Homolog (Fam {furthest})\nShould be mostly noise/mutations"
    ]
    
    out_path = os.path.join(args.out_dir, f"motif_heatmap_anchor_{anchor}.png")
    plot_targeted_heatmap(X, y, target_families, labels, out_path, features_to_show=100)
