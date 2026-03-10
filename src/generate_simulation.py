import numpy as np
import argparse

def generate_phylogenetic_gmm(n_samples, dim, n_families, shift_magnitude, seed, is_target=False):
    """
    Generates synthetic protein embeddings using a Hierarchical Gaussian Mixture Model.
    Simulates evolutionary motifs branching from a shared ancestral root.
    """
    # 1. Seed for the specific sample generation
    np.random.seed(seed)
    
    # 2. The Universe Topology (Fixed Seed)
    # We use a fixed random state here so the 'Root' and 'Base Families' 
    # are mathematically identical across all your Nextflow runs and source/target splits.
    rng_structure = np.random.RandomState(42) 
    
    # The primordial sequence / core motif
    root_motif = rng_structure.randn(dim)
    
    # Create the base families (The "Central Melodies")
    family_centroids = []
    for k in range(n_families):
        # Families diverge from the root
        mutation = rng_structure.randn(dim) * 2.5 
        centroid = root_motif + mutation
        family_centroids.append(centroid)
        
    family_centroids = np.array(family_centroids)
    
    # 3. Apply Evolutionary Drift (The Distribution Shift)
    # If this is target data, we mutate the centroids further away from the training distribution
    if is_target and shift_magnitude > 0:
        for k in range(n_families):
            drift_direction = np.random.randn(dim)
            drift_direction /= np.linalg.norm(drift_direction)
            # Scale the shift by sqrt(dim) so it doesn't vanish in high-dimensional space
            family_centroids[k] += drift_direction * (shift_magnitude * np.sqrt(dim))
            
    # 4. Generate the Sequence Embeddings (The Leaves)
    X, y = [], []
    
    # Distribute samples evenly across the families
    samples_per_family = n_samples // n_families
    remainders = n_samples % n_families
    
    for k in range(n_families):
        n_k = samples_per_family + (1 if k < remainders else 0)
        
        # Sub-family variation (simulating natural sequence variance around the motif)
        variance = 1.0
        noise = np.random.randn(n_k, dim) * np.sqrt(variance)
        
        # --- THE CONSERVATION MASK ---
        # Pick 30% of the dimensions to be "conserved motifs" for this specific family
        n_conserved = int(dim * 0.30)
        # Use a deterministic seed for the mask so the motif stays the same for Train vs Pool
        mask_rng = np.random.RandomState(42 + k) 
        conserved_indices = mask_rng.choice(dim, n_conserved, replace=False)
        
        # Zero out the noise at the conserved indices (perfectly locking the numbers)
        noise[:, conserved_indices] = 0.0 
        
        family_samples = family_centroids[k] + noise
        
        X.append(family_samples)
        y.extend([k] * n_k)
        
    X = np.vstack(X)
    y = np.array(y)
    
    # Shuffle the dataset
    shuffle_idx = np.random.permutation(n_samples)
    return X[shuffle_idx].astype(np.float32), y[shuffle_idx].astype(np.int64) # y MUST be int64 for CrossEntropyLoss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Global seed for sample reproducibility")
    parser.add_argument("--mode", type=str, choices=['source', 'target'], required=True)
    parser.add_argument("--shift", type=float, default=0.0, help="Magnitude of evolutionary drift")
    
    # Sizing
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_pool", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=1000)
    
    # New GMM/PLM Parameters
    parser.add_argument("--dim", type=int, default=1280, help="Embedding dimension (e.g., 1280 for ESM-2)")
    parser.add_argument("--n_families", type=int, default=10, help="Number of distinct protein families (classes)")
    
    args = parser.parse_args()

    if args.mode == 'source':
        print(f"Generating [SOURCE] data (Train) | Families: {args.n_families} | Dim: {args.dim} | Seed: {args.seed}")
        X, y = generate_phylogenetic_gmm(
            n_samples=args.n_train, dim=args.dim, n_families=args.n_families, 
            shift_magnitude=0.0, seed=args.seed, is_target=False
        )
        np.save(f"source_train_X.npy", X)
        np.save(f"source_train_y.npy", y)

    elif args.mode == 'target':
        total_target_samples = args.n_pool + args.n_test
        print(f"Generating [TARGET] data (Pool+Test) | Shift: {args.shift} | Families: {args.n_families} | Seed: {args.seed}")
        
        X, y = generate_phylogenetic_gmm(
            n_samples=total_target_samples, dim=args.dim, n_families=args.n_families, 
            shift_magnitude=args.shift, seed=args.seed, is_target=True
        )
        
        X_pool, y_pool = X[:args.n_pool], y[:args.n_pool]
        X_test, y_test = X[args.n_pool:], y[args.n_pool:]
        
        np.save("target_pool_X.npy", X_pool)
        np.save("target_pool_y.npy", y_pool)
        np.save("target_test_X.npy", X_test)
        np.save("target_test_y.npy", y_test)
