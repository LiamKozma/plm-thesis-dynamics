import numpy as np
import torch
import torch.nn as nn
import argparse

class RandomOracleNN(nn.Module):
    """
    The Frozen Random Neural Network.
    Acts as the ground-truth labeler to map 1280D space to n_classes.
    """
    def __init__(self, input_dim, num_classes, hidden_layers):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        # Dynamically build the hidden layers
        for h_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim
            
        # Final classification head
        layers.append(nn.Linear(current_dim, num_classes))
        self.net = nn.Sequential(*layers)
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.net(x)  

def calculate_diagnostics(family_assignments, y, n_families, n_classes):
    """Calculates Professor's target metrics for the biological landscape."""
    purities = []
    promiscuous_count = 0
    class_to_families = {c: set() for c in range(n_classes)}
    
    for k in range(n_families):
        idx = (family_assignments == k)
        if not np.any(idx): continue
        labels = y[idx]
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # a) Within-family purity (fraction sharing majority label)
        majority_fraction = counts.max() / len(labels)
        purities.append(majority_fraction)
        
        # c) Family promiscuity (fraction spanning 2+ classes)
        if len(unique_labels) > 1:
            promiscuous_count += 1
            
        # b) Class coverage
        for label in unique_labels:
            class_to_families[label].add(k)
            
    avg_purity = np.mean(purities) * 100
    promiscuity = (promiscuous_count / n_families) * 100
    
    # Filter out unused classes for the coverage average
    active_classes = [len(fams) for fams in class_to_families.values() if len(fams) > 0]
    coverage = np.mean(active_classes) if active_classes else 0
    
    print(f"\n--- Landscape Diagnostics ---")
    print(f"Within-family purity: {avg_purity:.1f}% \t(Target: 50-70%)")
    print(f"Family promiscuity:   {promiscuity:.1f}% \t(Target: 40-60%)")
    print(f"Class coverage:       {coverage:.1f} fams/class \t(Target: ~10)")
    print(f"-----------------------------\n")

def generate_dispersion_gmm(n_samples, dim, n_families, n_classes, hidden_layers, shift_k, seed, is_target=False):
    """
    Generates synthetic protein embeddings using Biased Sampling Covariate Shift
    and a Zipf (Power-Law) distribution for biological family sizes.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. Initialize the Frozen Oracle (deterministic based on seed)
    oracle = RandomOracleNN(dim, n_classes, hidden_layers)
    oracle.eval()

    # 2. Universe Topology 
    rng_structure = np.random.RandomState(42) 
    centroid_spread = 10.0 
    family_centroids = rng_structure.randn(n_families, dim) * centroid_spread
    
    # 3. Dispersion Logic (The Shift Mechanism)
    base_sigma = 2.0 
    if not is_target:
        current_sigma = base_sigma / max(1.0, shift_k) 
    else:
        current_sigma = base_sigma

    # 4. Biological Update: Generate Sequence Embeddings via Power-Law
    X, family_assignments = [], []
    
    # Create the Zipf probability distribution
    ranks = np.arange(1, n_families + 1)
    zipf_exponent = 1.5 # Standard skew for protein databases
    zipf_probs = 1.0 / (ranks ** zipf_exponent)
    zipf_probs /= zipf_probs.sum() # Normalize so they sum exactly to 1.0
    
    # Allocate the exact n_samples across families based on Zipf probabilities
    family_counts = np.random.multinomial(n_samples, zipf_probs)
    
    for k in range(n_families):
        n_k = family_counts[k]
        
        # The "Long Tail": If a family gets 0 samples, skip it
        if n_k == 0: 
            continue 
        
        # Sample using the calculated dispersion
        noise = np.random.randn(n_k, dim) * current_sigma
        family_samples = family_centroids[k] + noise
        
        X.append(family_samples)
        family_assignments.extend([k] * n_k)
        
    X = np.vstack(X).astype(np.float32)
    family_assignments = np.array(family_assignments)
    
    # 5. Label Assignment via Frozen NN
    with torch.no_grad():
        X_tensor = torch.tensor(X)
        logits = oracle(X_tensor)
        y = torch.argmax(logits, dim=1).numpy()
        
    # Shuffle the dataset
    shuffle_idx = np.random.permutation(n_samples)
    return X[shuffle_idx], y[shuffle_idx].astype(np.int64), family_assignments[shuffle_idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, choices=['source', 'target'], required=True)
    parser.add_argument("--shift", type=float, default=1.0, help="Shift k multiplier (k=1 is no shift, k>1 increases target dispersion relative to source)")
    
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_pool", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=1000)
    
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--n_families", type=int, default=1000)
    parser.add_argument("--n_classes", type=int, default=100) 
    parser.add_argument("--oracle_layers", type=str, default="256,128", help="Comma-separated hidden layer sizes")
    
    args = parser.parse_args()
    hidden_layer_sizes = [int(x) for x in args.oracle_layers.split(',')] if args.oracle_layers else []

    if args.mode == 'source':
        print(f"Generating [SOURCE] data | Shift k: {args.shift} | Families: {args.n_families} | Classes: {args.n_classes} | Seed: {args.seed}")
        X, y, fams = generate_dispersion_gmm(
            n_samples=args.n_train, dim=args.dim, n_families=args.n_families, 
            n_classes=args.n_classes, hidden_layers=hidden_layer_sizes, shift_k=args.shift, seed=args.seed, is_target=False
        )
        
        # Run Diagnostics on the Source generation
        calculate_diagnostics(fams, y, args.n_families, args.n_classes)
        
        np.save(f"source_train_X.npy", X)
        np.save(f"source_train_y.npy", y)

    elif args.mode == 'target':
        total_target_samples = args.n_pool + args.n_test
        print(f"Generating [TARGET] data | Shift k: {args.shift} | Families: {args.n_families} | Classes: {args.n_classes} | Seed: {args.seed}")
        
        X, y, fams = generate_dispersion_gmm(
            n_samples=total_target_samples, dim=args.dim, n_families=args.n_families, 
            n_classes=args.n_classes, hidden_layers=hidden_layer_sizes, shift_k=args.shift, seed=args.seed, is_target=True
        )
        
        X_pool, y_pool = X[:args.n_pool], y[:args.n_pool]
        X_test, y_test = X[args.n_pool:], y[args.n_pool:]
        
        np.save("target_pool_X.npy", X_pool)
        np.save("target_pool_y.npy", y_pool)
        np.save("target_test_X.npy", X_test)
        np.save("target_test_y.npy", y_test)
