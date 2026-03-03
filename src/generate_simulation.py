import numpy as np
import argparse

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- OPTION A: (The Manifold Hypothesis) ---
def generate_manifold_data(n_samples=1000, dim=1280, manifold_dim=50, shift_magnitude=0.0, seed=42):
    """
    Generates data on a linear subspace with a 'Fitness Peak' landscape.
    """
    np.random.seed(seed)
    
    # 1. Define Landscape Structure (Weights) FIRST
    weights_linear = np.random.randn(manifold_dim)
    
    # 2. Create Manifold
    basis_matrix = np.random.randn(dim, manifold_dim)
    q, _ = np.linalg.qr(basis_matrix)
    
    # 3. Latent space z
    z = np.random.randn(n_samples, manifold_dim)
    
    # 4. Apply Shift
    if shift_magnitude > 0:
        shift_vector = np.ones(manifold_dim) * shift_magnitude
        z = z + shift_vector
        
    X = np.dot(z, q.T)
    
    # 5. Fitness Function: "The Quadratic Peak"
    y_linear = np.dot(z, weights_linear)
    y_quad = -0.5 * np.mean(z**2, axis=1) * (manifold_dim / 5.0)
    
    y_raw = y_linear + 0.1 * y_quad + 2.0
    y = sigmoid(y_raw)
    
    return X.astype(np.float32), y.astype(np.float32)

# --- OPTION B: (High-Dim Statistics) ---
def generate_highdim_data(n_samples=1000, dim=1280, shift_magnitude=0.0, seed=42):
    """
    Generates unstructured data with a NON-UNIFORM shift to test the Wasserstein metric.
    """
    np.random.seed(seed)
    
    # Weights First
    w_linear = np.random.randn(dim) / np.sqrt(dim) 
    
    X = np.random.randn(n_samples, dim)
    
    if shift_magnitude > 0:
        direction = np.random.rand(dim) 
        direction = direction / np.linalg.norm(direction)
        shift_vector = direction * (shift_magnitude * np.sqrt(dim))
        X = X + shift_vector

    # High-Dim Fitness
    y_linear = np.dot(X, w_linear)
    y_quad = -0.5 * np.mean(X**2, axis=1)
    
    y_raw = y_linear + y_quad + 2.0
    y = sigmoid(y_raw)
    
    return X.astype(np.float32), y.astype(np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=['manifold', 'highdim'], default='manifold')
    parser.add_argument("--shift", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42, help="Global seed for reproducibility")
    
    # New arguments for data sizing
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_pool", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=1000)
    
    # Replaces explicit file outputs with a mode switch
    parser.add_argument("--mode", type=str, choices=['source', 'target'], required=True, 
                        help="'source' generates Shift 0 Train data. 'target' generates Shift X Pool/Test data.")
    args = parser.parse_args()

    if args.mode == 'source':
        print(f"Generating [SOURCE] data (Train) with Seed={args.seed}...")
        
        # Hardcode shift to 0.0 to guarantee baseline distribution
        if args.type == 'manifold':
            X, y = generate_manifold_data(n_samples=args.n_train, shift_magnitude=0.0, seed=args.seed)
        else:
            X, y = generate_highdim_data(n_samples=args.n_train, shift_magnitude=0.0, seed=args.seed)
        
        # Output source files
        np.save("source_train_X.npy", X)
        np.save("source_train_y.npy", y)
        print(f"--> Saved source_train_X.npy and source_train_y.npy ({args.n_train} samples)")

    elif args.mode == 'target':
        print(f"Generating [TARGET] data (Pool + Test) with Shift={args.shift}, Seed={args.seed}...")
        
        # Combine pool and test for a single massive dataset generation
        total_target_samples = args.n_pool + args.n_test
        
        if args.type == 'manifold':
            X, y = generate_manifold_data(n_samples=total_target_samples, shift_magnitude=args.shift, seed=args.seed)
        else:
            X, y = generate_highdim_data(n_samples=total_target_samples, shift_magnitude=args.shift, seed=args.seed)
        
        # Slicing guarantees they are disjoint but from the exact same mathematical distribution
        X_pool, y_pool = X[:args.n_pool], y[:args.n_pool]
        X_test, y_test = X[args.n_pool:], y[args.n_pool:]
        
        # Output target files
        np.save("target_pool_X.npy", X_pool)
        np.save("target_pool_y.npy", y_pool)
        np.save("target_test_X.npy", X_test)
        np.save("target_test_y.npy", y_test)
        print(f"--> Saved target_pool ({args.n_pool} samples) and target_test ({args.n_test} samples)")
