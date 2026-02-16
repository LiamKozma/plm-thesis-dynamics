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
    # We do this before z generation so weights are identical across all shifts
    weights_linear = np.random.randn(manifold_dim)
    
    # 2. Create Manifold
    basis_matrix = np.random.randn(dim, manifold_dim)
    q, _ = np.linalg.qr(basis_matrix)
    
    # 3. Latent space z
    z = np.random.randn(n_samples, manifold_dim)
    
    # 4. Apply Shift
    if shift_magnitude > 0:
        # Move directly away from the fitness peak
        shift_vector = np.ones(manifold_dim) * shift_magnitude
        z = z + shift_vector
        
    X = np.dot(z, q.T)
    
    # 5. Fitness Function: "The Quadratic Peak"
    # Source (z~0) is at the peak. Shifting away causes fitness to drop.
    # y = Linear - Distance^2
    
    y_linear = np.dot(z, weights_linear)
    y_quad = -0.5 * np.mean(z**2, axis=1) * (manifold_dim / 5.0)
    
    # Combine and Squash to [0, 1]
    # We add a bias (+10) to center the source distribution in the 'active' region of the sigmoid
    y_raw = y_linear + 0.1 * y_quad + 2.0
    
    # CRITICAL FIX: Use Sigmoid instead of MinMax
    # This ensures that if the population shifts to a 'dead' zone, 
    # the labels naturally become 0.0, rather than being forced to 1.0.
    y = sigmoid(y_raw)
    
    return X.astype(np.float32), y.astype(np.float32)

# --- OPTION B: (High-Dim Statistics) ---
def generate_highdim_data(n_samples=1000, dim=1280, shift_magnitude=0.0, seed=42):
    """
    Generates unstructured data with the same Sigmoid logic for fair comparison.
    """
    np.random.seed(seed)
    
    # Weights First
    w_linear = np.random.randn(dim) / np.sqrt(dim) # Scale weights for high dim
    
    X = np.random.randn(n_samples, dim)
    
    if shift_magnitude > 0:
        direction = np.ones(dim) 
        direction = direction / np.linalg.norm(direction)
        shift_vector = direction * (shift_magnitude * np.sqrt(dim))
        X = X + shift_vector

    # High-Dim Fitness
    y_linear = np.dot(X, w_linear)
    y_quad = -0.5 * np.mean(X**2, axis=1) # Quadratic penalty
    
    # Same Sigmoid Logic
    y_raw = y_linear + y_quad + 2.0
    y = sigmoid(y_raw)
    
    return X.astype(np.float32), y.astype(np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=['manifold', 'highdim'], default='manifold')
    parser.add_argument("--shift", type=float, default=0.0)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--output_x", type=str, required=True)
    parser.add_argument("--output_y", type=str, required=True)
    args = parser.parse_args()

    if args.type == 'manifold':
        print(f"Generating [MANIFOLD] data with Shift={args.shift}...")
        X, y = generate_manifold_data(n_samples=args.n_samples, shift_magnitude=args.shift)
    else:
        print(f"Generating [HIGH-DIM] data with Shift={args.shift}...")
        X, y = generate_highdim_data(n_samples=args.n_samples, shift_magnitude=args.shift)
    
    np.save(args.output_x, X)
    np.save(args.output_y, y)
