import pandas as pd
import numpy as np
import multiprocessing as mp
from itertools import product
from tqdm import tqdm
import time

# Import your existing logic directly from your scripts!
from generate_simulation import generate_dispersion_gmm
from metrics import calculate_feature_wasserstein

def calculate_wasserstein_for_run(params):
    """Worker function to generate the landscape and compute distance in memory."""
    seed, shift, n_train, n_pool, base_sigma, dim, n_families, n_classes, topology = params
    
    # 1. Generate Source Environment (Train)
    X_source, _, _ = generate_dispersion_gmm(
        n_samples=n_train,
        dim=dim,
        n_families=n_families,
        n_classes=n_classes,
        hidden_layers=[256, 128],
        shift_k=shift,       # Shift is applied to the source
        seed=seed,
        is_target=False,
        centroid_spread=10.0,
        base_sigma=base_sigma,
        topology=topology
    )
    
    # 2. Generate Target Environment (Pool)
    X_target, _, _ = generate_dispersion_gmm(
        n_samples=n_pool,
        dim=dim,
        n_families=n_families,
        n_classes=n_classes,
        hidden_layers=[256, 128],
        shift_k=shift,       # generate_dispersion_gmm ignores this when is_target=True
        seed=seed,
        is_target=True,      
        centroid_spread=10.0,
        base_sigma=base_sigma,
        topology=topology
    )
    
    # 3. Compute feature-wise 1D Wasserstein distance
    w_dist = calculate_feature_wasserstein(X_source, X_target)
    
    ratio = n_pool / n_train
    return {
        'Seed': seed,
        'Shift': shift,
        'N_Train': n_train,
        'N_Pool': n_pool,
        'Ratio': ratio,
        'Log_Ratio': np.log(ratio),
        'Wasserstein_Distance': w_dist
    }

if __name__ == "__main__":
    # --- EXPERIMENT GRID ---
    # Define every combination you want to test
    seeds = [42, 43, 44, 45, 46]
    shifts = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
    n_trains = [250000, 500000, 750000, 1000000] 
    n_pools = [50000, 100000, 250000, 500000, 1000000, 1500000]
    
    # Base parameters for the PLM space
    base_sigma = 2.0
    dim = 1280 
    n_families = 1000
    n_classes = 100
    topology = "gaussian"
    
    tasks = [
        (s, sh, nt, npool, base_sigma, dim, n_families, n_classes, topology)
        for s, sh, nt, npool in product(seeds, shifts, n_trains, n_pools)
    ]
    
    print(f"Total environment pairs to generate: {len(tasks)}")
    
    # --- MULTIPROCESSING ---
    # Leaves a little breathing room on the node
    num_cores = min(mp.cpu_count(), 30) 
    print(f"Spinning up {num_cores} parallel workers...")
    
    start_time = time.time()
    results = []
    
    # imap_unordered is slightly faster as it doesn't wait for sequence completion
    with mp.Pool(num_cores) as pool:
        for res in tqdm(pool.imap_unordered(calculate_wasserstein_for_run, tasks), total=len(tasks)):
            results.append(res)
            
    # --- SAVE TO CSV ---
    df = pd.DataFrame(results)
    df.to_csv("massive_wasserstein_grid.csv", index=False)
    
    print(f"\nCompleted {len(tasks)} combinations in {(time.time() - start_time)/60:.1f} minutes.")
