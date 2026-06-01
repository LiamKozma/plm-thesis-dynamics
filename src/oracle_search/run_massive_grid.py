import csv
import gc
import multiprocessing as mp
import os
import time
from itertools import product

import numpy as np
from tqdm import tqdm

from generate_simulation import generate_dispersion_gmm
from metrics import calculate_feature_wasserstein


def process_seed_shift_combo(params):
    """
    Worker function: Generates the 'Universe' once per (Seed, Shift, N_Pool, Sigma) combo,
    then iterates through ratios via true random subsampling.
    """
    # Notice we unpack 'sigma' here now!
    seed, shift, n_pool_max, sigma, dim, n_families, n_classes, topology, ratios = params

    max_ratio = max(ratios)
    n_train_max = int(n_pool_max * max_ratio)

    # 1. Generate the MASSIVE Universes ONCE
    X_source_universe, _, _ = generate_dispersion_gmm(
        n_samples=n_train_max, dim=dim, n_families=n_families, n_classes=n_classes,
        hidden_layers=[256, 128], shift_k=shift, seed=seed, is_target=False,
        centroid_spread=10.0, base_sigma=sigma, topology=topology
    )

    X_target_universe, _, _ = generate_dispersion_gmm(
        n_samples=n_pool_max, dim=dim, n_families=n_families, n_classes=n_classes,
        hidden_layers=[256, 128], shift_k=shift, seed=seed, is_target=True,
        centroid_spread=10.0, base_sigma=sigma, topology=topology
    )

    results = []

    # 2. Iterate through ratios by true random SUBSAMPLING
    for r in ratios:
        n_train = int(n_pool_max * r)
        n_pool = n_pool_max

        idx_source = np.random.choice(n_train_max, n_train, replace=False)
        X_source_sub = X_source_universe[idx_source]
        X_target_sub = X_target_universe

        w_dist = calculate_feature_wasserstein(X_source_sub, X_target_sub)

        # Added Base_Sigma to the output dictionary!
        results.append({
            'Seed': seed, 'Shift': shift, 'Base_Sigma': sigma, 'N_Train': n_train,
            'N_Pool': n_pool, 'Ratio': r, 'Log_Ratio': np.log(r),
            'Wasserstein_Distance': w_dist
        })

        del X_source_sub

    del X_source_universe
    del X_target_universe
    gc.collect()

    return results

if __name__ == "__main__":
    seeds = [42, 43, 44]
    shifts = [1.0, 1.5, 2.0, 3.0, 6.0]
    n_pools = [1000000]
    ratios = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0, 1.25, 1.5]
    base_sigmas = [0.5, 1.0, 2.0, 4.0]

    dim = 1280
    n_families = 10000
    n_classes = 1000
    topology = "gaussian"

    # FIXED: Added base_sigmas to the product loop!
    tasks = []
    for s, sh, npool, sig in product(seeds, shifts, n_pools, base_sigmas):
        tasks.append((s, sh, npool, sig, dim, n_families, n_classes, topology, ratios))

    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    num_cores = int(slurm_cpus) if slurm_cpus else min(mp.cpu_count(), 4)

    print(f"Total isolated Universes to generate: {len(tasks)}")
    print(f"Spinning up {num_cores} parallel workers...")

    start_time = time.time()
    csv_filename = "massive_wasserstein_grid.csv"

    # FIXED: Added Base_Sigma to the fields list
    fields = ['Seed', 'Shift', 'Base_Sigma', 'N_Train', 'N_Pool', 'Ratio', 'Log_Ratio', 'Wasserstein_Distance']

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

    with mp.Pool(num_cores) as pool:
        for res_list in tqdm(pool.imap_unordered(process_seed_shift_combo, tasks), total=len(tasks)):
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                for res in res_list:
                    writer.writerow(res)

    print(f"\nCompleted all combinations in {(time.time() - start_time)/60:.1f} minutes.")
