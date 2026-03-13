import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


class RandomOracleNN(nn.Module):
    """
    The Frozen Random Neural Network.
    Acts as the ground-truth labeler to map 1280D space to n_classes.
    """

    def __init__(self, input_dim, num_classes, hidden_layers):
        super().__init__()

        layers = []
        current_dim = input_dim

        # Keep LayerNorm + Kaiming to prevent dimensional collapse
        for h_dim in hidden_layers:
            linear = nn.Linear(current_dim, h_dim, bias=True)
            nn.init.kaiming_normal_(
                linear.weight, mode="fan_in", nonlinearity="relu"
            )
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim

        # THE FIX: Re-center the positive-only ReLU outputs to the origin
        layers.append(nn.LayerNorm(current_dim))

        # Final classification head
        final_layer = nn.Linear(current_dim, num_classes, bias=False)

        # Normalize the 1000 class vectors so no class is "louder"
        with torch.no_grad():
            nn.init.normal_(final_layer.weight)
            final_layer.weight.copy_(
                torch.nn.functional.normalize(final_layer.weight, p=2, dim=1)
            )

        layers.append(final_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def calculate_diagnostics(family_assignments, y, n_families, n_classes):
    """Calculates Professor's target metrics for the biological landscape."""
    purities = []
    promiscuous_count = 0
    class_to_families = {c: set() for c in range(n_classes)}

    for k in range(n_families):
        idx = family_assignments == k
        if not np.any(idx):
            continue
        labels = y[idx]
        unique_labels, counts = np.unique(labels, return_counts=True)

        # a) Within-family purity (fraction sharing majority label)
        majority_fraction = counts.max() / len(labels)
        purities.append(majority_fraction)

        # b) Family promiscuity (fraction spanning 2+ classes)
        if len(unique_labels) > 1:
            promiscuous_count += 1

        # c) Class coverage (Assign family ONLY to its primary/majority class)
        majority_label = unique_labels[np.argmax(counts)]
        class_to_families[majority_label].add(k)

    avg_purity = np.mean(purities) * 100
    promiscuity = (promiscuous_count / n_families) * 100

    # Calculate coverage across ALL 1000 target classes
    all_classes = [len(fams) for fams in class_to_families.values()]
    coverage = np.mean(all_classes) if all_classes else 0

    print(f"\n--- Landscape Diagnostics ---")
    print(f"Within-family purity: {avg_purity:.1f}% \t(Target: 50-70%)")
    print(f"Family promiscuity:   {promiscuity:.1f}% \t(Target: 40-60%)")
    print(f"Class coverage:       {coverage:.1f} fams/class \t(Target: ~10)")
    print(f"-----------------------------\n")


def generate_dispersion_gmm(
    n_samples,
    dim,
    n_families,
    n_classes,
    hidden_layers,
    shift_k,
    seed,
    is_target=False,
    centroid_spread=10.0,
    base_sigma=2.0,
    topology="gaussian",
):
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
    print(f"Generating landscape using topology: {topology.upper()}")

    if topology == "hypercube":
        family_centroids = (
            np.random.uniform(-1.0, 1.0, size=(n_families, dim))
            * centroid_spread
        )

    elif topology == "hypersphere":
        v = np.random.randn(n_families, dim)
        v_norm = np.linalg.norm(v, axis=1, keepdims=True)
        u = v / v_norm
        r = np.random.uniform(0, 1, size=(n_families, 1)) ** (1.0 / dim)
        family_centroids = r * u * centroid_spread

    elif topology == "projection":
        latent_dim = 20
        latent_centroids = np.random.uniform(
            -1.0, 1.0, size=(n_families, latent_dim)
        )
        projection_matrix = np.random.randn(latent_dim, dim)
        family_centroids = np.dot(latent_centroids, projection_matrix) * (
            centroid_spread / np.sqrt(latent_dim)
        )

    else:  # default gaussian
        family_centroids = np.random.randn(n_families, dim) * centroid_spread

    # 3. Dispersion Logic (The Shift Mechanism)
    if not is_target:
        current_sigma = base_sigma / max(1.0, shift_k)
    else:
        current_sigma = base_sigma

    # 4. Memory-Efficient & Fast Allocation
    X_concat = np.zeros((n_samples, dim), dtype=np.float32)

    ranks = np.arange(1, n_families + 1)
    zipf_probs = 1.0 / (ranks**1.5)
    zipf_probs /= zipf_probs.sum()

    family_counts = np.ones(n_families, dtype=int)
    family_counts += np.random.multinomial(n_samples - n_families, zipf_probs)

    # Pre-build a flat array of all family assignments (e.g., [0, 0, ..., 1, 1, ..., 9999])
    family_assignments = np.repeat(np.arange(n_families), family_counts).astype(
        np.int32
    )

    # Optional: np.random.shuffle(family_assignments) # Uncomment if sequence order matters

    print(f"Sampling {n_samples:,} embeddings (Chunked & Vectorized)...")

    batch_size = 50000  # Safe, fast chunk size
    for i in tqdm(range(0, n_samples, batch_size), desc="Generating Chunks"):
        end_idx = min(i + batch_size, n_samples)
        batch_assign = family_assignments[i:end_idx]

        # Fetch centroids for this specific chunk
        batch_centroids = family_centroids[batch_assign]

        # Generate float32 noise directly for this chunk
        noise = np.random.normal(
            0, current_sigma, size=(len(batch_assign), dim)
        ).astype(np.float32)

        X_concat[i:end_idx] = batch_centroids + noise

    # 5. Memory-Safe Batched Label Assignment
    print("Assigning labels via Oracle (Batched)...")
    y = np.zeros(n_samples, dtype=np.int32)
    batch_size = 10000  # 10,000 is a safe sweet spot for RAM

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_concat)

        for i in tqdm(range(0, n_samples, batch_size), desc="Labeling Batches"):
            end_idx = min(i + batch_size, n_samples)

            # Forward pass only on the current chunk
            logits = oracle(X_tensor[i:end_idx])

            # Extract predictions directly into the pre-allocated numpy array
            y[i:end_idx] = torch.argmax(logits, dim=1).numpy()
    return X_concat, y, family_assignments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode", type=str, choices=["source", "target"], required=True
    )
    parser.add_argument(
        "--shift", type=float, default=1.0, help="Shift k multiplier"
    )

    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_pool", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=1000)

    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--n_families", type=int, default=1000)
    parser.add_argument("--n_classes", type=int, default=100)
    parser.add_argument(
        "--oracle_layers",
        type=str,
        default="256,128",
        help="Comma-separated hidden layer sizes",
    )

    parser.add_argument(
        "--centroid_spread",
        type=float,
        default=10.0,
        help="Distance between family centers",
    )
    parser.add_argument(
        "--base_sigma",
        type=float,
        default=2.0,
        help="Variance/spread within a family",
    )
    parser.add_argument(
        "--topology",
        type=str,
        choices=["hypercube", "hypersphere", "projection", "gaussian"],
        default="gaussian",
    )  # <-- ADD THIS

    args = parser.parse_args()
    hidden_layer_sizes = (
        [int(x) for x in args.oracle_layers.split(",")]
        if args.oracle_layers
        else []
    )

    if args.mode == "source":
        print(
            f"Generating [SOURCE] data | Shift k: {args.shift} | Families: {args.n_families} | Classes: {args.n_classes} | Seed: {args.seed}"
        )
        X, y, fams = generate_dispersion_gmm(
            n_samples=args.n_train,
            dim=args.dim,
            n_families=args.n_families,
            n_classes=args.n_classes,
            hidden_layers=hidden_layer_sizes,
            shift_k=args.shift,
            seed=args.seed,
            is_target=False,
            centroid_spread=args.centroid_spread,
            base_sigma=args.base_sigma,
            topology=args.topology,
        )

        # Run Diagnostics on the Source generation
        calculate_diagnostics(fams, y, args.n_families, args.n_classes)

        np.save(f"source_train_X.npy", X)
        np.save(f"source_train_y.npy", y)

    elif args.mode == "target":
        total_target_samples = args.n_pool + args.n_test
        print(
            f"Generating [TARGET] data | Shift k: {args.shift} | Families: {args.n_families} | Classes: {args.n_classes} | Seed: {args.seed}"
        )

        X, y, fams = generate_dispersion_gmm(
            n_samples=total_target_samples,
            dim=args.dim,
            n_families=args.n_families,
            n_classes=args.n_classes,
            hidden_layers=hidden_layer_sizes,
            shift_k=args.shift,
            seed=args.seed,
            is_target=True,
            centroid_spread=args.centroid_spread,
            base_sigma=args.base_sigma,  # ---> 3. PASS TO TARGET CALL HERE <---
        )

        X_pool, y_pool = X[: args.n_pool], y[: args.n_pool]
        X_test, y_test = X[args.n_pool :], y[args.n_pool :]

        np.save("target_pool_X.npy", X_pool)
        np.save("target_pool_y.npy", y_pool)
        np.save("target_test_X.npy", X_test)
        np.save("target_test_y.npy", y_test)
