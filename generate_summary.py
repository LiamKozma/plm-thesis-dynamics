import pandas as pd
import os

def generate_summary_stats(master_csv_path):
    if not os.path.exists(master_csv_path):
        print(f"File not found: {master_csv_path}")
        return

    # Load the data
    df = pd.read_csv(master_csv_path)

    # 1. Overall Pipeline Stats
    total_runs = df[["optimizer", "seed", "n_pool", "shift", "batch_size", "hidden_dim"]].drop_duplicates().shape[0]
    total_evaluations = len(df)

    # 2. Hyperparameters Sweeped
    shifts = sorted(df["shift"].unique())
    n_pools = sorted(df["n_pool"].unique())
    batch_sizes = sorted(df["batch_size"].unique())
    hidden_dims = sorted(df["hidden_dim"].unique())
    seeds = sorted(df["seed"].unique())

    # 3. Print the Report Header
    print("=" * 70)
    print("         PLM ADAPTATION EXPERIMENT: HEAD-TO-HEAD REPORT")
    print("=" * 70)
    print(f"Total Successful Runs             : {total_runs} (Expected 144)")
    print(f"Total Model Checkpoints Evaluated : {total_evaluations:,}")
    print("-" * 70)
    print("HYPERPARAMETER GRID SCOPE:")
    print(f"  Seeds                : {seeds} (n={len(seeds)})")
    print(f"  Target Pool Sizes    : {n_pools}")
    print(f"  Distribution Shifts  : {shifts}")
    print(f"  Batch Sizes          : {batch_sizes}")
    print(f"  Hidden Dimensions    : {hidden_dims}")
    print("-" * 70)

    # 4. Performance Stats Head-to-Head
    print("BIOLOGICAL LANDSCAPE METRICS (ADAM vs. ADAMW):")
    
    for opt in df['optimizer'].unique():
        opt_df = df[df['optimizer'] == opt]
        
        # Baseline (Samples seen == 0)
        baseline_df = opt_df[opt_df["samples_seen"] == 0]
        mean_baseline_f1 = baseline_df["test_f1"].mean()
        std_baseline_f1 = baseline_df["test_f1"].std()
        
        # Max Achieved During Adaptation
        adapt_df = opt_df[opt_df["samples_seen"] > 0]
        run_max_f1 = adapt_df.groupby(["seed", "n_pool", "shift", "batch_size", "hidden_dim"])["test_f1"].max().reset_index()
        mean_max_f1 = run_max_f1["test_f1"].mean()
        
        print(f"\n  Optimizer: {opt}")
        print(f"    Zero-Shot Baseline F1      : {mean_baseline_f1:.4f} ± {std_baseline_f1:.4f}")
        print(f"    Mean Max F1 (Adaptation)   : {mean_max_f1:.4f}")
        print(f"    Average F1 Gain            : +{(mean_max_f1 - mean_baseline_f1):.4f}")

    print("-" * 70)
    print("  * Note: Drop indicates textbook catastrophic forgetting.")
    print("=" * 70)

if __name__ == "__main__":
    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    FILE_PATH = os.path.join(RESULTS_DIR, "master_adaptation_results.csv")
    generate_summary_stats(FILE_PATH)
