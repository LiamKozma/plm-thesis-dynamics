import pandas as pd
import os


def generate_summary_stats(master_csv_path):
    if not os.path.exists(master_csv_path):
        print(f"File not found: {master_csv_path}")
        return

    # Load the data
    df = pd.read_csv(master_csv_path)

    # 1. Overall Pipeline Stats
    total_runs = (
        df[["seed", "n_pool", "shift", "batch_size", "hidden_dim"]]
        .drop_duplicates()
        .shape[0]
    )
    total_evaluations = len(df)

    # 2. Hyperparameters Sweeped
    shifts = sorted(df["shift"].unique())
    n_pools = sorted(df["n_pool"].unique())
    batch_sizes = sorted(df["batch_size"].unique())
    hidden_dims = sorted(df["hidden_dim"].unique())
    seeds = sorted(df["seed"].unique())

    # 3. Performance Stats (Baseline vs Max Achieved)
    baseline_df = df[df["samples_seen"] == 0]
    mean_baseline_f1 = baseline_df["test_f1"].mean()
    std_baseline_f1 = baseline_df["test_f1"].std()

    adapt_df = df[df["samples_seen"] > 0]
    run_max_f1 = (
        adapt_df.groupby(
            ["seed", "n_pool", "shift", "batch_size", "hidden_dim"]
        )["test_f1"]
        .max()
        .reset_index()
    )
    mean_max_f1 = run_max_f1["test_f1"].mean()

    # 4. Wasserstein Distances per Shift
    if "wasserstein_distance" in df.columns:
        w_stats = (
            df[["shift", "wasserstein_distance"]]
            .drop_duplicates()
            .groupby("shift")
            .agg(["mean", "std"])
        )
        w_stats.columns = ["W-Dist Mean", "W-Dist Std"]

        # Format for clean printing
        w_stats["W-Dist Mean"] = w_stats["W-Dist Mean"].apply(
            lambda x: f"{x:.4f}"
        )
        w_stats["W-Dist Std"] = w_stats["W-Dist Std"].apply(
            lambda x: f"± {x:.4f}"
        )
        w_str = w_stats.to_string()
    else:
        w_str = "  [!] Wasserstein distances not found in master CSV."

    # 5. Print the Report
    print("=" * 65)
    print("          PLM ADAPTATION EXPERIMENT: SUMMARY REPORT")
    print("=" * 65)
    print(f"Total Successful Runs             : {total_runs}")
    print(f"Total Model Checkpoints Evaluated : {total_evaluations:,}")
    print("-" * 65)
    print("HYPERPARAMETER GRID SCOPE:")
    print(f"  Seeds               : {seeds} (n={len(seeds)})")
    print(f"  Target Pool Sizes   : {n_pools}")
    print(f"  Distribution Shifts : {shifts}")
    print(f"  Batch Sizes         : {batch_sizes}")
    print(f"  Hidden Dimensions   : {hidden_dims}")
    print("-" * 65)
    print("BIOLOGICAL LANDSCAPE METRICS:")
    print(
        f"  Zero-Shot Baseline F1             : {mean_baseline_f1:.4f} ± {std_baseline_f1:.4f}"
    )
    print(f"  Mean Maximum F1 During Adaptation : {mean_max_f1:.4f}")
    print("  * Note: Drop indicates textbook catastrophic forgetting.")
    print("-" * 65)
    print("WASSERSTEIN DISTANCE BY SHIFT SEVERITY:")
    print(w_str)
    print("=" * 65)

    # Optionally save this to a text file for your records
    summary_file = os.path.join(
        os.path.dirname(master_csv_path), "experiment_summary_report.txt"
    )
    with open(summary_file, "w") as f:
        f.write("PLM ADAPTATION EXPERIMENT SUMMARY\n")
        f.write(f"Total Runs: {total_runs}\n")
        f.write(f"Baseline F1: {mean_baseline_f1:.4f}\n")
        f.write(f"Wasserstein Distances:\n{w_str}\n")
    print(f"\nSaved condensed log to {summary_file}")


if __name__ == "__main__":
    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    FILE_PATH = os.path.join(RESULTS_DIR, "master_adaptation_results.csv")
    generate_summary_stats(FILE_PATH)
