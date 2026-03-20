import os
import pandas as pd
import numpy as np


def run_diagnostics(csv_path):
    print(f"Loading empirical data from {csv_path}...\n")
    df = pd.read_csv(csv_path)

    # Isolate asymptotic performance
    idx = df.groupby(
        ["optimizer", "shift", "seed", "n_pool", "batch_size", "hidden_dim"]
    )["samples_seen"].idxmax()
    final_df = df.loc[idx].copy().dropna(subset=["wasserstein"])

    # Calculate Diagnostics
    diagnostics = (
        final_df.groupby(["optimizer", "shift"])
        .agg(
            Count=("wasserstein", "size"),
            W_Mean=("wasserstein", "mean"),
            W_Std=("wasserstein", "std"),
            W_Min=("wasserstein", "min"),
            W_Max=("wasserstein", "max"),
            F1_Mean=("test_f1", "mean"),
            F1_Std=("test_f1", "std"),
        )
        .reset_index()
    )

    print("=" * 90)
    print(
        f"{'Optimizer':<18} | {'Shift':<6} | {'Count':<5} | {'W-Dist (Mean ± Std)':<22} | {'F1 Score (Mean ± Std)'}"
    )
    print("=" * 90)

    for _, row in diagnostics.iterrows():
        w_str = f"{row['W_Mean']:.3f} ± {row['W_Std']:.3f}"
        f1_str = f"{row['F1_Mean']:.3f} ± {row['F1_Std']:.3f}"
        print(
            f"{row['optimizer']:<18} | {row['shift']:<6.1f} | {row['Count']:<5} | {w_str:<22} | {f1_str}"
        )
    print("=" * 90)


if __name__ == "__main__":
    CSV_FILE = "/work/ah2lab/LiamK/plm-thesis-dynamics/results/master_adaptation_results_with_W.csv"
    if os.path.exists(CSV_FILE):
        run_diagnostics(CSV_FILE)
    else:
        print("CSV not found. Check the path.")
