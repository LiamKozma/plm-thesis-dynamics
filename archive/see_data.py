import os
import pandas as pd


def dump_raw_data():
    csv_path = "/work/ah2lab/LiamK/plm-thesis-dynamics/results/master_adaptation_results_with_W.csv"

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Isolate the final performance metrics for each unique run
    idx = df.groupby(
        ["optimizer", "shift", "seed", "n_pool", "batch_size", "hidden_dim"]
    )["samples_seen"].idxmax()
    final_df = df.loc[idx].copy().dropna(subset=["wasserstein"])

    # Select only the columns we care about for the plot
    plot_data = final_df[["optimizer", "shift", "wasserstein", "test_f1"]]

    # Sort it so it's easy to read
    plot_data = plot_data.sort_values(by=["optimizer", "shift", "wasserstein"])

    print("--- START COPY BELOW ---")
    print(plot_data.to_csv(index=False))
    print("--- END COPY ABOVE ---")


if __name__ == "__main__":
    dump_raw_data()
