import os
import pandas as pd
import numpy as np

def summarize_experiment_data(csv_path):
    print(f"{'='*60}")
    print(f"📊 DATA SUMMARY: {csv_path}")
    print(f"{'='*60}\n")

    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found at {csv_path}")
        print("Please check the path or ensure your master CSV has been compiled.")
        return

    # Load the data
    df = pd.read_csv(csv_path)

    # 1. Basic Shape and Columns
    print("--- 1. DATASET SHAPE & COLUMNS ---")
    print(f"Total Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}\n")

    # 2. Missing Data Check
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("--- 2. MISSING DATA ALERT ---")
        print(missing[missing > 0])
        print()
    else:
        print("--- 2. MISSING DATA ---")
        print("✅ No missing values detected in the dataset.\n")

    # 3. Sweep Variables (Unique Values)
    print("--- 3. EXPERIMENT SWEEPS (UNIQUE VALUES) ---")
    # Identify likely categorical/sweep columns based on your yaml files
    expected_sweeps = ['optimizer', 'shift', 'n_pool', 'batch_size', 'hidden_dim', 'seed']
    for col in expected_sweeps:
        if col in df.columns:
            unique_vals = sorted(df[col].dropna().unique())
            print(f"{col:<15}: {unique_vals}")
    print()

    # 4. Summary Statistics for Key Metrics
    print("--- 4. METRIC RANGES ---")
    metrics = ['test_f1', 'test_ce', 'train_loss', 'samples_seen']
    for m in metrics:
        if m in df.columns:
            print(f"{m:<15} -> Min: {df[m].min():.4f} | Max: {df[m].max():.4f} | Mean: {df[m].mean():.4f}")
    
    # Check for Wasserstein specifically
    if 'wasserstein' in df.columns or 'w_dist' in df.columns:
        w_col = 'wasserstein' if 'wasserstein' in df.columns else 'w_dist'
        print(f"\n✅ Wasserstein distance column found: '{w_col}'")
        print(f"{w_col:<15} -> Min: {df[w_col].min():.4f} | Max: {df[w_col].max():.4f}")
    else:
        print("\n⚠️ WARNING: 'wasserstein' distance column NOT found in this CSV.")
        print("We will need this for the topological correlation plots!")
    print()

    # 5. Endpoint Performance (Final Batch Snapshot)
    print("--- 5. FINAL PERFORMANCE SNAPSHOT (Per Optimizer & Shift) ---")
    if 'samples_seen' in df.columns and 'optimizer' in df.columns and 'shift' in df.columns:
        # Get the max samples seen for each run to isolate the final state
        idx = df.groupby(['optimizer', 'shift', 'seed', 'n_pool', 'batch_size', 'hidden_dim'])['samples_seen'].idxmax()
        final_df = df.loc[idx]
        
        # Aggregate the final F1 scores
        summary = final_df.groupby(['optimizer', 'shift'])['test_f1'].agg(['mean', 'std', 'count']).round(4)
        print(summary)
    else:
        print("Required columns for endpoint snapshot are missing.")

if __name__ == "__main__":
    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    CSV_FILE = os.path.join(RESULTS_DIR, "master_adaptation_results.csv")
    
    summarize_experiment_data(CSV_FILE)
