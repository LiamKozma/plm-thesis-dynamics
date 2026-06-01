import numpy as np
import pandas as pd


def profile_dataset(csv_path):
    print(f"--- Loading data from {csv_path} ---\n")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: File not found. Please check the path.")
        return

    # 1. Basic Shape and Memory
    print("### 1. DATASET OVERVIEW ###")
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")

    # 2. Data Types and Missing Values
    print("### 2. COLUMN PROFILING ###")
    profile_data = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = df[col].isna().sum()
        unique_vals = df[col].nunique()
        
        # Check for inf/-inf in numeric columns
        infs = 0
        if pd.api.types.is_numeric_dtype(df[col]):
            infs = np.isinf(df[col]).sum()
            
        profile_data.append({
            'Column': col,
            'Type': dtype,
            'Missing': missing,
            'Infs': infs,
            'Unique Values': unique_vals
        })
    
    profile_df = pd.DataFrame(profile_data)
    print(profile_df.to_string(index=False))
    print("\n")

    # 3. Experimental Design Check (Cardinality)
    print("### 3. EXPERIMENTAL DESIGN BALANCE ###")
    # Assuming these are your structural columns based on compile_results.py
    expected_hyperparams = ['optimizer', 'n_pool', 'batch_size', 'hidden_dim', 'seed']
    actual_hyperparams = [col for col in expected_hyperparams if col in df.columns]
    
    if actual_hyperparams:
        for hp in actual_hyperparams:
            print(f"Value counts for {hp}:")
            # Show top 5 to avoid spamming the console
            print(df[hp].value_counts().head().to_dict())
        
        # Check if we have a fully crossed design
        total_combinations = len(df.groupby(actual_hyperparams))
        print(f"\nTotal unique hyperparameter combinations (excluding shift/Wasserstein): {total_combinations}")
    print("\n")

    # 4. A Sneak Peek at the Metrics
    print("### 4. TARGET METRIC DISTRIBUTIONS ###")
    metrics_to_check = ['test_f1', 'test_ce']
    # Add whatever you named your Wasserstein column
    possible_wasserstein_names = ['wasserstein', 'wasserstein_dist', 'w2_dist']
    for name in possible_wasserstein_names:
        if name in df.columns:
            metrics_to_check.append(name)
            break

    available_metrics = [m for m in metrics_to_check if m in df.columns]
    if available_metrics:
        print(df[available_metrics].describe().round(4).to_string())
    
    print("\n--- Profiling Complete ---")

if __name__ == "__main__":
    # Update this path to point to your new CSV
    FILE_PATH = "/work/ah2lab/LiamK/plm-thesis-dynamics/results/master_adaptation_results_with_W.csv"
    profile_dataset(FILE_PATH)
