import pandas as pd

def verify_dataset(file_path):
    print("="*70)
    print("DATASET VERIFICATION AND STATISTICAL SUMMARY REPORT")
    print("="*70)
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}'. Did the compile script finish?")
        return
    
    # 1. Basic Dimensions
    print(f"\n[1. OVERALL DIMENSIONS]")
    print(f"Total Rows (Timesteps/Observations): {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Columns present: {', '.join(df.columns)}")
    
    # 2. Experimental Design (Factors & Levels)
    print(f"\n[2. EXPERIMENTAL DESIGN (INDEPENDENT VARIABLES)]")
    print("This shows the exact 'grid' of your parameter sweep:")
    factors = ['Experiment', 'Seed', 'Pool_Size', 'Shift', 'Batch_Size', 'Hidden_Dim']
    
    for factor in factors:
        if factor in df.columns:
            unique_vals = sorted(list(df[factor].dropna().unique()))
            print(f"  - {factor:<15}: {len(unique_vals)} distinct levels -> {unique_vals}")
            
    # Calculate expected vs actual unique runs
    if all(f in df.columns for f in factors):
        unique_runs = df[factors].drop_duplicates()
        print(f"\n  => Total Unique Experimental Runs (Combinations): {len(unique_runs)}")
    
    # 3. Missing Data Check
    print(f"\n[3. MISSING DATA CHECK]")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  Status: PERFECTLY CLEAN. No missing values (NaNs) found in any column.")
    else:
        print("  Warning: Missing values detected! Your statistician needs to know this:")
        print(missing[missing > 0].to_string())
        
    # 4. Dependent Variables Summary
    print(f"\n[4. METRICS SUMMARY (DEPENDENT VARIABLES)]")
    metrics = ['samples_seen', 'train_loss', 'test_ce', 'test_f1', 'Wasserstein_Distance']
    metrics = [m for m in metrics if m in df.columns] # Only grab columns that actually exist
    
    if metrics:
        stats_df = df[metrics].describe().T[['min', 'mean', 'max', 'std']]
        # Format numbers to 4 decimal places for clean reading
        for col in stats_df.columns:
            stats_df[col] = stats_df[col].apply(lambda x: f"{x:,.4f}" if pd.notnull(x) else "NaN")
        
        print("               Min          Mean         Max          Std Dev")
        print("-" * 70)
        for index, row in stats_df.iterrows():
            print(f"{index:<15} {row['min']:<12} {row['mean']:<12} {row['max']:<12} {row['std']:<12}")
            
    # 5. Longitudinal Integrity (Time-Series Checks)
    print(f"\n[5. TIME-SERIES INTEGRITY]")
    if all(f in df.columns for f in factors):
        # Count how many rows (timesteps) belong to each run
        timesteps_per_run = df.groupby(factors).size()
        print(f"  Minimum timesteps in a single run: {timesteps_per_run.min()}")
        print(f"  Maximum timesteps in a single run: {timesteps_per_run.max()}")
        print(f"  Average timesteps per run:         {timesteps_per_run.mean():.1f}")
        
        if timesteps_per_run.min() == timesteps_per_run.max():
            print("  Note: All runs have the EXACT SAME number of timesteps. The dataset is perfectly balanced temporally.")
        else:
            print("  Note: Runs have VARYING numbers of timesteps. (Likely due to different Pool Sizes changing the total batch count).")

    print("\n" + "="*70)
    print("End of Report. Ready for statistical analysis.")
    print("="*70)

if __name__ == "__main__":
    verify_dataset("plm_timeseries_dataset.csv")
