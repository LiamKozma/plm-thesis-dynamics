import glob
import os
import re

import pandas as pd


def extract_time_series_data(base_dir, experiment_name):
    """Scrapes logs and full CSVs for a given experiment directory."""
    target_dir = os.path.join(base_dir, experiment_name, 'experiments', 'adapt')
    log_pattern = os.path.join(target_dir, 'adapt_log_*.log')
    log_files = glob.glob(log_pattern)
    
    filename_regex = re.compile(r'adapt_log_S(\d+)_NP(\d+)_Shf([0-9.]+)_B(\d+)_H(\d+)\.log')
    
    # 1. Deduplication Phase: Find the most recent files before reading data
    # Dictionary format: { (params_tuple): (log_file_path, csv_file_path, mod_time) }
    runs_dict = {}
    
    for log_file in log_files:
        match = filename_regex.search(log_file)
        if not match:
            continue
            
        seed, npool, shift, batch, hdim = match.groups()
        csv_name = f"adapted_model_S{seed}_NP{npool}_Shf{shift}_B{batch}_H{hdim}_batch_log.csv"
        csv_path = os.path.join(target_dir, csv_name)
        
        if os.path.exists(csv_path):
            mod_time = os.path.getmtime(log_file)
            params_key = (seed, npool, shift, batch, hdim)
            
            # If this parameter combo is new, or if this file is newer than the stored one, update it.
            if params_key not in runs_dict or mod_time > runs_dict[params_key][2]:
                runs_dict[params_key] = (log_file, csv_path, mod_time)
                
    # 2. Extraction Phase: Read the full CSVs of only the latest runs
    compiled_data = []
    
    for params_key, (log_file, csv_path, mod_time) in runs_dict.items():
        seed, npool, shift, batch, hdim = params_key
        
        # Scrape Wasserstein from the .log file
        wasserstein_val = None
        with open(log_file, 'r') as f:
            w_match = re.search(r'Wasserstein\s+\|\s+([0-9.]+)', f.read())
            if w_match:
                wasserstein_val = float(w_match.group(1))
                
        # Read the ENTIRE CSV to capture the time-series progression
        try:
            df_csv = pd.read_csv(csv_path)
            if df_csv.empty:
                continue
                
            # Broadcast the metadata across all rows for this specific run
            df_csv['Experiment'] = experiment_name
            df_csv['Seed'] = int(seed)
            df_csv['Pool_Size'] = int(npool)
            df_csv['Shift'] = float(shift)
            df_csv['Batch_Size'] = int(batch)
            df_csv['Hidden_Dim'] = int(hdim)
            df_csv['Wasserstein_Distance'] = wasserstein_val
            
            compiled_data.append(df_csv)
            
        except pd.errors.EmptyDataError:
            print(f"Warning: Empty CSV found at {csv_path}")
            
    if compiled_data:
        return pd.concat(compiled_data, ignore_index=True)
    return pd.DataFrame()

if __name__ == "__main__":
    print("Scanning directories and compiling time-series results...")
    
    df_exp1 = extract_time_series_data('results', 'phylogenetic_gmm_exp1')
    df_exp2 = extract_time_series_data('results', 'phylogenetic_gmm_exp2')
    
    df_master = pd.concat([df_exp1, df_exp2], ignore_index=True)
    
    if df_master.empty:
        print("Error: No data successfully extracted. Check your file paths.")
        exit()
        
    # --- Health Check & Validation ---
    # We now count UNIQUE RUNS instead of total rows, since each run has many rows.
    expected_per_exp = 3 * 2 * 3 * 2 * 2 # Seeds * Pools * Shifts * Batches * HiddenDims
    total_expected_runs = expected_per_exp * 2 
    
    # Identify unique runs by their hyperparameters
    unique_runs = df_master[['Experiment', 'Seed', 'Pool_Size', 'Shift', 'Batch_Size', 'Hidden_Dim']].drop_duplicates()
    actual_run_count = len(unique_runs)
    total_rows = len(df_master)
    
    print("\n" + "="*50)
    print("DATA EXTRACTION REPORT")
    print("="*50)
    print(f"Total Timesteps (Rows) Extracted: {total_rows:,}")
    print(f"Expected Unique Runs: {total_expected_runs} ({expected_per_exp} per experiment)")
    print(f"Actual Unique Runs:   {actual_run_count}")
    
    if actual_run_count == total_expected_runs:
        print("Status: PERFECT MATCH. All jobs successfully accounted for.")
    elif actual_run_count < total_expected_runs:
        print(f"Status: MISSING {total_expected_runs - actual_run_count} RUNS. Some SLURM jobs may have failed or timed out.")
    else:
        print(f"Status: OVERCOUNT by {actual_run_count - total_expected_runs} RUNS. Check for unexpected parameter combinations.")
        
    # Rearrange columns so metadata is first, metrics are last for readability
    cols = ['Experiment', 'Seed', 'Pool_Size', 'Shift', 'Batch_Size', 'Hidden_Dim', 
            'Wasserstein_Distance', 'batch_number', 'samples_seen', 'train_loss', 'test_ce', 'test_f1']
    
    # Include current_lr if it exists (from exp2 / adamw)
    if 'current_lr' in df_master.columns:
        cols.append('current_lr')
        
    df_master = df_master[cols]
    
    output_filename = "plm_timeseries_dataset.csv"
    df_master.to_csv(output_filename, index=False)
    print(f"\nSuccess! Final dataset saved to '{output_filename}'.")
