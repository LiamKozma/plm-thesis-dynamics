import glob
import os
import re

import pandas as pd


def extract_time_series_data(base_dir, experiment_name):
    """Scrapes logs and full CSVs for a given experiment directory."""
    target_dir = os.path.join(base_dir, experiment_name, 'experiments', 'adapt')
    log_pattern = os.path.join(target_dir, 'adapt_log_*.log')
    log_files = glob.glob(log_pattern)

    # -------------------------------------------------------------------------
    # THE FIX: Added '(?:_N(\d+))?' to optionally capture the new train_size tag
    # -------------------------------------------------------------------------
    filename_regex = re.compile(r'adapt_log_S(\d+)(?:_N(\d+))?_NP(\d+)_Shf([0-9.]+)_B(\d+)_H(\d+)\.log')

    # 1. Deduplication Phase
    runs_dict = {}

    for log_file in log_files:
        match = filename_regex.search(log_file)
        if not match:
            continue

        # Extract the matched groups
        seed = match.group(1)
        ntrain = match.group(2) # This will be None for Exp 3 if it lacks the tag
        npool = match.group(3)
        shift = match.group(4)
        batch = match.group(5)
        hdim = match.group(6)

        # Reconstruct the expected CSV filename based on whether ntrain exists
        if ntrain:
            csv_name = f"adapted_model_S{seed}_N{ntrain}_NP{npool}_Shf{shift}_B{batch}_H{hdim}_batch_log.csv"
        else:
            csv_name = f"adapted_model_S{seed}_NP{npool}_Shf{shift}_B{batch}_H{hdim}_batch_log.csv"
            
        csv_path = os.path.join(target_dir, csv_name)

        if os.path.exists(csv_path):
            mod_time = os.path.getmtime(log_file)
            params_key = (seed, ntrain, npool, shift, batch, hdim)

            # Update if this parameter combo is new or newer than the stored one
            if params_key not in runs_dict or mod_time > runs_dict[params_key][2]:
                runs_dict[params_key] = (log_file, csv_path, mod_time)

    # 2. Extraction Phase
    compiled_data = []

    for params_key, (log_file, csv_path, mod_time) in runs_dict.items():
        seed, ntrain, npool, shift, batch, hdim = params_key

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
            df_csv['Wasserstein_Distance'] = wasserstein_val
            df_csv['Seed'] = int(seed)
            
            # If ntrain is missing (Exp3), fallback to the known config value (250000)
            df_csv['Train_Size'] = int(ntrain) if ntrain else 250000
            
            df_csv['Pool_Size'] = int(npool)
            df_csv['Shift'] = float(shift)
            df_csv['Batch_Size'] = int(batch)
            df_csv['Hidden_Dim'] = int(hdim)

            compiled_data.append(df_csv)

        except pd.errors.EmptyDataError:
            print(f"Warning: Empty CSV found at {csv_path}")

    if compiled_data:
        return pd.concat(compiled_data, ignore_index=True)
    return pd.DataFrame()

if __name__ == "__main__":
    print("Scanning directories and compiling results for Exp 3 and Exp 4...")

    df_exp3 = extract_time_series_data('results', 'phylogenetic_gmm_exp3')
    df_exp4 = extract_time_series_data('results', 'phylogenetic_gmm_exp4')

    df_master = pd.concat([df_exp3, df_exp4], ignore_index=True)

    if df_master.empty:
        print("Error: No data successfully extracted. Check your file paths.")
        exit()

    # Identify unique runs by their hyperparameters
    unique_runs = df_master[['Experiment', 'Seed', 'Train_Size', 'Pool_Size', 'Shift', 'Batch_Size', 'Hidden_Dim']].drop_duplicates()
    actual_run_count = len(unique_runs)
    total_rows = len(df_master)

    print("\n" + "="*50)
    print("DATA EXTRACTION REPORT")
    print("="*50)
    print(f"Total Timesteps (Rows) Extracted: {total_rows:,}")
    print(f"Actual Unique Runs Found:         {actual_run_count}")
    
    # Let's add a quick printout to confirm it found both experiments!
    print("\nRuns per Experiment:")
    print(unique_runs['Experiment'].value_counts().to_string())

    # Rearrange columns so Experiment, Wasserstein, and Train_Size are prioritized
    base_cols = ['Experiment', 'Wasserstein_Distance', 'Seed', 'Train_Size', 'Pool_Size', 'Shift', 'Batch_Size', 'Hidden_Dim']
    
    remaining_cols = [col for col in df_master.columns if col not in base_cols]
    final_cols = base_cols + remaining_cols

    df_master = df_master[final_cols]

    output_filename = "exp3_exp4_timeseries_dataset.csv"
    df_master.to_csv(output_filename, index=False)
    print(f"\nSuccess! Final dataset saved to '{output_filename}'.")
