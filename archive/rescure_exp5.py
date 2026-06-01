import glob
import os
import re

import pandas as pd


def rescue_exp5_from_work_dir(work_dir_path):
    print(f"Crawling Nextflow work directory: {work_dir_path} ...")
    
    csv_files = glob.glob(os.path.join(work_dir_path, '*/*', '*_batch_log.csv'))
    
    compiled_data = []
    rescued_count = 0

    for csv_path in csv_files:
        dir_path = os.path.dirname(csv_path)
        filename = os.path.basename(csv_path)
        
        # 1. Extract known parameters
        seed_m = re.search(r'_S(\d+)', filename)
        ntrain_m = re.search(r'_N(\d+)_NP', filename)
        npool_m = re.search(r'_NP(\d+)', filename)
        shift_m = re.search(r'_Shf([0-9.]+)', filename)
        batch_m = re.search(r'_B(\d+)', filename)
        hdim_m = re.search(r'_H(\d+)', filename)

        if not (seed_m and shift_m):
            continue

        seed = seed_m.group(1)
        shift = shift_m.group(1)
        
        # 2. THE SYMLINK HACK: Trace the source data file back to GEN_SOURCE
        sigma = "UNKNOWN"
        source_files = glob.glob(os.path.join(dir_path, 'source_X_*.npy'))
        
        if source_files:
            # os.path.realpath resolves the symlink to its true origin directory
            true_origin_path = os.path.realpath(source_files[0])
            origin_dir = os.path.dirname(true_origin_path)
            origin_cmd_path = os.path.join(origin_dir, '.command.sh')
            
            # Read the GEN_SOURCE command script to find the sigma
            if os.path.exists(origin_cmd_path):
                with open(origin_cmd_path, 'r') as f:
                    cmd_content = f.read()
                    # Look for the sigma argument passed to the generation script
                    sigma_match = re.search(r'(?:--base_sigma|--sigma|-s)\s+([0-9.]+)', cmd_content)
                    if sigma_match:
                        sigma = sigma_match.group(1)
        
        # 3. Scrape Wasserstein from the isolated log file
        log_files = glob.glob(os.path.join(dir_path, 'adapt_log_*.log'))
        wasserstein_val = None
        if log_files:
            with open(log_files[0], 'r') as f:
                w_match = re.search(r'Wasserstein\s+\|\s+([0-9.]+)', f.read())
                if w_match:
                    wasserstein_val = float(w_match.group(1))

        # 4. Build the DataFrame
        try:
            df_csv = pd.read_csv(csv_path)
            if df_csv.empty:
                continue

            df_csv['Experiment'] = 'phylogenetic_gmm_exp5'
            df_csv['Wasserstein_Distance'] = wasserstein_val
            df_csv['Seed'] = int(seed)
            df_csv['Train_Size'] = int(ntrain_m.group(1)) if ntrain_m else 500000
            df_csv['Pool_Size'] = int(npool_m.group(1)) if npool_m else 1000000
            df_csv['Shift'] = float(shift)
            df_csv['Base_Sigma'] = float(sigma) if sigma != "UNKNOWN" else None
            df_csv['Batch_Size'] = int(batch_m.group(1)) if batch_m else 256
            df_csv['Hidden_Dim'] = int(hdim_m.group(1)) if hdim_m else 512

            compiled_data.append(df_csv)
            rescued_count += 1

        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    if compiled_data:
        master_df = pd.concat(compiled_data, ignore_index=True)
        print(f"\nSuccessfully rescued {rescued_count} out of 36 expected runs!")
        return master_df
    return pd.DataFrame()

if __name__ == "__main__":
    df_rescued = rescue_exp5_from_work_dir('work_exp5')
    
    if not df_rescued.empty:
        output_filename = "rescued_exp5_data.csv"
        df_rescued.to_csv(output_filename, index=False)
        print(f"Data saved to {output_filename}")
        
        print("\nVerification of Rescued Sigmas:")
        print(df_rescued[['Seed', 'Shift', 'Base_Sigma']].drop_duplicates().sort_values(by=['Shift', 'Base_Sigma']).to_string(index=False))
