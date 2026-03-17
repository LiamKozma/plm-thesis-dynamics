import pandas as pd
import glob
import os
import re

def extract_wasserstein_from_logs(base_dir):
    """
    Parses the adapt_log_*.log files to extract the calculated Wasserstein distance
    and merges it with the existing master_adaptation_results.csv.
    """
    target_dir = os.path.join(base_dir, "phylogenetic_gmm", "experiments", "adapt")
    log_files = glob.glob(os.path.join(target_dir, "adapt_log_*.log"))
    
    if not log_files:
        print(f"No .log files found in {target_dir}")
        return
        
    print(f"Found {len(log_files)} log files. Extracting Wasserstein distances...")
    
    # Regex to parse the filename
    file_pattern = re.compile(r"adapt_log_S(\d+)_NP(\d+)_Shf([\d\.]+)_B(\d+)_H(\d+)\.log")
    
    data = []
    
    for log_file in log_files:
        filename = os.path.basename(log_file)
        match = file_pattern.match(filename)
        
        if match:
            seed = int(match.group(1))
            n_pool = int(match.group(2))
            shift = float(match.group(3))
            batch_size = int(match.group(4))
            hidden_dim = int(match.group(5))
            
            wasserstein_dist = None
            
            # Read the log file line by line from the bottom up (since it's printed at the end)
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if "Wasserstein" in line and "|" in line:
                        # e.g., "Wasserstein     | 12.345678      "
                        parts = line.split("|")
                        if len(parts) == 2:
                            try:
                                wasserstein_dist = float(parts[1].strip())
                                break
                            except ValueError:
                                pass
            
            if wasserstein_dist is not None:
                data.append({
                    'seed': seed,
                    'n_pool': n_pool,
                    'shift': shift,
                    'batch_size': batch_size,
                    'hidden_dim': hidden_dim,
                    'wasserstein_distance': wasserstein_dist
                })
        else:
            print(f"Filename {filename} did not match expected pattern.")
            
    # Convert to dataframe
    w_df = pd.DataFrame(data)
    
    # Load the existing master CSV
    master_csv_path = os.path.join(base_dir, "master_adaptation_results.csv")
    if not os.path.exists(master_csv_path):
        print(f"Could not find {master_csv_path}. Please run compile_results.py first.")
        return
        
    master_df = pd.read_csv(master_csv_path)
    
    # Merge the Wasserstein distances into the master dataframe
    merged_df = pd.merge(
        master_df, 
        w_df, 
        on=['seed', 'n_pool', 'shift', 'batch_size', 'hidden_dim'], 
        how='left'
    )
    
    # Save the updated master CSV
    merged_df.to_csv(master_csv_path, index=False)
    
    print("\n--- Wasserstein Extraction Complete ---")
    print(f"Successfully added 'wasserstein_distance' to {master_csv_path}")
    
    # Print a quick summary mapping 'shift' to average 'wasserstein_distance'
    summary = merged_df[['shift', 'wasserstein_distance']].drop_duplicates().groupby('shift').mean()
    print("\nAverage Wasserstein Distance by Shift Severity:")
    print(summary)

if __name__ == "__main__":
    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    extract_wasserstein_from_logs(RESULTS_DIR)
