import glob
import os
import re
import pandas as pd

def extract_wasserstein_and_merge(results_dir, master_csv_path):
    print(f"🔍 Searching for log files in {results_dir}...")
    
    search_pattern = os.path.join(results_dir, "**", "adapt_log_*.log")
    log_files = glob.glob(search_pattern, recursive=True)
    
    if not log_files:
        print("❌ No log files found. Please check the directory path.")
        return

    print(f"Found {len(log_files)} log files. Extracting metrics...")
    
    extracted_data = []
    w_pattern = re.compile(r"Wasserstein\s+\|\s+([0-9\.]+)")
    file_pattern = re.compile(r"adapt_log_S(\d+)_NP(\d+)_Shf([0-9\.]+)_B(\d+)_H(\d+)\.log")

    for filepath in log_files:
        filename = os.path.basename(filepath)
        match = file_pattern.search(filename)
        
        if match:
            seed = int(match.group(1))
            n_pool = int(match.group(2))
            shift = float(match.group(3))
            batch = int(match.group(4))
            h_dim = int(match.group(5))
            
            # THE FIX: Determine experiment from the filepath
            if "phylogenetic_gmm_exp1" in filepath:
                exp_name = "phylogenetic_gmm_exp1"
            else:
                exp_name = "phylogenetic_gmm_exp2"
            
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    w_match = w_pattern.search(content)
                    if w_match:
                        w_dist = float(w_match.group(1))
                        
                        extracted_data.append({
                            'experiment': exp_name, # ADDED TO DICT
                            'seed': seed,
                            'n_pool': n_pool,
                            'shift': shift,
                            'batch_size': batch,
                            'hidden_dim': h_dim,
                            'wasserstein': w_dist
                        })
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    df_wasserstein = pd.DataFrame(extracted_data)
    print(f"✅ Successfully extracted {len(df_wasserstein)} Wasserstein records.")
    
    print(f"Loading master dataset from {master_csv_path}...")
    master_df = pd.read_csv(master_csv_path)
    
    # THE FIX: Add 'experiment' to the merge keys to prevent duplication
    merge_keys = ['experiment', 'seed', 'n_pool', 'shift', 'batch_size', 'hidden_dim']
    
    if 'wasserstein' in master_df.columns:
        master_df = master_df.drop(columns=['wasserstein'])
        
    final_df = pd.merge(master_df, df_wasserstein, on=merge_keys, how='left')
    
    missing = final_df['wasserstein'].isnull().sum()
    if missing > 0:
        print(f"⚠️ Warning: {missing} rows in master CSV could not be matched with a Wasserstein distance.")
    
    updated_csv_path = master_csv_path.replace(".csv", "_with_W.csv")
    final_df.to_csv(updated_csv_path, index=False)
    print(f"🎉 Success! Updated dataset saved to: {updated_csv_path}")

if __name__ == "__main__":
    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    CSV_FILE = os.path.join(RESULTS_DIR, "master_adaptation_results.csv")
    extract_wasserstein_and_merge(RESULTS_DIR, CSV_FILE)
