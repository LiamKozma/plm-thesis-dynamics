import os
import glob
import re
import sys
import pandas as pd
from datetime import datetime

def extract_wasserstein(log_path):
    """Parses the text log to harvest the pre-calculated Wasserstein distance."""
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
        # Searches for the exact print string from your PyTorch scripts: "Wasserstein    | 0.123456"
        match = re.search(r"Wasserstein\s+\|\s+([\d\.]+)", content)
        if match:
            return float(match.group(1))
    return None

def compile_and_audit(results_base_dir, output_file):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=====================================================================")
    print(f" 🧬 PLM RECOVERY THRESHOLD: MASTER COMPILATION & AUDIT")
    print(f" 📅 Pipeline Executed: {current_time}")
    print("=====================================================================\n")
    
    experiments = {
        "phylogenetic_gmm_exp1": "Adam (OG)",
        "phylogenetic_gmm_exp2": "AdamW + Cosine"
    }
    
    all_data = []
    missing_logs = 0
    missing_wasserstein = 0

    print("--- 1. DATA EXTRACTION & HARVESTING ---")
    for exp_folder, optimizer_name in experiments.items():
        search_path = os.path.join(results_base_dir, exp_folder, "experiments", "adapt", "*_batch_log.csv")
        csv_files = glob.glob(search_path)
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            
            # Extract parameters safely
            match = re.search(r"S(\d+).*?NP(\d+).*?Shf([\d\.]+).*?B(\d+).*?H(\d+)", filename, re.IGNORECASE)
            if not match:
                continue
                
            seed, n_pool, shift, batch_size, hidden_dim = map(float, match.groups())
            
            # Reconstruct the expected text log filename based on main.nf structure
            log_filename = f"adapt_log_S{int(seed)}_NP{int(n_pool)}_Shf{shift}_B{int(batch_size)}_H{int(hidden_dim)}.log"
            log_path = os.path.join(os.path.dirname(csv_file), log_filename)
            
            # Harvest Wasserstein
            w_dist = extract_wasserstein(log_path)
            
            if w_dist is None:
                if not os.path.exists(log_path): missing_logs += 1
                else: missing_wasserstein += 1
                continue

            # Load the CSV payload
            df = pd.read_csv(csv_file)
            
            # Tag all metadata and the harvested distance
            df['seed'] = int(seed)
            df['n_pool'] = int(n_pool)
            df['shift'] = shift
            df['batch_size'] = int(batch_size)
            df['hidden_dim'] = int(hidden_dim)
            df['optimizer'] = optimizer_name
            df['experiment'] = exp_folder
            df['wasserstein'] = w_dist
            
            all_data.append(df)

    if not all_data:
        print("❌ FATAL: No valid data found. Check your directory paths.")
        sys.exit(1)

    master_df = pd.concat(all_data, ignore_index=True)
    master_df.to_csv(output_file, index=False)
    print(f"✅ Successfully compiled {len(master_df):,} rows into {output_file}")
    
    if missing_logs > 0 or missing_wasserstein > 0:
        print(f"⚠️ WARNING: {missing_logs} missing logs, {missing_wasserstein} logs missing Wasserstein strings.")

    print("\n--- 2. FINAL STATISTICAL AUDIT ---")
    
    # Isolate asymptotic performance for the audit
    idx = master_df.groupby(["optimizer", "shift", "seed", "n_pool", "batch_size", "hidden_dim"])["samples_seen"].idxmax()
    final_df = master_df.loc[idx].copy()
    
    total_runs = len(final_df)
    print(f"Total Unique Runs: {total_runs} (Expected: 144)")
    
    if total_runs == 144:
        print("✅ PASS: Experimental matrix is 100% complete.")
    else:
        print("❌ FAIL: Matrix is incomplete.")

    opt_counts = final_df['optimizer'].value_counts().to_dict()
    if all(count == 72 for count in opt_counts.values()):
        print("✅ PASS: Optimizer distribution is perfectly balanced (72 vs 72).")
    else:
        print(f"❌ FAIL: Optimizer imbalance detected: {opt_counts}")

    pool_counts = final_df['n_pool'].value_counts().to_dict()
    if all(count == 72 for count in pool_counts.values()):
        print("✅ PASS: Target pool sizes are perfectly balanced (72 vs 72).")
    else:
        print(f"❌ FAIL: Pool size imbalance detected: {pool_counts}")
        
    w_missing = final_df['wasserstein'].isna().sum()
    if w_missing == 0:
        print(f"✅ PASS: 100% of runs have empirical Wasserstein distances [{final_df['wasserstein'].min():.3f} to {final_df['wasserstein'].max():.3f}].")
    else:
        print(f"❌ FAIL: {w_missing} runs are missing Wasserstein data.")
        
    print("\n=====================================================================")
    if total_runs == 144 and w_missing == 0:
        print(" 🟢 VERIFIED: DATASET IS FLAWLESS AND READY FOR PLOTTING.")
    else:
        print(" 🔴 AUDIT FAILED: REVIEW WARNINGS ABOVE.")
    print("=====================================================================\n")

if __name__ == "__main__":
    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    OUTPUT_PATH = os.path.join(RESULTS_DIR, "master_thesis_results.csv")
    
    compile_and_audit(RESULTS_DIR, OUTPUT_PATH)
