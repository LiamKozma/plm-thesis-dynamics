import os
import pandas as pd
import glob
import re

def compile_all_runs(results_base_dir, output_file):
    print("Crawling Nextflow outputs...")
    
    # Define our two experiment folders and their optimizers
    experiments = {
        "phylogenetic_gmm_exp1": "Adam (OG)",
        "phylogenetic_gmm_exp2": "AdamW + Cosine"
    }
    
    all_data = []
    
    for exp_folder, optimizer_name in experiments.items():
        search_path = os.path.join(results_base_dir, exp_folder, "experiments", "adapt", "*_batch_log.csv")
        csv_files = glob.glob(search_path)
        
        print(f"Found {len(csv_files)} logs in {exp_folder}")
        
        for file in csv_files:
            # Extract hyperparameters from filename: adapted_model_S42_NP100000_Shf2.0_B64_H512_batch_log.csv
            filename = os.path.basename(file)
            match = re.search(r"S(\d+)_NP(\d+)_Shf([\d\.]+)_B(\d+)_H(\d+)", filename)
            
            if match:
                df = pd.read_csv(file)
                df['seed'] = int(match.group(1))
                df['n_pool'] = int(match.group(2))
                df['shift'] = float(match.group(3))
                df['batch_size'] = int(match.group(4))
                df['hidden_dim'] = int(match.group(5))
                df['optimizer'] = optimizer_name
                df['experiment'] = exp_folder
                
                all_data.append(df)
            else:
                print(f"Could not parse parameters from {filename}")

    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        master_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully compiled {len(all_data)} runs into {output_file}!")
        print(f"Total rows: {len(master_df):,}")
    else:
        print("No data found to compile.")

if __name__ == "__main__":
    RESULTS_DIR = "/work/ah2lab/LiamK/plm-thesis-dynamics/results"
    OUTPUT_PATH = os.path.join(RESULTS_DIR, "master_adaptation_results.csv")
    compile_all_runs(RESULTS_DIR, OUTPUT_PATH)
