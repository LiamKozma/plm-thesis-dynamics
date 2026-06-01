import glob
import os
import re

import pandas as pd


def extract_time_series_data(base_dir, experiment_name):
    target_dir = os.path.join(base_dir, experiment_name, 'experiments', 'adapt')
    log_pattern = os.path.join(target_dir, 'adapt_log_*.log')
    log_files = glob.glob(log_pattern)

    runs_dict = {}
    for log_file in log_files:
        filename = os.path.basename(log_file)

        seed_m = re.search(r'_S(\d+)', filename)
        ntrain_m = re.search(r'_N(\d+)_NP', filename)
        npool_m = re.search(r'_NP(\d+)', filename)
        shift_m = re.search(r'_Shf([0-9.]+)', filename)
        batch_m = re.search(r'_B(\d+)', filename)
        hdim_m = re.search(r'_H(\d+)', filename)
        sigma_m = re.search(r'_(?:Sig|Sigma|Sg)([0-9.]+)', filename)

        if not (seed_m and npool_m and shift_m and batch_m and hdim_m):
            continue 

        seed = seed_m.group(1)
        ntrain = ntrain_m.group(1) if ntrain_m else None
        npool = npool_m.group(1)
        shift = shift_m.group(1)
        batch = batch_m.group(1)
        hdim = hdim_m.group(1)
        sigma = sigma_m.group(1) if sigma_m else "0.5"

        csv_name = filename.replace('adapt_log_', 'adapted_model_').replace('.log', '_batch_log.csv')
        csv_path = os.path.join(target_dir, csv_name)

        if os.path.exists(csv_path):
            mod_time = os.path.getmtime(log_file)
            params_key = (seed, ntrain, npool, shift, sigma, batch, hdim)
            if params_key not in runs_dict or mod_time > runs_dict[params_key][2]:
                runs_dict[params_key] = (log_file, csv_path, mod_time)

    compiled_data = []
    for params_key, (log_file, csv_path, mod_time) in runs_dict.items():
        seed, ntrain, npool, shift, sigma, batch, hdim = params_key

        wasserstein_val = None
        with open(log_file, 'r') as f:
            w_match = re.search(r'Wasserstein\s+\|\s+([0-9.]+)', f.read())
            if w_match:
                wasserstein_val = float(w_match.group(1))

        try:
            df_csv = pd.read_csv(csv_path)
            if df_csv.empty: continue

            df_csv['Experiment'] = experiment_name
            df_csv['Wasserstein_Distance'] = wasserstein_val
            df_csv['Seed'] = int(seed)
            if ntrain: df_csv['Train_Size'] = int(ntrain)
            else: df_csv['Train_Size'] = 500000 if experiment_name == 'phylogenetic_gmm_exp5' else 250000
            df_csv['Pool_Size'] = int(npool)
            df_csv['Shift'] = float(shift)
            df_csv['Base_Sigma'] = float(sigma)
            df_csv['Batch_Size'] = int(batch)
            df_csv['Hidden_Dim'] = int(hdim)

            compiled_data.append(df_csv)
        except pd.errors.EmptyDataError:
            pass

    if compiled_data: return pd.concat(compiled_data, ignore_index=True)
    return pd.DataFrame()
