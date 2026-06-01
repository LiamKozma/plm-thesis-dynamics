import pandas as pd
from compile_3_4_5 import extract_time_series_data  

print("Gathering Experiment 3 and 4 from results directory...")
df_exp3 = extract_time_series_data('results', 'phylogenetic_gmm_exp3')
df_exp4 = extract_time_series_data('results', 'phylogenetic_gmm_exp4')

print("Loading rescued Experiment 5 data...")
df_exp5 = pd.read_csv("rescued_exp5_data.csv")

print("Stitching datasets together...")
df_master = pd.concat([df_exp3, df_exp4, df_exp5], ignore_index=True)

output_filename = "master_thesis_dataset_exps3_4_5.csv"
df_master.to_csv(output_filename, index=False)

print("\n" + "="*50)
print("FINAL DATASET REPORT")
print("="*50)
print(f"Total Timesteps (Rows): {len(df_master):,}")
print(f"Total Unique Runs:      {len(df_master[['Experiment', 'Seed', 'Train_Size', 'Pool_Size', 'Shift', 'Base_Sigma', 'Batch_Size', 'Hidden_Dim']].drop_duplicates())}")
print("\nRuns per Experiment:")
print(df_master[['Experiment', 'Seed', 'Train_Size', 'Pool_Size', 'Shift', 'Base_Sigma', 'Batch_Size', 'Hidden_Dim']].drop_duplicates()['Experiment'].value_counts().to_string())
print(f"\nSuccess! Master dataset saved to '{output_filename}'.")
