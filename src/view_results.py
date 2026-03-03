import os
import glob
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import re

def parse_adapt_log(filepath):
    """Extracts metrics from the final evaluation block of an adaptation log."""
    last_mse, last_rbme, w_dist = None, None, None
    with open(filepath, 'r') as f:
        for line in f:
            if "Wasserstein" in line:
                w_dist = float(line.split('|')[1].strip())
            elif "Final Test MSE" in line:
                last_mse = float(line.split('|')[1].strip())
            elif "Final Test rBME" in line:
                last_rbme = float(line.split('|')[1].strip())
    return last_mse, last_rbme, w_dist

def parse_eval_log(filepath):
    """Extracts metrics from an evaluation log."""
    mse, rbme, w_dist = None, None, None
    with open(filepath, 'r') as f:
        for line in f:
            if "MSE Loss" in line:
                mse = float(line.split('|')[1].strip())
            elif "rBME" in line:
                rbme = float(line.split('|')[1].strip())
            elif "Wasserstein" in line:
                w_dist = float(line.split('|')[1].strip())
    return mse, rbme, w_dist

def generate_crash_plot(df, output_path, title_prefix="", mode_name=""):
    """Generates the dual-axis static crash plot based on Wasserstein distance."""
    plot_df = df.sort_values("Wasserstein")
    if plot_df.empty: return

    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Wasserstein Distance (Source vs Target)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation MSE (Standard)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(plot_df['Wasserstein'], plot_df['MSE (Standard)'], color=color, marker='o', linewidth=2, label='MSE (Loss)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, plot_df['MSE (Standard)'].max() * 1.5) 

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('rBME (Tail-Sensitive)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(plot_df['Wasserstein'], plot_df['rBME (Tail)'], color=color, marker='s', linestyle='--', linewidth=2, label='rBME')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')

    # Annotate crash
    crash_row = plot_df.loc[plot_df['rBME (Tail)'].idxmax()]
    if crash_row['rBME (Tail)'] > 1.0:
        ax2.annotate(f"CRASH\nrBME={crash_row['rBME (Tail)']:.1e}", 
                     xy=(crash_row['Wasserstein'], crash_row['rBME (Tail)']), 
                     xytext=(crash_row['Wasserstein'], crash_row['rBME (Tail)'] * 1.5), 
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='center')

    plt.title(f'{title_prefix} Results: {mode_name} Dynamics', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"-> Summary plot saved to {output_path}")
    plt.close()

def generate_batch_plots(batch_df, output_dir, title_prefix=""):
    """Generates the batch-by-batch recovery curves with shaded standard deviation."""
    if batch_df.empty: return
    
    # We group by Train Size and Pool Size to avoid putting 180 lines on one messy chart.
    # This generates a clean, separate plot for each condition in your matrix.
    for (n_train, n_pool), group_df in batch_df.groupby(['N_Train', 'N_Pool']):
        sns.set_style("whitegrid")
        
        # --- 1. Plot MSE Recovery ---
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=group_df, x='batch_number', y='test_mse', hue='Shift', 
            palette='viridis', errorbar='sd', linewidth=2, ax=ax
        )
        ax.set_title(f"{title_prefix} MSE Recovery (Train: {n_train}, Pool: {n_pool})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Number of Pool Batches Consumed", fontsize=12, fontweight='bold')
        ax.set_ylabel("Validation MSE", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"recovery_mse_NTr{n_train}_NP{n_pool}.png"), dpi=300)
        plt.close()

        # --- 2. Plot rBME Recovery ---
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=group_df, x='batch_number', y='test_rbme', hue='Shift', 
            palette='magma', errorbar='sd', linewidth=2, ax=ax
        )
        ax.set_title(f"{title_prefix} rBME Recovery (Train: {n_train}, Pool: {n_pool})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Number of Pool Batches Consumed", fontsize=12, fontweight='bold')
        ax.set_ylabel("Validation rBME (Tail-Sensitive)", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"recovery_rbme_NTr{n_train}_NP{n_pool}.png"), dpi=300)
        plt.close()
        
        print(f"-> Batch dynamics plots saved for N_Train={n_train}, N_Pool={n_pool}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="highdim", choices=['manifold', 'highdim', 'old'])
    parser.add_argument("--mode", type=str, required=True, choices=['adapt', 'eval'])
    args = parser.parse_args()

    base_dir = f"results/experiments/{args.mode}" if args.data == 'old' else f"results/{args.data}/experiments/{args.mode}"
    print(f"Reading logs from: {base_dir}...")
    
    # 1. Parse standard logs for summary table
    data = []
    log_files = glob.glob(os.path.join(base_dir, "**/*.log"), recursive=True)

    for log in log_files:
        filename = os.path.basename(log)
        match = re.search(r'NTr(\d+)_NP(\d+)_Shf([0-9.]+)_s(\d+)', filename)
        if not match: continue
            
        n_train, n_pool, shift_val, seed = int(match.group(1)), int(match.group(2)), float(match.group(3)), int(match.group(4))

        mse, rbme, w_dist, mode_label = 0.0, 0.0, 0.0, "Unknown"
        if args.mode == "adapt" and "adapt" in filename:
            mode_label = "Adaptation (Train)"
            mse, rbme, w_dist = parse_adapt_log(log)
        elif args.mode == "eval" and "eval" in filename:
            mode_label = "Inference (Eval)"
            mse, rbme, w_dist = parse_eval_log(log)
            
        if mse is not None and rbme is not None and mode_label != "Unknown":
            data.append({
                "Wasserstein": w_dist if w_dist is not None else shift_val,
                "Mode": mode_label, "N_Train": n_train, "N_Pool": n_pool, 
                "Seed": seed, "MSE (Standard)": mse, "rBME (Tail)": rbme
            })

    output_dir = f"results/{args.data}/{args.mode}"
    os.makedirs(output_dir, exist_ok=True)

    if data:
        df = pd.DataFrame(data).sort_values(by=["Wasserstein"])
        print("\n" + "="*70)
        print(f" RESULTS SUMMARY: {args.data.upper()} DATA | {args.mode.upper()} MODE")
        print("="*70)
        print(df.to_string(index=False, formatters={'MSE (Standard)': '{:,.6f}'.format, 'rBME (Tail)': '{:,.4f}'.format}))
        print("="*70 + "\n")
        
        generate_crash_plot(df, os.path.join(output_dir, "results_plot.png"), title_prefix=args.data.upper(), mode_name="Adaptation" if args.mode == "adapt" else "Evaluation")
    else:
        print(f"No summary logs found in {base_dir}.")

    # 2. Parse batch CSVs for recovery plots (Only for adapt mode)
    if args.mode == "adapt":
        print("\nProcessing batch-by-batch recovery logs...")
        batch_data = []
        batch_files = glob.glob(os.path.join(base_dir, "**/*_batch_log.csv"), recursive=True)
        
        for b_file in batch_files:
            filename = os.path.basename(b_file)
            match = re.search(r'NTr(\d+)_NP(\d+)_Shf([0-9.]+)_s(\d+)', filename)
            if not match: continue
            
            df_b = pd.read_csv(b_file)
            df_b['N_Train'] = int(match.group(1))
            df_b['N_Pool'] = int(match.group(2))
            df_b['Shift'] = float(match.group(3)) # Use Shift to color the lines!
            df_b['Seed'] = int(match.group(4))
            batch_data.append(df_b)
            
        if batch_data:
            batch_df = pd.concat(batch_data, ignore_index=True)
            generate_batch_plots(batch_df, output_dir, title_prefix=args.data.upper())
        else:
            print("No _batch_log.csv files found.")

if __name__ == "__main__":
    main()
