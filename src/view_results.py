import os
import glob
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import re

def parse_adapt_log(filepath):
    """Extracts metrics from the final evaluation block of an adaptation log."""
    last_ce, last_f1, w_dist = None, None, None
    with open(filepath, 'r') as f:
        for line in f:
            if "Wasserstein" in line:
                w_dist = float(line.split('|')[1].strip())
            elif "Final Test CE" in line:
                last_ce = float(line.split('|')[1].strip())
            elif "Final Test F1" in line:
                last_f1 = float(line.split('|')[1].strip())
    return last_ce, last_f1, w_dist

def parse_eval_log(filepath):
    """Extracts metrics from an evaluation log."""
    ce, f1, w_dist = None, None, None
    with open(filepath, 'r') as f:
        for line in f:
            if "CE Loss" in line:
                ce = float(line.split('|')[1].strip())
            elif "Macro F1" in line:
                f1 = float(line.split('|')[1].strip())
            elif "Wasserstein" in line:
                w_dist = float(line.split('|')[1].strip())
    return ce, f1, w_dist

def generate_crash_plot(df, output_path, title_prefix="", mode_name=""):
    """Generates the dual-axis static crash plot based on Wasserstein distance."""
    plot_df = df.sort_values("Wasserstein")
    if plot_df.empty: return

    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left Axis: Cross-Entropy Loss
    color = 'tab:blue'
    ax1.set_xlabel('Wasserstein Distance (Evolutionary Drift)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Cross-Entropy Loss', color=color, fontsize=12, fontweight='bold')
    ax1.plot(plot_df['Wasserstein'], plot_df['CE Loss'], color=color, marker='o', linewidth=2, label='CE Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # Right Axis: Macro F1 Score
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Macro F1 Score (Classification)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(plot_df['Wasserstein'], plot_df['Macro F1'], color=color, marker='s', linestyle='--', linewidth=2, label='Macro F1')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.05) # F1 is bounded between 0 and 1

    # Annotate crash (Lowest F1 Score)
    crash_row = plot_df.loc[plot_df['Macro F1'].idxmin()]
    if crash_row['Macro F1'] < 0.5: # Arbitrary threshold for a "crash" in predictive power
        ax2.annotate(f"CRASH\nF1={crash_row['Macro F1']:.2f}", 
                     xy=(crash_row['Wasserstein'], crash_row['Macro F1']), 
                     xytext=(crash_row['Wasserstein'], crash_row['Macro F1'] + 0.15), 
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
    
    for (n_train, n_pool), group_df in batch_df.groupby(['N_Train', 'N_Pool']):
        sns.set_style("whitegrid")
        
        # --- 1. Plot CE Loss Recovery ---
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=group_df, x='batch_number', y='test_ce', hue='Shift', 
            palette='viridis', errorbar='sd', linewidth=2, ax=ax
        )
        ax.set_title(f"{title_prefix} CE Loss Recovery (Train: {n_train}, Pool: {n_pool})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Number of Pool Batches Consumed", fontsize=12, fontweight='bold')
        ax.set_ylabel("Validation CE Loss", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"recovery_ce_NTr{n_train}_NP{n_pool}.png"), dpi=300)
        plt.close()

        # --- 2. Plot Macro F1 Recovery Threshold ---
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            data=group_df, x='batch_number', y='test_f1', hue='Shift', 
            palette='magma', errorbar='sd', linewidth=2, ax=ax
        )
        ax.set_title(f"{title_prefix} F1 Recovery Threshold (Train: {n_train}, Pool: {n_pool})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Number of Pool Batches Consumed", fontsize=12, fontweight='bold')
        ax.set_ylabel("Validation Macro F1 Score", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"recovery_f1_NTr{n_train}_NP{n_pool}.png"), dpi=300)
        plt.close()
        
        print(f"-> Batch dynamics plots saved for N_Train={n_train}, N_Pool={n_pool}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="phylogenetic_gmm", help="Dataset name for titles/paths")
    parser.add_argument("--mode", type=str, required=True, choices=['adapt', 'eval'])
    args = parser.parse_args()

    # Create the output directory based on mode
    base_dir = f"results/{args.data}/experiments/{args.mode}"
    print(f"Reading logs from: {base_dir}...")
    
    # 1. Parse standard logs for summary table
    data = []
    log_files = glob.glob(os.path.join(base_dir, "**/*.log"), recursive=True)

    for log in log_files:
        filename = os.path.basename(log)
        match = re.search(r'NTr(\d+)_NP(\d+)_Shf([0-9.]+)_s(\d+)', filename)
        if not match: continue
            
        n_train, n_pool, shift_val, seed = int(match.group(1)), int(match.group(2)), float(match.group(3)), int(match.group(4))

        ce, f1, w_dist, mode_label = None, None, None, "Unknown"
        if args.mode == "adapt" and "adapt" in filename:
            mode_label = "Adaptation (Train)"
            ce, f1, w_dist = parse_adapt_log(log)
        elif args.mode == "eval" and "eval" in filename:
            mode_label = "Inference (Eval)"
            ce, f1, w_dist = parse_eval_log(log)
            
        if ce is not None and f1 is not None and mode_label != "Unknown":
            data.append({
                "Wasserstein": w_dist if w_dist is not None else shift_val,
                "Mode": mode_label, "N_Train": n_train, "N_Pool": n_pool, 
                "Seed": seed, "CE Loss": ce, "Macro F1": f1
            })

    output_dir = f"results/{args.data}/{args.mode}"
    os.makedirs(output_dir, exist_ok=True)

    if data:
        df = pd.DataFrame(data).sort_values(by=["Wasserstein"])
        print("\n" + "="*70)
        print(f" RESULTS SUMMARY: {args.data.upper()} DATA | {args.mode.upper()} MODE")
        print("="*70)
        print(df.to_string(index=False, formatters={'CE Loss': '{:,.6f}'.format, 'Macro F1': '{:,.4f}'.format}))
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
            df_b['Shift'] = float(match.group(3))
            df_b['Seed'] = int(match.group(4))
            batch_data.append(df_b)
            
        if batch_data:
            batch_df = pd.concat(batch_data, ignore_index=True)
            generate_batch_plots(batch_df, output_dir, title_prefix=args.data.upper())
        else:
            print("No _batch_log.csv files found.")

if __name__ == "__main__":
    main()
