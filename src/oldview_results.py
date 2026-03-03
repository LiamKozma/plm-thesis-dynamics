import os
import glob
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def parse_adapt_log(filepath):
    """Extracts metrics from the FINAL epoch (20) of a training log."""
    last_mse = None
    last_rbme = None
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in reversed(lines):
            parts = line.split('|')
            if len(parts) == 4 and parts[0].strip().isdigit():
                last_mse = float(parts[2].strip())
                last_rbme = float(parts[3].strip())
                break
    return last_mse, last_rbme

def parse_eval_log(filepath):
    """Extracts metrics from an evaluation log."""
    mse = None
    rbme = None
    with open(filepath, 'r') as f:
        for line in f:
            if "MSE Loss" in line:
                mse = float(line.split('|')[1].strip())
            elif "rBME" in line:
                rbme = float(line.split('|')[1].strip())
    return mse, rbme

def generate_crash_plot(df, output_path, title_prefix=""):
    """Generates the dual-axis plot: MSE (Standard) vs rBME (Tail)."""
    
    # Filter for Adaptation mode only (where the "Crash" happens)
    plot_df = df[df['Mode'] == "Adaptation (Train)"].sort_values("Shift")
    
    if plot_df.empty:
        return

    # Setup Plot
    sns.set_style("whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Axis 1: MSE (The "Lie") ---
    color = 'tab:blue'
    ax1.set_xlabel(r'Distribution Shift Magnitude ($\lambda$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation MSE (Standard)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(plot_df['Shift'], plot_df['MSE (Standard)'], color=color, marker='o', linewidth=2, label='MSE (Loss)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Adjust ylim to match your previous style (zoomed in to show stability)
    # or keep it dynamic if data varies wildly
    max_mse = plot_df['MSE (Standard)'].max()
    ax1.set_ylim(0, max_mse * 1.5) 

    # --- Axis 2: rBME (The "Truth") ---
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('rBME (Tail-Sensitive)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(plot_df['Shift'], plot_df['rBME (Tail)'], color=color, marker='s', linestyle='--', linewidth=2, label='rBME')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log') # Log scale for the crash

    # --- Annotate the Max Failure Point ---
    # Find the row with the highest rBME (The Crash)
    crash_row = plot_df.loc[plot_df['rBME (Tail)'].idxmax()]
    crash_shift = crash_row['Shift']
    crash_val = crash_row['rBME (Tail)']
    
    # Only annotate if it's significant (e.g., > 1.0)
    if crash_val > 1.0:
        ax2.annotate(f'CRASH\nrBME={crash_val:.1e}', 
                     xy=(crash_shift, crash_val), 
                     xytext=(crash_shift, crash_val * 1.5), 
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='center')

    # Title and Layout
    plt.title(f'{title_prefix} Results: Adaptation Dynamics', fontsize=14)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="manifold", choices=['manifold', 'highdim', 'old'],
                        help="Which dataset results to view?")
    args = parser.parse_args()

    # Dynamic Directory Selection
    if args.data == 'old':
        base_dir = "results/experiments"
    else:
        base_dir = f"results/{args.data}/experiments"

    print(f"Reading logs from: {base_dir}...")
    
    data = []
    log_files = glob.glob(os.path.join(base_dir, "**/*.log"), recursive=True)
    
    for log in log_files:
        filename = os.path.basename(log)
        try:
            shift_str = filename.replace('.log', '').split('_')[-1]
            shift_val = float(shift_str)
        except:
            continue

        mode = "Unknown"
        mse = 0.0
        rbme = 0.0
        
        if "adapt" in log or "adapt" in filename:
            mode = "Adaptation (Train)"
            mse, rbme = parse_adapt_log(log)
        elif "eval" in log:
            mode = "Inference (Eval)"
            mse, rbme = parse_eval_log(log)
            
        if mse is not None and rbme is not None:
            data.append({
                "Shift": shift_val, 
                "Mode": mode, 
                "MSE (Standard)": mse, 
                "rBME (Tail)": rbme
            })

    if data:
        df = pd.DataFrame(data)
        df = df.sort_values(by=["Mode", "Shift"])
        
        # 1. Print Table
        print("\n" + "="*60)
        print(f" RESULTS SUMMARY: {args.data.upper()} DATA")
        print("="*60)
        print(df.to_string(index=False, formatters={
            'MSE (Standard)': '{:,.6f}'.format,
            'rBME (Tail)': '{:,.4f}'.format
        }))
        print("="*60 + "\n")
        
        # 2. Generate Plot
        output_plot = f"results/{args.data}/results_plot.png"
        
        # Make sure directory exists (if using 'old' or custom paths)
        os.makedirs(os.path.dirname(output_plot), exist_ok=True)
        
        generate_crash_plot(df, output_plot, title_prefix=args.data.upper())
        
    else:
        print(f"No logs found in {base_dir}.")

if __name__ == "__main__":
    main()
