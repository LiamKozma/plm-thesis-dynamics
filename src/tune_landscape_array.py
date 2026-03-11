import itertools
import re
import subprocess
import sys
import time

import pandas as pd


def run_experiment(oracle_layers, spread, sigma, seed, n_train=100000, n_families=1000, n_classes=100):
    """Runs the data generation script and captures the diagnostic metrics."""
    cmd = [
        "python", "src/generate_simulation.py",
        "--mode", "source",
        "--n_train", str(n_train),
        "--n_families", str(n_families),
        "--n_classes", str(n_classes),
        "--oracle_layers", oracle_layers,
        "--centroid_spread", str(spread),
        "--base_sigma", str(sigma),
        "--seed", str(seed)
    ]
    
    print(f"-> Testing: Oracle [{oracle_layers}] | Spread [{spread}] | Sigma [{sigma}]")
    start_time = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    output = result.stdout

    purity_match = re.search(r"Within-family purity:\s+([\d.]+)%", output)
    promiscuity_match = re.search(r"Family promiscuity:\s+([\d.]+)%", output)
    coverage_match = re.search(r"Class coverage:\s+([\d.]+)", output)

    if purity_match and promiscuity_match and coverage_match:
        purity = float(purity_match.group(1))
        promiscuity = float(promiscuity_match.group(1))
        print(f"   Done in {duration:.1f}s | Purity: {purity}% | Promiscuity: {promiscuity}%")
        return {
            "Oracle_Layers": oracle_layers,
            "Centroid_Spread": spread,
            "Base_Sigma": sigma,
            "Seed": seed,
            "Purity_%": purity,
            "Promiscuity_%": promiscuity,
            "Coverage": float(coverage_match.group(1))
        }
    else:
        print(f"   Failed to parse output. Error/Output:\n{result.stderr[:200]}\n{output[:200]}")
        return None

def main():
    # Grab the SLURM_ARRAY_TASK_ID passed from the bash script
    if len(sys.argv) != 2:
        print("Usage: python tune_landscape_array.py <task_id>")
        sys.exit(1)
        
    task_id = int(sys.argv[1])
    
    n_train = 100000  # Miniaturized for tuning!
    n_families = 1000
    n_classes = 100
    seed = 42
    
    architectures = ["512,256", "1024,512", "2048,1024", "1024,1024,512"]
    spreads = [10.0, 15.0, 20.0]
    sigmas = [2.0, 4.0, 6.0]
    
    # Generate the 36 possible combinations
    configs = list(itertools.product(architectures, spreads, sigmas))
    
    # Select ONLY the configuration assigned to this specific node
    arch, spread, sigma = configs[task_id]
    
    # Run just this one experiment
    metrics = run_experiment(arch, spread, sigma, seed, n_train, n_families, n_classes)
    
    if metrics:
        df = pd.DataFrame([metrics])
        # Save to a unique CSV to avoid multiple nodes writing to the same file
        df.to_csv(f"tuning_result_{task_id}.csv", index=False)

if __name__ == "__main__":
    main()
