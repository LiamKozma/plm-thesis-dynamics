import itertools
import re
import subprocess
import sys
import time
import pandas as pd

def run_experiment(topology, sigma, seed, n_train=1000000, n_families=1000, n_classes=100):
    oracle_layers = "512,256"
    spread = 1.0 # Set to 1.0 since the bounding boxes handle the scale
    
    cmd = [
        "python", "src/generate_simulation.py",
        "--mode", "source",
        "--n_train", str(n_train),
        "--n_families", str(n_families),
        "--n_classes", str(n_classes),
        "--oracle_layers", oracle_layers,
        "--centroid_spread", str(spread),
        "--base_sigma", str(sigma),
        "--topology", topology,
        "--seed", str(seed)
    ]
    
    print(f"-> Testing: {topology.upper()} | Sigma [{sigma}]")
    start_time = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    output = result.stdout

    purity_match = re.search(r"Within-family purity:\s+([\d.]+)%", output)
    promiscuity_match = re.search(r"Family promiscuity:\s+([\d.]+)%", output)

    if purity_match and promiscuity_match:
        purity = float(purity_match.group(1))
        promiscuity = float(promiscuity_match.group(1))
        print(f"   Done in {duration:.1f}s | Purity: {purity}% | Promiscuity: {promiscuity}%")
        return {
            "Topology": topology,
            "Base_Sigma": sigma,
            "Purity_%": purity,
            "Promiscuity_%": promiscuity
        }
    else:
        print("Failed to parse output.")
        return None

def main():
    if len(sys.argv) != 2:
        sys.exit(1)
        
    task_id = int(sys.argv[1])
    
    topologies = ['hypercube', 'hypersphere', 'projection']
    sigmas = [0.01, 0.05, 0.10, 0.50] # Testing tiny decimal sigmas
    
    configs = list(itertools.product(topologies, sigmas))
    
    # We only have 12 combinations now (3 topologies * 4 sigmas)
    if task_id >= len(configs):
        print("Task ID out of bounds for this grid.")
        return
        
    topology, sigma = configs[task_id]
    metrics = run_experiment(topology, sigma, seed=42)
    
    if metrics:
        df = pd.DataFrame([metrics])
        df.to_csv(f"tuning_result_{task_id}.csv", index=False)
