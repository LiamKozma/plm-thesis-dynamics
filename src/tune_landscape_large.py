import subprocess
import re
import pandas as pd
import itertools
import time

def run_experiment(oracle_layers, spread, sigma, seed, n_train=1000000, n_families=1000, n_classes=100):
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
    
    # Run the subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    output = result.stdout

    # Parse the metrics using Regex
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
    # Lock in your 10:1 ratio and massive scale
    n_train = 1000000 
    n_families = 1000
    n_classes = 100
    seed = 42
    
    # The Parameter Grid to Sweep
    architectures = ["512,256", "1024,512", "2048,1024", "1024,1024,512"]
    spreads = [10.0, 15.0, 20.0]  # Distance between cluster centers
    sigmas = [2.0, 4.0, 6.0]      # Variance/overlap of the clusters
    
    results = []
    
    print(f"Starting Massive Landscape Tuning ({n_train:,} embeddings per run)...")
    for arch, spread, sigma in itertools.product(architectures, spreads, sigmas):
        metrics = run_experiment(arch, spread, sigma, seed, n_train, n_families, n_classes)
        if metrics:
            results.append(metrics)
            
    # Save the full results matrix
    df = pd.DataFrame(results)
    df.to_csv("massive_landscape_tuning.csv", index=False)
    
    print("\n" + "="*70)
    print(" TUNING COMPLETE: Top Configurations near targets")
    print(" Targets: Purity (50-70%), Promiscuity (40-60%)")
    print("="*70)
    
    # Filter for the sweet spot
    sweet_spots = df[(df['Promiscuity_%'] >= 35) & (df['Promiscuity_%'] <= 65) & 
                     (df['Purity_%'] >= 45) & (df['Purity_%'] <= 75)]
    
    if not sweet_spots.empty:
        print(sweet_spots.sort_values(by=['Promiscuity_%', 'Purity_%']).to_string(index=False))
    else:
        print("No configurations hit the exact sweet spot. Check massive_landscape_tuning.csv for the closest matches.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
