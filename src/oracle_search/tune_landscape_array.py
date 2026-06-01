import itertools
import re
import subprocess
import sys
import time

import pandas as pd


def run_experiment(
    oracle_layers,
    sigma,
    seed,
    n_train=1000000,
    n_families=10000,
    n_classes=1000,
):
    # Lock spread to 1.0 and use the default gaussian topology per Dr. Hoarfrost
    cmd = [
        "python",
        "src/generate_simulation.py",
        "--mode",
        "source",
        "--n_train",
        str(n_train),
        "--n_families",
        str(n_families),
        "--n_classes",
        str(n_classes),
        "--oracle_layers",
        oracle_layers,
        "--centroid_spread",
        "1.0",
        "--base_sigma",
        str(sigma),
        "--topology",
        "gaussian",
        "--seed",
        str(seed),
    ]

    print(f"-> Testing: Oracle [{oracle_layers}] | Sigma [{sigma}]")
    start_time = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    output = result.stdout

    purity_match = re.search(r"Within-family purity:\s+([\d.]+)%", output)
    promiscuity_match = re.search(r"Family promiscuity:\s+([\d.]+)%", output)
    coverage_match = re.search(
        r"Class coverage:\s+([\d.]+)", output
    )  # <-- Added

    if purity_match and promiscuity_match and coverage_match:
        purity = float(purity_match.group(1))
        promiscuity = float(promiscuity_match.group(1))
        coverage = float(coverage_match.group(1))  # <-- Added

        print(
            f"   Done in {duration:.1f}s | Purity: {purity}% | Promiscuity: {promiscuity}% | Coverage: {coverage}"
        )
        return {
            "Oracle_Layers": oracle_layers,
            "Base_Sigma": sigma,
            "Purity_%": purity,
            "Promiscuity_%": promiscuity,
            "Coverage": coverage,  # <-- Added
        }
    else:
        print(f"Failed to parse output. Error:\n{result.stderr[:1000]}")
        return None


def main():
    if len(sys.argv) != 2:
        sys.exit(1)

    task_id = int(sys.argv[1])

    # The Professor's Parameters
    architectures = ["512,256", "1024,512", "2048,1024", "1024,1024,512"]
    sigmas = [0.1, 0.3, 0.5, 0.7, 1.0, 1.4]

    configs = list(itertools.product(architectures, sigmas))

    # 4 architectures * 6 sigmas = 24 combinations
    if task_id >= len(configs):
        print("Task ID out of bounds.")
        return

    arch, sigma = configs[task_id]
    metrics = run_experiment(arch, sigma, seed=42)

    if metrics:
        df = pd.DataFrame([metrics])
        df.to_csv(f"tuning_result_{task_id}.csv", index=False)


if __name__ == "__main__":
    main()
