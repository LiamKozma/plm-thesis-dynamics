#!/bin/bash
#SBATCH --job-name=w_dist_grid
#SBATCH --partition=hugemem_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24         # 24 parallel workers (3x faster!)
#SBATCH --mem=500gb               # Snag 1.9 TB of the available 2 TB
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# 1. Load your specific environment
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics

# 2. Define your paths (CORRECTED)
WORK_DIR="/work/ah2lab/LiamK/plm-thesis-dynamics"
SCRATCH_DIR="/scratch/ah2lab/LiamK/w_dist_grid_${SLURM_JOB_ID}"

# 3. Create a safe, temporary workspace on the high-speed scratch drive
mkdir -p ${SCRATCH_DIR}
cd ${SCRATCH_DIR}

# 4. Copy the python scripts over from your work directory (CORRECTED)
cp ${WORK_DIR}/src/oracle_search/generate_simulation.py .
cp ${WORK_DIR}/src/oracle_search/metrics.py .
cp ${WORK_DIR}/src/oracle_search/run_massive_grid.py .

echo "Starting Wasserstein Grid Execution in ${SCRATCH_DIR}..."

# 5. Execute the Python script
python run_massive_grid.py

# 6. Rescue the results back to your work directory and clean up
echo "Job complete. Moving results back to /work..."
cp massive_wasserstein_grid.csv ${WORK_DIR}/results/
cd ${WORK_DIR}

# Be a good cluster citizen and delete the temporary scratch folder
rm -rf ${SCRATCH_DIR}
