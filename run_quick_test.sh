#!/bin/bash
export LD_LIBRARY_PATH=/work/ah2lab/LiamK/conda_envs/plm_dynamics/lib:$LD_LIBRARY_PATH

# Notice -log is moved BEFORE 'run'
nextflow -log nextflow_quick_test.log run main.nf -profile standard -params-file configs/parity_test.yaml
