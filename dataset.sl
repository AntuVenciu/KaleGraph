#!/bin/bash

#SBATCH --job-name=graph_build
#SBATCH --array=1000-1033 # 1033 for 1 / 3 of the dataset for test ~ 150k graphs
#SBATCH --partition=short
#SBATCH --output=graph_%A_%a.out
#SBATCH --error=graph_%A_%a.err

# command to remove all files (they are too many not to do it in parallel)
#rm /meg/data1/shared/subprojects/cdch/ext-venturini_a/GNN/NoPileUpMC/file0$SLURM_ARRAY_TASK_ID.npz

module load Python/3.9.10
source ~/venv_gnn/bin/activate

echo "Running on file ID $SLURM_ARRAY_TASK_ID"
python3 src/build_graph_segmented.py $SLURM_ARRAY_TASK_ID
