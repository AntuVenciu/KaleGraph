#!/bin/bash

#SBATCH --job-name=graph_build
#SBATCH --array=1000-1099
#SBATCH --partition=short
#SBATCH --output=graph_%A_%a.out
#SBATCH --error=graph_%A_%a.err

module load Python/3.9.10
source ~/venv_gnn/bin/activate

echo "Running on file ID $SLURM_ARRAY_TASK_ID"
python3 src/build_graph_segmented.py $SLURM_ARRAY_TASK_ID
