#!/bin/bash

#SBATCH --job-name=graph_build
#SBATCH --array=1002-1099
#SBATCH --partition=short
#SBATCH --output=out_%A_%a.log
#SBATCH --error=err_%A_%a.log

module load Python/3.9.10
source ~/venv_gnn/bin/activate

echo "Running on file ID $SLURM_ARRAY_TASK_ID"
python3 build_graph_segmented.py $SLURM_ARRAY_TASK_ID
