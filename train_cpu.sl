#!/bin/bash

#SBATCH --job-name=train
#SBATCH --output=train.out
#SBATCH --error=train.err


module load Python/3.9.10
source ~/venv_gnn/bin/activate

echo "python3 train.py --batch-size 128 --epochs 25 --lr 0.009 --step-size 10 --gamma 0.95 --hidden-size 140"
python3 src/train.py --batch-size 128 --epochs 25 --lr 0.005 --step-size 10 --gamma 0.95 --hidden-size 140

