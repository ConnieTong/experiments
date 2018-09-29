#!/bin/bash -l
#SBATCH --gres=gpu:1 -t 00:20:00 --mem=10G -c 3
#SBATCH --constraint='kepler|pascal|volta'

module load anaconda2
python -u write_predictions.py