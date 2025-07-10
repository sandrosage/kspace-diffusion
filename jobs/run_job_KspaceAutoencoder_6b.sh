#!/bin/bash -l
#SBATCH  --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --job-name=logs/KspaceAutoencoder-16l-4d-6b-32c
#SBATCH --export=NONE
#SBATCH --mail-user=sandro.sage@fau.de 
#SBATCH --mail-type=ALL

unset SLURM_EXPORT_ENV
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
module load python/3.12-conda
conda activate kdiff
python train_first_stage.py --config configs/KspaceAutoencoder-16l-4d-6b-32c.yaml