#!/bin/bash -l
#SBATCH  --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=3:00:00
#SBATCH --job-name=logs/Latent-extraction
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
module load python/3.12-conda
conda activate kdiff
python extract_latent.py