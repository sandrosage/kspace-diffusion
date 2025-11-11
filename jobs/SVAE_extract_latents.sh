#!/bin/bash -l
#SBATCH  --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=4:00:00
#SBATCH --job-name=logs/SVAE_extract_latents
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
module load python/3.12-conda
conda activate kdiff 
python extract_latent.py --ckpt "WeightedSSIMKspaceAutoencoder/8fvbciuk/checkpoints/WeightedSSIMKspaceAutoencoder-epoch=36.ckpt" --partition "train"
python extract_latent.py --ckpt "WeightedSSIMKspaceAutoencoder/8fvbciuk/checkpoints/WeightedSSIMKspaceAutoencoder-epoch=36.ckpt" --partition "val"