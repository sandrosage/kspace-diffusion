#!/bin/bash -l
#SBATCH  --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --job-name=logs/LDM_resume
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
module load python/3.12-conda
conda activate kdiff
python resume_train_ldm.py --config configs/LDM-128-WeightedSSIMKspaceAutoencoderKL.yaml  --id 48t17rzs