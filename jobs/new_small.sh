#!/bin/bash -l
#SBATCH  --gres=gpu:a100:2
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --job-name=logs/new_small
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
module load python/3.12-conda
conda activate kdiff 
timeout 23h python resume_train_ldm.py --config cfg/new_small.yaml

# if [[ $? -eq 124 ]]; then
#   sbatch jobs/run_on_alex.sh
# fi