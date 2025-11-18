#!/bin/bash -l
#SBATCH  --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --job-name=logs/eval_ours_100_300_50_v1
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
module load python/3.12-conda
conda activate kdiff 
python run_pretrained_ours.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --timesteps 100
python run_pretrained_ours.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --timesteps 300
python run_pretrained_ours.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --timesteps 50 --cgs
