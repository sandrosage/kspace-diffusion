#!/bin/bash -l
#SBATCH  --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=24:00:00
#SBATCH --job-name=logs/varnet_eval
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
module load python/3.12-conda
conda activate kdiff 
python run_pretrained_varnet.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --state_dict_file knee_leaderboard_state_dict.pt 
python run_pretrained_varnet.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --state_dict_file knee_leaderboard_state_dict.pt --accelerations 8
python run_pretrained_varnet.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --state_dict_file knee_leaderboard_state_dict.pt --mask_type equispaced
python run_pretrained_varnet.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --state_dict_file knee_leaderboard_state_dict.pt --mask_type equispaced --accelerations 8