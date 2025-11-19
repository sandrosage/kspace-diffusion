#!/bin/bash -l
#SBATCH  --gres=gpu:a40:8
#SBATCH --partition=a40
#SBATCH --time=24:00:00
#SBATCH --job-name=logs/eval_ours
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
module load python/3.12-conda
conda activate kdiff 
python run_pretrained_ours.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --timesteps 100 
python run_pretrained_ours.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --timesteps 100 --accelerations 8 
python run_pretrained_ours.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --timesteps 100 --mask_type equispaced
python run_pretrained_ours.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --timesteps 100 --accelerations 8 --mask_type equispaced
python run_pretrained_ours.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --timesteps 200 
python run_pretrained_ours.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --timesteps 200 --accelerations 8 
python run_pretrained_ours.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --timesteps 200 --mask_type equispaced
python run_pretrained_ours.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --timesteps 200 --accelerations 8 --mask_type equispaced
python run_pretrained_ours.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --timesteps 300 
python run_pretrained_ours.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --timesteps 300 --accelerations 8 
python run_pretrained_ours.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --timesteps 300 --mask_type equispaced
python run_pretrained_ours.py  --data_path /home/janus/iwbi-cip-datasets/shared/fastMRI/knee/singlecoil_val --timesteps 300 --accelerations 8 --mask_type equispaced