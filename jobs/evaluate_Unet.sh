#!/bin/bash -l
#SBATCH  --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --job-name=logs/evaluate_Unet
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
module load python/3.12-conda
conda activate kdiff 
python test_first_stage.py --config test_cfg/UnetWN.yaml 
python test_first_stage.py --config test_cfg/UnetWN.yaml --undersampling
python test_first_stage.py --config test_cfg/UnetWN.yaml --undersampling --accelerations 8
python test_first_stage.py --config test_cfg/UnetWON.yaml 
python test_first_stage.py --config test_cfg/UnetWON.yaml --undersampling
python test_first_stage.py --config test_cfg/UnetWON.yaml --undersampling --accelerations 8
python test_first_stage.py --config test_cfg/UnetWONO.yaml 
python test_first_stage.py --config test_cfg/UnetWONO.yaml --undersampling
python test_first_stage.py --config test_cfg/UnetWONO.yaml --undersampling --accelerations 8