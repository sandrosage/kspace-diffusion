#!/bin/bash -l
#SBATCH  --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --job-name=logs/evaluate_SVAE
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
module load python/3.12-conda
conda activate kdiff 
python test_first_stage.py --config test_cfg/SAE_KL_3_yes.yaml 
python test_first_stage.py --config test_cfg/SAE_KL_3_yes.yaml --undersampling
python test_first_stage.py --config test_cfg/SAE_KL_3_yes.yaml --undersampling --accelerations 8
python test_first_stage.py --config test_cfg/SAE_KL_4_yes.yaml 
python test_first_stage.py --config test_cfg/SAE_KL_4_yes.yaml --undersampling
python test_first_stage.py --config test_cfg/SAE_KL_4_yes.yaml --undersampling --accelerations 8
python test_first_stage.py --config test_cfg/SAE_KL_5_yes.yaml 
python test_first_stage.py --config test_cfg/SAE_KL_5_yes.yaml --undersampling
python test_first_stage.py --config test_cfg/SAE_KL_5_yes.yaml --undersampling --accelerations 8
python test_first_stage.py --config test_cfg/SAE_KL_3_no.yaml 
python test_first_stage.py --config test_cfg/SAE_KL_3_no.yaml --undersampling
python test_first_stage.py --config test_cfg/SAE_KL_3_no.yaml --undersampling --accelerations 8
python test_first_stage.py --config test_cfg/SAE_KL_4_no.yaml 
python test_first_stage.py --config test_cfg/SAE_KL_4_no.yaml --undersampling
python test_first_stage.py --config test_cfg/SAE_KL_4_no.yaml --undersampling --accelerations 8
python test_first_stage.py --config test_cfg/SAE_KL_5_no.yaml 
python test_first_stage.py --config test_cfg/SAE_KL_5_no.yaml --undersampling
python test_first_stage.py --config test_cfg/SAE_KL_5_no.yaml --undersampling --accelerations 8