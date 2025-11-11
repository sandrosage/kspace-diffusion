#!/bin/bash -l
#SBATCH  --gres=gpu:a100:4
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --job-name=logs/new
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1
module load python/3.12-conda
conda activate kdiff 
timeout 23h python resume_train_ldm.py --config cfg/new.yaml --id 37di910m

if [[ $? -eq 124 ]]; then
  sbatch jobs/new.sh
fi