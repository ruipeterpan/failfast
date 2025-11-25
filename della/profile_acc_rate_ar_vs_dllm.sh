#!/bin/bash
#SBATCH --job-name=profile_acc_rate_aime_dllm              # Job name
#SBATCH --output="/home/rp2773/slurm_logs/%A.out"       # Standard output log
#SBATCH --error="/home/rp2773/slurm_logs/%A.err"         # Standard error log
#SBATCH --ntasks=1                            # Number of tasks (1 process)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --gres=gpu:2                        # Number of GPUs to allocate
##SBATCH --constraint="gpu80"
#SBATCH --time=3:00:00                        # Time limit (24 hours max)
#SBATCH --mem=20G                            # Memory allocation (adjust as needed)
#SBATCH --mail-user=ruipan@princeton.edu  # Your email
#SBATCH --mail-type=ALL  # Options: BEGIN, END, FAIL, REQUEUE, TIME_LIMIT, etc.
##SBATCH --partition=pli
##SBATCH --account=specreason
#SBATCH --partition=pli-lc
#SBATCH --account=ravi-group

# CLUSTER="ravi"
CLUSTER="della"

# initialization: set environment variables based on the cluster
if [ "$CLUSTER" = "ravi" ]; then
    DATA_DIR="/home/ruipan/data2"
    DLLM_DIR="/data2/ruipan/Fast_dLLM_v2_1.5B"
elif [ "$CLUSTER" = "della" ]; then
    DATA_DIR="/scratch/gpfs/RAVIAN/rp2773/data"
    DLLM_DIR="/home/rp2773/data/Fast_dLLM_v2_1.5B"
    export HF_HOME="/scratch/gpfs/RAVIAN/rp2773/hf_cache"
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    source /scratch/gpfs/RAVIAN/rp2773/miniconda3/etc/profile.d/conda.sh
    nvidia-smi
else
    echo "Error: CLUSTER must be either 'ravi' or 'della'"
    exit 1
fi
conda activate vllm_dllm
OUTPUT_DIR="${DATA_DIR}/diffspec"

DATASET_NAME="aime"
# DATASET_NAME="math"

# MODEL_TYPE="ar"
MODEL_TYPE="dllm"

NUM_QUESTIONS=30

python ../profiling/profile_acc_rate.py \
    --dataset_name "${DATASET_NAME}" \
    --model_type "${MODEL_TYPE}" \
    --num_questions "${NUM_QUESTIONS}" > "${OUTPUT_DIR}/logs/acc_rate_${DATASET_NAME}_${MODEL_TYPE}.txt"

