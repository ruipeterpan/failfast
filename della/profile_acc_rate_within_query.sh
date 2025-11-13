#!/bin/bash
#SBATCH --job-name=profile_acc_rate_within_query_math              # Job name
#SBATCH --output="/home/rp2773/slurm_logs/%A.out"       # Standard output log
#SBATCH --error="/home/rp2773/slurm_logs/%A.err"         # Standard error log
#SBATCH --ntasks=1                            # Number of tasks (1 process)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --gres=gpu:2                        # Number of GPUs to allocate
##SBATCH --constraint="gpu80"
#SBATCH --time=4:00:00                        # Time limit (24 hours max)
#SBATCH --mem=20G                            # Memory allocation (adjust as needed)
#SBATCH --mail-user=ruipan@princeton.edu  # Your email
#SBATCH --mail-type=ALL  # Options: BEGIN, END, FAIL, REQUEUE, TIME_LIMIT, etc.
##SBATCH --partition=pli
##SBATCH --account=specreason
#SBATCH --partition=pli-lc
#SBATCH --account=ravi-group

CLUSTER="ravi"
# CLUSTER="della"

# initialization: set environment variables based on the cluster
if [ "$CLUSTER" = "ravi" ]; then
    DATA_DIR="/home/ruipan/data2"
    DLLM_DIR="/data2/ruipan/Fast_dLLM_v2_1.5B"
    source /data2/ruipan/miniconda3/etc/profile.d/conda.sh
elif [ "$CLUSTER" = "della" ]; then
    DATA_DIR="/scratch/gpfs/RAVIAN/rp2773/data"
    DLLM_DIR="/hoome/rp2773/data/Fast_dLLM_v2_1.5B"
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

# # actual run
DATASETS=("math" "aime")
NUM_QUESTIONS=30
# DRAFTER_THRESHOLDS=(0.9 0.7 0.5 0.3 0.1 0.01)
DRAFTER_THRESHOLDS=(0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10 0.05)
# debug
# DATASETS=("math")
# NUM_QUESTIONS=1
# DRAFTER_THRESHOLDS=(0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.30 0.25 0.20 0.15 0.10 0.05)

timestamp=$(date +"%Y_%m_%d_%H_%M")  # equivalent of datetime.now().strftime("%Y_%m_%d_%H_%M") in python

for DATASET_NAME in "${DATASETS[@]}"; do
    python ../profiling/profile_acc_rate_within_query.py \
        --dataset_name "${DATASET_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --dllm_dir "${DLLM_DIR}" \
        --num_questions "${NUM_QUESTIONS}" \
        --log_level INFO \
        --drafter_thresholds "${DRAFTER_THRESHOLDS[@]}" \
        --overwrite \
        --run_ar > "${OUTPUT_DIR}/logs/${timestamp}_${DATASET_NAME}.ansi" 2>&1
done
        # --read_pickle \
        # 
