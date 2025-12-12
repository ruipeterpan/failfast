#!/bin/bash
#SBATCH --job-name=sweep_lookahead_math              # Job name
#SBATCH --output="/home/rp2773/slurm_logs/%A.out"       # Standard output log
#SBATCH --error="/home/rp2773/slurm_logs/%A.err"         # Standard error log
#SBATCH --ntasks=1                            # Number of tasks (1 process)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --gres=gpu:2                        # Number of GPUs to allocate
##SBATCH --constraint="gpu80"
#SBATCH --time=6:00:00                        # Time limit (24 hours max)
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
    source /data2/ruipan/miniconda3/etc/profile.d/conda.sh
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

# OUTPUT_DIR="${DATA_DIR}/diffspec"
OUTPUT_DIR="${DATA_DIR}/diffspec_quicktry"

# TARGET_MODEL="Qwen/Qwen2.5-32B-Instruct"
TARGET_MODEL="Qwen/Qwen2.5-14B-Instruct"
DATASETS=("math")  #  "aime"
NUM_QUESTIONS=30
DRAFTER_THRESHOLDS=(0.05)
SWEEP_lowconf_threshold=(0.4 0.45)  # 0.2, 0.25, 0.3, 0.35, 0.4, 0.45
SWEEP_max_spec_len=(40 45 50 55 60)
SWEEP_incr_len=(10)


for DATASET_NAME in "${DATASETS[@]}"; do
    timestamp=$(date +"%Y_%m_%d_%H_%M")  # equivalent of datetime.now().strftime("%Y_%m_%d_%H_%M") in python
    echo "Dataset ${DATASET_NAME} timestamp: ${timestamp}"
    logfile="${OUTPUT_DIR}/logs/${timestamp}_${DATASET_NAME}.ansi"

    while [ -f "$logfile" ]; do
        echo "Log file ${logfile} exists. Sleeping 60 seconds and retaking timestamp..."
        sleep 60
        timestamp=$(date +"%Y_%m_%d_%H_%M")
        logfile="${OUTPUT_DIR}/logs/${timestamp}_${DATASET_NAME}.ansi"
    done

    python ../failfast_quicktry.py \
        --dataset_name "${DATASET_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --target_model_name "${TARGET_MODEL}" \
        --dllm_dir "${DLLM_DIR}" \
        --num_questions "${NUM_QUESTIONS}" \
        --spec_len 10 \
        --max_new_tokens 1024 \
        --drafter_thresholds "${DRAFTER_THRESHOLDS[@]}" \
        --sweep_lowconf_threshold "${SWEEP_lowconf_threshold[@]}" \
        --sweep_max_spec_len "${SWEEP_max_spec_len[@]}" \
        --sweep_incr_len "${SWEEP_incr_len[@]}" \
        --log_level INFO \
        --overwrite \
        --run_ar > "$logfile" 2>&1
done

        # --read_pickle \
        # 

