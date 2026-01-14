#!/bin/bash
#SBATCH --job-name=sweep_lookahead_gpqa              # Job name
#SBATCH --output="/home/USER_ID/slurm_logs/%A.out"       # Standard output log
#SBATCH --error="/home/USER_ID/slurm_logs/%A.err"         # Standard error log
#SBATCH --ntasks=1                            # Number of tasks (1 process)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --gres=gpu:2                        # Number of GPUs to allocate
##SBATCH --constraint="gpu80"
#SBATCH --time=6:00:00                        # Time limit (24 hours max)
#SBATCH --mem=20G                            # Memory allocation (adjust as needed)
#SBATCH --mail-user=USERNAME@SCHOOL.edu  # Your email
#SBATCH --mail-type=ALL  # Options: BEGIN, END, FAIL, REQUEUE, TIME_LIMIT, etc.

#SBATCH --partition=PARTITION
#SBATCH --account=account

# CLUSTER="local_cluster"
CLUSTER="shared_cluster"

# initialization: set environment variables based on the cluster
if [ "$CLUSTER" = "local_cluster" ]; then
    DATA_DIR="/home/USERNAME/data2"
    DLLM_DIR="/data2/USERNAME/Fast_dLLM_v2_1.5B"
    source /data2/USERNAME/miniconda3/etc/profile.d/conda.sh
elif [ "$CLUSTER" = "shared_cluster" ]; then
    DATA_DIR="/scratch/gpfs/local_cluster/USER_ID/data"
    DLLM_DIR="/home/USER_ID/data/Fast_dLLM_v2_1.5B"
    export HF_HOME="/scratch/gpfs/local_cluster/USER_ID/hf_cache"
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    source /scratch/gpfs/local_cluster/USER_ID/miniconda3/etc/profile.d/conda.sh
    nvidia-smi
else
    echo "Error: CLUSTER must be either 'local_cluster' or 'shared_cluster'"
    exit 1
fi
conda activate vllm_dllm

OUTPUT_DIR="${DATA_DIR}/failfast"

TARGET_MODEL="Qwen/Qwen2.5-32B-Instruct"
# TARGET_MODEL="Qwen/Qwen2.5-14B-Instruct"
DATASETS=("gpqa")  #  "aime"
NUM_QUESTIONS=30
DRAFTER_THRESHOLDS=(0.05)  # fail fast
SWEEP_lowconf_threshold=(0.35 0.4 0.45)  # 0.2, 0.25, 0.3, 0.35, 0.4, 0.45
SWEEP_max_spec_len=(35 40 45 50 55 60)  # win big!
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

    python ../failfast.py \
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

