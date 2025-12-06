#!/bin/bash
#SBATCH --job-name=sweep_dllm_0.9             # Job name
#SBATCH --output="./slurm_logs/sweep_dllm_0.9_%A.out"       # Standard output log
#SBATCH --error="./slurm_logs/sweep_dllm_0.9_%A.err"         # Standard error log
#SBATCH --ntasks=1                            # Number of tasks (1 process)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --gres=gpu:2                        # Number of GPUs to allocate
##SBATCH --constraint="gpu80"
#SBATCH --time=6:00:00                        # Time limit (24 hours max)
#SBATCH --mem=20G                            # Memory allocation (adjust as needed)
#SBATCH --partition=pli-lc
#SBATCH --account=ravi-group

source ~/.bashrc

. ~/init.sh <<EOF
vllm_dllm
1
EOF

# initialization: set environment variables
DLLM_DIR="/scratch/gpfs/RAVIAN/zhuofuc/Fast_dLLM_v2_1.5B"
nvidia-smi

cd $SCRATCH/diffspec_private

OUTPUT_DIR="./outputs"

# TARGET_MODEL="Qwen/Qwen2.5-32B-Instruct"
TARGET_MODEL="Qwen/Qwen2.5-14B-Instruct"
DATASETS=("mmlu")  #  "aime"
NUM_QUESTIONS=30
DRAFTER_THRESHOLDS=(0.9)
VERI_FREQS=(3 4 5 6 7 8 9 10 11 12 13 14 15 16)  # 0.9
# VERI_FREQS=(5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)  # 0.05

for DATASET_NAME in "${DATASETS[@]}"; do
    timestamp=$(date +"%Y_%m_%d_%H_%M")  # equivalent of datetime.now().strftime("%Y_%m_%d_%H_%M") in python
    echo "Dataset ${DATASET_NAME} timestamp: ${timestamp}"
    logfile="${OUTPUT_DIR}/logs/${timestamp}_${DATASET_NAME}_dllm.ansi"

    while [ -f "$logfile" ]; do
        echo "Log file ${logfile} exists. Sleeping 60 seconds and retaking timestamp..."
        sleep 60
        timestamp=$(date +"%Y_%m_%d_%H_%M")
        logfile="${OUTPUT_DIR}/logs/${timestamp}_${DATASET_NAME}_dllm.ansi"
    done

    for FREQ in "${VERI_FREQS[@]}"; do
        python failfast.py \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "${OUTPUT_DIR}" \
            --target_model_name "${TARGET_MODEL}" \
            --dllm_dir "${DLLM_DIR}" \
            --num_questions "${NUM_QUESTIONS}" \
            --spec_len "${FREQ}" \
            --max_new_tokens 1024 \
            --drafter_thresholds "${DRAFTER_THRESHOLDS[@]}" \
            --log_level INFO \
            --run_dllm_sf \
            --baseline_sweep \
            --overwrite \
            >> "$logfile" 2>&1
    done
done

        # --read_pickle \
        # --overwrite \

