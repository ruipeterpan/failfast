#!/bin/bash
#SBATCH --job-name=sweep_diffullama             # Job name
#SBATCH --output="./slurm_logs/sweep_diffullama_%A.out"       # Standard output log
#SBATCH --error="./slurm_logs/sweep_diffullama_%A.err"         # Standard error log
#SBATCH --ntasks=1                            # Number of tasks (1 process)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --gres=gpu:4                        # Number of GPUs to allocate
##SBATCH --constraint="gpu80"
#SBATCH --time=2:00:00                        # Time limit (24 hours max)
#SBATCH --mem-per-cpu=8G                            # Memory allocation (adjust as needed)
#SBATCH --partition=ailab​

source ~/.bashrc

. ~/init.sh <<EOF
-1
vllm_dllm
1
EOF

# initialization: set environment variables
nvidia-smi

cd $SCRATCH/diffspec_private

OUTPUT_DIR="./outputs"

TARGET_MODEL="meta-llama/Llama-2-70b-hf"
DATASETS=("math")  #  "aime"
NUM_QUESTIONS=5
DIFFUSION_STEPS=64
VERI_FREQS=(3 6 9 12 15)
# VERI_FREQS=(3 4 5 6 7 8 9 10 11 12 13 14 15 16)  # 0.9
# VERI_FREQS=(5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)  # 0.05

for DATASET_NAME in "${DATASETS[@]}"; do
    timestamp=$(date +"%Y_%m_%d_%H_%M")  # equivalent of datetime.now().strftime("%Y_%m_%d_%H_%M") in python
    echo "Dataset ${DATASET_NAME} timestamp: ${timestamp}"
    logfile="${OUTPUT_DIR}/logs/${timestamp}_${DATASET_NAME}_diffullama.ansi"

    while [ -f "$logfile" ]; do
        echo "Log file ${logfile} exists. Sleeping 60 seconds and retaking timestamp..."
        sleep 60
        timestamp=$(date +"%Y_%m_%d_%H_%M")
        logfile="${OUTPUT_DIR}/logs/${timestamp}_${DATASET_NAME}_diffullama.ansi"
    done

    for FREQ in "${VERI_FREQS[@]}"; do
        echo "Running FREQ: ${FREQ}, using diffusion steps: ${DIFFUSION_STEPS}"
        python failfast.py \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "${OUTPUT_DIR}" \
            --target_model_name "${TARGET_MODEL}" \
            --num_questions "${NUM_QUESTIONS}" \
            --spec_len "${FREQ}" \
            --max_new_tokens 1024 \
            --diffullama_diffusion_steps "${DIFFUSION_STEPS}" \
            --log_level INFO \
            --run_diffullama_sf \
            --baseline_sweep \
            --overwrite \
            >> "$logfile" 2>&1
    done
done

        # --read_pickle \
        # --overwrite \

