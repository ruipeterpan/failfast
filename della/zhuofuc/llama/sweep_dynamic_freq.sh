#!/bin/bash
#SBATCH --job-name=llama_failfast              # Job name
#SBATCH --output="/home/rp2773/slurm_logs/%A.out"       # Standard output log
#SBATCH --error="/home/rp2773/slurm_logs/%A.err"         # Standard error log
#SBATCH --ntasks=1                            # Number of tasks (1 process)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --gres=gpu:1                        # Number of GPUs to allocate
#SBATCH --constraint="gpu80"
#SBATCH --time=4:00:00                        # Time limit (24 hours max)
#SBATCH --mem=20G                            # Memory allocation (adjust as needed)
#SBATCH --mail-user=ruipan@princeton.edu  # Your email
#SBATCH --mail-type=ALL  # Options: BEGIN, END, FAIL, REQUEUE, TIME_LIMIT, etc.
#SBATCH --partition=pli
#SBATCH --account=specreason
##SBATCH --partition=pli-lc
##SBATCH --account=ravi-group

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
    export HF_DATASETS_CACHE="/scratch/gpfs/RAVIAN/rp2773/hf_cache/datasets"
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

OUTPUT_DIR="${DATA_DIR}/failfast_llama"

TARGET_MODEL="meta-llama/Llama-2-13b-hf"
DATASETS=("math" "gpqa" "humaneval")  #  "aime"
NUM_QUESTIONS=30
DIFFUSION_STEPS=(1)
SWEEP_lowconf_threshold=(0.4)  # 0.2, 0.25, 0.3, 0.35, 0.4, 0.45
SWEEP_max_spec_len=(40)
SWEEP_incr_len=(10)


for DATASET_NAME in "${DATASETS[@]}"; do
    timestamp=$(date +"%Y_%m_%d_%H_%M")  # equivalent of datetime.now().strftime("%Y_%m_%d_%H_%M") in python
    echo "Dataset ${DATASET_NAME} timestamp: ${timestamp}"
    logfile="${OUTPUT_DIR}/logs/${timestamp}_${DATASET_NAME}_diffullama_ff.ansi"

    while [ -f "$logfile" ]; do
        echo "Log file ${logfile} exists. Sleeping 60 seconds and retaking timestamp..."
        sleep 60
        timestamp=$(date +"%Y_%m_%d_%H_%M")
        logfile="${OUTPUT_DIR}/logs/${timestamp}_${DATASET_NAME}_diffullama_ff.ansi"
    done

    for DIFFUSION_STEPS in "${DIFFUSION_STEPS[@]}"; do
        python failfast.py \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "${OUTPUT_DIR}" \
            --target_model_name "${TARGET_MODEL}" \
            --num_questions "${NUM_QUESTIONS}" \
            --spec_len 10 \
            --max_new_tokens 1024 \
            --diffullama_diffusion_steps "${DIFFUSION_STEPS}" \
            --sweep_lowconf_threshold "${SWEEP_lowconf_threshold[@]}" \
            --sweep_max_spec_len "${SWEEP_max_spec_len[@]}" \
            --sweep_incr_len "${SWEEP_incr_len[@]}" \
            --log_level INFO \
            --overwrite \
            --diffullama_dir "/home/rp2773/DiffuLLaMA" \
            --diffullama_dynamic > "$logfile" 2>&1
    done
done

        # --read_pickle \
        # 

