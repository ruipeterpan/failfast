#!/bin/bash
#SBATCH --job-name=sweep_dynamic_freq_math_16              # Job name
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

OUTPUT_DIR="${DATA_DIR}/diffspec"

# # actual run
DATASETS=("math")  #  "aime"
NUM_QUESTIONS=30
DRAFTER_THRESHOLDS=(0.05)
SWEEP_CONF_THRESHOLD_FOR_LOWCONF_TOKENS=(0.3 0.35 0.4 0.45)  # 0.2, 0.25, 0.3, 0.35, 0.4, 0.45
SWEEP_NUM_TOKENS_UPPER_BOUND=(25 30 35 40 45 50)
SWEEP_NUM_TOKENS_TO_INCREMENT=(5 7 10)

timestamp=$(date +"%Y_%m_%d_%H_%M")  # equivalent of datetime.now().strftime("%Y_%m_%d_%H_%M") in python
echo "timestamp: ${timestamp}"

for DATASET_NAME in "${DATASETS[@]}"; do
    python ../sweep_dynamic_frequency_exploration.py \
        --dataset_name "${DATASET_NAME}" \
        --output_dir "${OUTPUT_DIR}" \
        --dllm_dir "${DLLM_DIR}" \
        --num_questions "${NUM_QUESTIONS}" \
        --veri_freq 16 \
        --drafter_thresholds "${DRAFTER_THRESHOLDS[@]}" \
        --sweep_conf_threshold_for_lowconf_tokens "${SWEEP_CONF_THRESHOLD_FOR_LOWCONF_TOKENS[@]}" \
        --sweep_num_tokens_upper_bound "${SWEEP_NUM_TOKENS_UPPER_BOUND[@]}" \
        --sweep_num_tokens_to_increment "${SWEEP_NUM_TOKENS_TO_INCREMENT[@]}" \
        --log_level INFO \
        --overwrite \
        --run_ar > "${OUTPUT_DIR}/logs/${timestamp}_${DATASET_NAME}.ansi" 2>&1
done

        # --read_pickle \
        # 

