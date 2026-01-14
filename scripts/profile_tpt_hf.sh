#!/bin/bash
#SBATCH --job-name=profile_tpt_hf               # Job name
#SBATCH --output="/home/USER_ID/slurm_logs/%A.out"       # Standard output log
#SBATCH --error="/home/USER_ID/slurm_logs/%A.err"         # Standard error log
#SBATCH --ntasks=1                            # Number of tasks (1 process)
#SBATCH --cpus-per-task=8                     # Number of CPU cores per task
#SBATCH --gres=gpu:2                        # Number of GPUs to allocate
##SBATCH --constraint="gpu80"
#SBATCH --time=3:00:00                        # Time limit (24 hours max)
#SBATCH --mem=100G                            # Memory allocation (adjust as needed)
#SBATCH --mail-user=USERNAME@SCHOOL.edu  # Your email
#SBATCH --mail-type=ALL  # Options: BEGIN, END, FAIL, REQUEUE, TIME_LIMIT, etc.

#SBATCH --partition=PARTITION
#SBATCH --account=account

# CLUSTER="local_cluster"
CLUSTER="shared_cluster"

# initialization: set environment variables based on the cluster
if [ "$CLUSTER" = "local_cluster" ]; then
    DATA_DIR="/home/USERNAME/data2"
elif [ "$CLUSTER" = "shared_cluster" ]; then
    DATA_DIR="/scratch/gpfs/local_cluster/USER_ID/data"
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

DATASET_NAME="aime"
NUM_QUESTIONS=30
OUTPUT_DIR="${DATA_DIR}/failfast"


python ../profiling/profile_tpt_hf.py > "${OUTPUT_DIR}/logs/profile_tpt_hf.log"

