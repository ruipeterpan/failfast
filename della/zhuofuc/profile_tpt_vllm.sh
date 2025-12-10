#!/bin/bash
#SBATCH --job-name=profile_tpt_vllm              # Job name
#SBATCH --output="./slurm_logs/profile_tpt_vllm_%A.out"       # Standard output log
#SBATCH --error="./slurm_logs/profile_tpt_vllm_%A.err"         # Standard error log
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
12.6
vllm_dllm
1
EOF

# initialization: set environment variables
nvidia-smi

cd $SCRATCH/diffspec_private

OUTPUT_DIR="./outputs"

# TARGET_MODEL="Qwen/Qwen2.5-32B-Instruct"
# TARGET_MODEL="Qwen/Qwen2.5-14B-Instruct"
TARGET_MODEL="Qwen/Qwen2.5-7B-Instruct"
TP_SIZE=2
NUM_QUESTIONS=30
MAX_NEW_TOKENS=1024
export VLLM_PORT=$(( ( 10000 + (SLURM_JOB_ID % 50000) * 16 ) % 65536 ))

# Function to check if a server is ready
wait_for_server() {
    local port=$1
    while true; do
        # Try to connect to the server
        curl -s http://localhost:$port/get_model_info > /dev/null
        if [ $? -eq 0 ]; then
            echo "Server on port $port is ready!"
            break
        else
            echo "Waiting for server on port $port to start..."
            sleep 10  # Wait 10 seconds before retrying
        fi
    done
}

vllm serve "$TARGET_MODEL" --dtype auto -tp "$TP_SIZE" --max_model_len 4096 --gpu-memory-utilization 0.95 --port $VLLM_PORT --enforce-eager &
VLLM_BASE_PID=$!
wait_for_server $VLLM_PORT
nvidia-smi

timestamp=$(date +"%Y_%m_%d_%H_%M")

echo "Profiling TPT for vLLM on model ${TARGET_MODEL} ..."
python profiling/profile_tpt_vllm.py --port $VLLM_PORT --target_model_name "$TARGET_MODEL" --num_questions $NUM_QUESTIONS --max_new_tokens $MAX_NEW_TOKENS --output_file "${OUTPUT_DIR}/logs/${timestamp}_tpt_vllm_${TARGET_MODEL//\//_}.log"
