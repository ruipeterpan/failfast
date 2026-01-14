# Instructions on Running Experiments

## Environment Setup

- Transformers: 4.53.3
- vLLM: 0.13.0

We have included a (dated version of our) `environment.yaml`, which has not been throughly tested. Sorry lol! We'll update this soon.

## Fast-dLLM-V2

Both our Fast-dLLM baseline and FailFast uses [Efficient-Large-Model/Fast_dLLM_v2_1.5B](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B) underneath the hood. On top of `Fast_dLLM_v2_1.5B`, we implemented additional generation functions for efficient usage of dLLMs as drafters in speculative decoding.

First, we need to clone the Fast-dLLM weights from HuggingFace, replace the remote, and plug in our own `generate.py`, which includes efficient drafting mechanisms introduced in FailFast. This process consumes ~3GB of disk space.

```
git clone https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B
cd Fast_dLLM_v2_1.5B
# pulls our own generation functions
git remote set-url origin https://github.com/USERNAME/Fast_dLLM_v2_1.5B.git
git pull origin
```

Overview of the different generation methods within Fast-dLLM:
- `generate()`: Fast-dLLM's vanilla implementation for standalone generation. Incorporates optimizations like approximate KV caching and confidence-aware parallel decoding.
- `generate_draft_tokens()`: generates the next `spec_len` tokens. This is used by our Fast-dLLM baseline. It finishes as soon as the next `spec_len` tokens are all unmasked.
- `generate_draft_tokens_arbitrary_length()`: implements Algorithm 1 in FailFast.

## Running FailFast

We provide bash scripts (that are also Slurm job scripts) for running FailFast on our own hardware setup. You might need to tweak them for them to be runnable on custom setups.

- `scripts/sweep_ar_drafter_freqs.sh`: Used for parameter sweeps for AR drafters
- `scripts/sweep_dllm_drafter_freqs.sh`: Used for parameter sweeps for Fast-dLLM drafters
- `scripts/sweep_dynamic_freqs.sh`: Used for parameter sweeps for FailFast.
- Other scripts are either used for microbenchmarks or have been deprecated.

All three scripts invoke `failfast.py`. Its primary purpose is to evaluate the efficiency, acceptance rates, and speedups of different "drafter" models (Auto-Regressive vs. dLLM) when paired with a larger target model (e.g., Qwen-32B) across various reasoning datasets.

This script is a benchmarking tool for **Speculative Decoding (SD)**, designed to evaluate the performance of different drafter models (Standard Auto-Regressive vs. Fast-dLLM) against a large target model.

### Core Script Logic
1.  **Initialization:** Loads a large target model (e.g., Qwen-32B) and the selected drafter models (e.g., Qwen-1.5B or Fast-dLLM).
2.  **Dataset Processing:** Iterates through a specified number of questions from reasoning datasets (MATH, GSM8K, etc.).
3.  **Speculative Loop:** For each token generation sequence:
    *   **Propose:** The drafter generates a block of "speculative" tokens.
    *   **Verify:** The target model performs a single parallel forward pass to validate the proposed tokens.
    *   **Accept/Reject:** The script identifies the first mismatch, accepts the valid prefix, and appends a "bonus" or corrected token from the target model.
4.  **Adaptive Logic:** Optionally employs "Dynamic Frequency" to adjust the speculation length based on drafter confidence and previous rejection history.
5.  **Metrics & Export:** Calculates acceptance rates and speedups (mapped to A6000 hardware latency profiles), then saves the data to pickle files and generates visualization plots.

### Key Functionalities
*   **Multi-Strategy Drafting:** Supports standard Auto-Regressive (AR) drafting and block-based dLLM drafting.
*   **KV Cache Management:** Includes logic to reuse drafter KV caches across verification rounds to minimize redundant computation.
*   **Hardware Profiling:** Models End-to-End (E2E) latency by applying real-world TPT (Time Per Token) constants for various GPUs.
*   **Trajectory Analysis:** Provides detailed logging of "acceptance trajectories" to visualize where drafters succeed or fail during long-form reasoning.
*   **Parameter Sweeping:** Automatically tests ranges of confidence thresholds and speculation lengths to find optimal configurations.

### Parsing the output of failfast.py

See `parse_log.py`. It orders the drafter configs from most speedup to least speedup. Note that the average is "average of average acceptance rates of each query" and is technically not a true average.

We also open-source all the plotting scripts under `/plotting`.

## EAGLE-3

We use [SpecForge](https://github.com/sgl-project/SpecForge) to pretrain our own EAGLE-3 weights. We are still in the process of training more EAGLE-3 weights, so we only 
include the commands we used to profile EAGLE-3's speedup. We will add more documentations and release EAGLE-3 weights soon.

```
# vanilla inference
vllm serve Qwen/Qwen2.5-7B-Instruct --dtype auto -tp 2 --max_model_len 2048 --gpu-memory-utilization 0.8 --port 30000

# EAGLE-3
vllm serve Qwen/Qwen2.5-7B-Instruct --dtype auto -tp 2 --max_model_len 2048 --gpu-memory-utilization 0.9 --port 30000 --speculative_config '{"model": "/path/to/eagle/weights", "draft_tensor_parallel_size": 1, "num_speculative_tokens": 5, "method": "eagle3"}'

# profiling
python profiling/profile_tpt_vllm_eagle3.py
```
