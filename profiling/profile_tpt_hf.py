# %%
import os
import time
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# Configuration
# ----------------------------
# draft_model_name = "Qwen/Qwen2.5-7B-Instruct"
draft_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
dllm_name = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
dataset_name = "aime"
device = "cuda" if torch.cuda.is_available() else "cpu"
n = 512  # number of new tokens to generate
num_problems = 5  # adjust as needed

# ----------------------------
# Load models and tokenizers
# ----------------------------
print(f"Loading models on {device}...\n")

print(f"→ Loading draft model: {draft_model_name}")
draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_name,
    torch_dtype="auto",
    device_map="auto",
)

print(f"→ Loading dllm model: {dllm_name}")
dllm_tokenizer = AutoTokenizer.from_pretrained(dllm_name, trust_remote_code=True)
dllm = AutoModelForCausalLM.from_pretrained(
    dllm_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

# ----------------------------
# Load dataset
# ----------------------------
if dataset_name == "aime":
    dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
elif dataset_name == "math":
    dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
elif dataset_name == "gpqa":
    if os.getenv("HF_HUB_OFFLINE", "0") == "1":
        dataset = load_from_disk("/scratch/gpfs/USER_ID/hf_cache/datasets/gpqa")
    else:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
else:
    raise NotImplementedError

# ----------------------------
# System prompt
# ----------------------------
system_prompt = """
Solve the following math problem efficiently and clearly. Please reason step by step, 
separate logical reasoning steps with two newline characters (\\n\\n), and put your final answer within \\boxed{{}}.
Problem: {problem}
"""

# ----------------------------
# Profiling helper
# ----------------------------
def profile_model(model, tokenizer, name, dataset, num_problems, n):
    print(f"\nProfiling {name}...")
    total_time = 0.0
    total_output_tokens = 0

    for problem_id in range(num_problems):
        if dataset_name == "aime":
            problem = dataset["problem"][problem_id]
        elif dataset_name == "math":
            problem = dataset["problem"][problem_id]
        elif dataset_name == "gpqa":
            problem = dataset["Question"][problem_id]
        else:
            raise NotImplementedError

        prompt = system_prompt.format(problem=problem)
        model_inputs = tokenizer(prompt, return_tensors="pt").to(device)

        start_time = time.perf_counter()
        if "Qwen" in name:
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=n,
                do_sample=False,  # greedy decoding
            )
        elif "Fast_dLLM_v2_1.5B" in name:
            generated_ids = dllm.generate(
                **model_inputs,
                max_new_tokens=n,  # NOTE: setting this to 8 will not lead to new tokens hmm
                small_block_size=8,
                threshold=0.9,
                # use greedy decoding, not sampling
                do_sample=False,
            )
        else:
            raise NotImplementedError
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        input_len = model_inputs["input_ids"].shape[1]
        output_len = generated_ids.shape[1] - input_len
        total_time += elapsed
        total_output_tokens += output_len

        print(f"[{problem_id}] Generated {output_len} tokens in {elapsed:.2f}s "
              f"({elapsed / output_len * 1000:.2f} ms/token)")

    avg_tpt = (total_time * 1000) / total_output_tokens
    print(f"\n{name} → Average TPT: {avg_tpt:.2f} ms/token")
    return avg_tpt

# ----------------------------
# Run profiling
# ----------------------------
tpt_draft = profile_model(draft_model, draft_tokenizer, draft_model_name, dataset, num_problems, n)
tpt_dllm = profile_model(dllm, dllm_tokenizer, dllm_name, dataset, num_problems, n)

print("\n=======================")
print(f"Qwen2.5-1.5B-Instruct: {tpt_draft:.2f} ms/token")
print(f"Fast_dLLM_v2_1.5B:     {tpt_dllm:.2f} ms/token")
print("=======================")

# %%
