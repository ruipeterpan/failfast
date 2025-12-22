# %%
import os
import sys
import time
import argparse
import logging
import httpx
from transformers import AutoTokenizer

# Add root directory to path to import utils
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, root_dir)
from utils import populate_dataset, format_problem_and_options, get_first_user_msg

"""
# sglang version: 0.5.6.post2

# vanilla generation
python -m sglang.launch_server  --model Qwen/Qwen2.5-7B-Instruct --dtype float16  --port 30000  --mem-fraction 0.7  --tp-size 2  --cuda-graph-max-bs 16 --cuda-graph-bs {1,2,3,4}

# EAGLE2
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server  --model Qwen/Qwen2.5-7B-Instruct  --speculative-algorithm EAGLE  --speculative-draft-model Tengyunw/qwen_2.5_7b_instruct_eagle2_v0  --speculative-num-steps 5  --speculative-eagle-topk 8  --speculative-num-draft-tokens 64  --dtype float16  --port 30000  --mem-fraction 0.7  --tp-size 2  --cuda-graph-max-bs 16 --cuda-graph-bs {1,2,3,4}
"""

parser = argparse.ArgumentParser(description="Profiles the vLLM TPT of a model on all datasets.")
parser.add_argument("--port", type=int, default=30000, help="Port to use for the vLLM server")
parser.add_argument("--output_file", type=str, help="Name of the output file")
parser.add_argument("--target_model_name", type=str, default="Qwen/Qwen2.5-32B-Instruct", 
                    help="Name of the base model to use")
parser.add_argument("--num_questions", type=int, default=1,
                    help="Number of questions to run profiling on")
parser.add_argument("--max_new_tokens", type=int, default=1024,
                    help="Max new tokens from the target model")
parser.add_argument("--log_level",
                    type=str,
                    default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Set the logging level")
args, _ = parser.parse_known_args()
logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%m%d %H:%M:%S",
    # datefmt="%m%d",
)

try:
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
except Exception as e:
    logging.warning(f"Could not load tokenizer locally for {args.target_model_name}. Prompt will be sent raw. Error: {e}")
    exit(1)

GEN_ENDPOINT = f"http://localhost:{args.port}/generate" 

transport = httpx.HTTPTransport()
client = httpx.Client(
    transport=transport,
    base_url=f"http://localhost:{args.port}", # Use base URL
    timeout=600,
)

tpts = {}
for dataset_name in ["math", "aime", "gpqa", "mmlu", "humaneval"]:
# for dataset_name in ["humaneval"]:
    args.dataset_name = dataset_name
    dataset = populate_dataset(args)

    # %%
    total_time = 0
    total_output_tokens = 0

    for problem_id in range(args.num_questions):
        raw_data = format_problem_and_options(args, problem_id)
        
        messages = [
            {"role": "user", "content": get_first_user_msg(args, raw_data)},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True
        )
        
        start_time = time.perf_counter()
        payload = {
            "input_ids": input_ids, # SGLang uses 'input_ids'
            "sampling_params": {
                "max_new_tokens": args.max_new_tokens, # Note: SGLang prefers 'max_new_tokens'
                "temperature": 0.0,
            },
            "return_logprob": False,
        }
        resp = client.post(GEN_ENDPOINT, json=payload)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.perf_counter() - start_time  # seconds
        
        num_output_tokens = data["meta_info"]["completion_tokens"]
        
        total_time += elapsed
        total_output_tokens += num_output_tokens

    avg_tpt = (total_time * 1000) / total_output_tokens
    print(f"Dataset {dataset_name}: Average TPT: {avg_tpt:.2f}={total_time * 1000:.2f}/{total_output_tokens:.2f} ms/token")
    tpts[dataset_name] = avg_tpt

with open(args.output_file, "w") as f:
    f.write("Dataset: Average TPT:\n")
    for dataset_name, tpt in tpts.items():
        f.write(f"{dataset_name}: {tpt:.2f} ms/token\n")

# %%
