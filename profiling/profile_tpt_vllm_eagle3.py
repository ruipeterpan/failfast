# %%
import os
import sys
import time
import argparse
import logging
# import httpx
# import openai
from openai import OpenAI
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# Add root directory to path to import utils
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, root_dir)
from utils import populate_dataset, format_problem_and_options, get_first_user_msg

"""
VLLM_USE_V1=0 vllm serve Qwen/Qwen2.5-7B-Instruct --dtype auto -tp 2 --max_model_len 8192 --gpu-memory-utilization 0.8 --port 30000
"""

parser = argparse.ArgumentParser(description="Profiles the vLLM TPT of a model on all datasets.")
parser.add_argument("--port", type=int, default=30000, help="Port to use for the vLLM server")
parser.add_argument("--output_file", type=str, help="Name of the output file")
# parser.add_argument("--target_model_name", type=str, default="Qwen/Qwen2.5-32B-Instruct", 
parser.add_argument("--target_model_name", type=str, default="Qwen/Qwen2.5-14B-Instruct", 
                    help="Name of the base model to use")
parser.add_argument("--num_questions", type=int, default=30,
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

port = 30000
client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://localhost:{port}/v1",
)
models = client.models.list()
model = models.data[0].id

try:
    local_model_path = snapshot_download(
        repo_id=args.target_model_name, 
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
except Exception as e:
    logging.warning(f"Could not load tokenizer locally for {args.target_model_name}. Prompt will be sent raw. Error: {e}")
    exit(1)

# # GEN_ENDPOINT = f"http://localhost:{args.port}/inference/v1/generate"  # V0
# GEN_ENDPOINT = f"http://localhost:{args.port}/v1/completions"  # V1
# transport = httpx.HTTPTransport()
# headers = {"Authorization": f"Bearer dummy"}
# client = httpx.Client(
#     transport=transport,
#     base_url=GEN_ENDPOINT,
#     timeout=600,
#     headers=headers,
# )

tpts = {}
for dataset_name in ["math", "aime",  "gpqa", "mmlu", "gsm8k", "humaneval"]:
    args.dataset_name = dataset_name
    dataset = populate_dataset(args)

    total_time = 0
    total_output_tokens = 0

    for problem_id in range(args.num_questions):
        raw_data = format_problem_and_options(args, problem_id)
        
        messages = [
            {"role": "user", "content": get_first_user_msg(args, raw_data)},
        ]
        extra_body = {"add_generation_prompt": True}
        
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=args.max_new_tokens,
            extra_body=extra_body,
        )
        num_output_tokens = response.usage.completion_tokens
        elapsed = time.perf_counter() - start_time  # seconds
        
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
