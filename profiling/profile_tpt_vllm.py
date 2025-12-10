# %%
import os
import sys
import time
import openai
import argparse
import logging
from openai import OpenAI

sys.path.insert(1, os.path.dirname(os.getcwd()) + "/..")
from utils import populate_dataset, format_problem_and_options, get_first_user_msg

"""
VLLM_USE_V1=0 vllm serve Qwen/Qwen2.5-7B-Instruct --dtype auto -tp 2 --max_model_len 8192 --gpu-memory-utilization 0.8 --port 30000
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

client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://localhost:{args.port}/v1",
)

for dataset_name in ["aime", "math", "gpqa", "mmlu", "humaneval"]:
    args.dataset_name = dataset_name
    dataset = populate_dataset(args)

    # %%
    total_time = 0
    total_output_tokens = 0

    # for problem_id in range(1):
    for problem_id in range(30):
        raw_data = format_problem_and_options(args, problem_id)
        
        messages = [
            {"role": "user", "content": get_first_user_msg(args, raw_data)},
        ]
        extra_body = {"add_generation_prompt": True}
        
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=args.target_model_name,
            messages=messages,
            max_tokens=args.max_new_tokens,
            extra_body=extra_body,
        )
        num_output_tokens = response.usage.completion_tokens
        elapsed = time.perf_counter() - start_time  # seconds
        
        total_time += elapsed
        total_output_tokens += num_output_tokens

    avg_tpt = (total_time * 1000) / total_output_tokens
    print(f"Dataset {dataset_name}: Average TPT: {avg_tpt:.2f} ms/token")

# %%
