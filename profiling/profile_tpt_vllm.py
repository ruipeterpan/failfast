# %%
import os
import time
import openai
from openai import OpenAI
from datasets import load_dataset, load_from_disk

port = 30000
client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://localhost:{port}/v1",
)
models = client.models.list()
model = models.data[0].id

system_prompt = """
Solve the following math problem efficiently and clearly. Please reason step by step, 
separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
Problem: {problem}
"""

dataset_name = "math"

if dataset_name == "aime":
    dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
elif dataset_name == "math":
    dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
elif dataset_name == "gpqa":
    if os.getenv("HF_HUB_OFFLINE", "0") == "1":
        dataset = load_from_disk("/scratch/gpfs/rp2773/hf_cache/datasets/gpqa")
    else:    
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
else:
    raise NotImplementedError
    
    




# %%
total_time = 0
total_output_tokens = 0

for problem_id in range(1):
    if dataset_name == "aime":
        problem = dataset["problem"][problem_id]
        options = None
    elif dataset_name == "math":
        problem = dataset["problem"][problem_id]
        options = None
    elif dataset_name == "gpqa":
        problem = dataset["Question"][problem_id]
        options = {
            "A": dataset["Correct Answer"][problem_id],
            "B": dataset["Incorrect Answer 1"][problem_id],
            "C": dataset["Incorrect Answer 2"][problem_id],
            "D": dataset["Incorrect Answer 3"][problem_id],
        }
    
    messages = [
        {"role": "user", "content": system_prompt.format(problem=problem)},
    ]
    extra_body = {"add_generation_prompt": True}
    
    start_time = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=512,
        extra_body=extra_body,
    )
    num_output_tokens = response.usage.completion_tokens
    elapsed = time.perf_counter() - start_time  # seconds
    
    total_time += elapsed
    total_output_tokens += num_output_tokens

avg_tpt = (total_time * 1000) / total_output_tokens
print(f"Average TPT: {avg_tpt:.2f} ms/token")

# %%
