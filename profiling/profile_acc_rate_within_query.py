# %%
import os
import sys
import time
import torch
import openai
import pickle
import pprint
import argparse
import transformers
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
# transformers.logging.set_verbosity_info()
transformers.logging.set_verbosity_error()

sys.path.insert(1, os.path.dirname(os.getcwd()))
from plotting import (
    visualize_boolean_series,
)

class Colors:
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def is_interactive():
    """Return True if running in an interactive shell (Jupyter or terminal)."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":  # Jupyter/IPython
            return True
    except Exception:
        pass
    return sys.stdout.isatty()

system_prompt = """
Solve the following math problem efficiently and clearly. Please reason step by step, 
separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
Problem: {problem}
"""


# %%
draft_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
target_model_name = "Qwen/Qwen2.5-32B-Instruct"
# target_model_name = "Qwen/Qwen2.5-7B-Instruct"
dllm_name = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"

draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_name,
    torch_dtype="auto",
    device_map="auto"
)
draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
target_model = AutoModelForCausalLM.from_pretrained(
    target_model_name,
    torch_dtype="auto",
    device_map="auto"
)
target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
# NOTE(ruipan): maybe they should use the same tokenizer?
dllm = AutoModelForCausalLM.from_pretrained(
    dllm_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
dllm_tokenizer = AutoTokenizer.from_pretrained(dllm_name, trust_remote_code=True)


# %%
def get_dataset(dataset_name):
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
    return dataset

def get_first_user_msg(problem, options=None):
    if options is None:
        system_prompt = """
        Solve the following math problem efficiently and clearly. Please reason step by step, 
        separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
        Problem: {problem}
        """
        return system_prompt.format(problem=problem)
    else:
        system_prompt = """
        What is the correct answer to the following problem? Please reason step by step. 
        Separate logical reasoning steps with two newline characters (\n\n).
        Put the final answer **strictly** in the format \\boxed{{X}}, where X is a single letter (A, B, C, or D).

        **Example output:** \\boxed{{A}}

        Problem: {problem}.
        Choices: 
        (A) {ans_a}
        (B) {ans_b}
        (C) {ans_c}
        (D) {ans_d}
        """
        return system_prompt.format(
            problem=problem,
            ans_a=options["A"],
            ans_b=options["B"],
            ans_c=options["C"],
            ans_d=options["D"],
        )

def format_problem_and_options(args, problem_id):
    if args.dataset_name == "aime":
        problem = dataset["problem"][problem_id]
        options = None
    elif args.dataset_name == "math":
        problem = dataset["problem"][problem_id]
        options = None
    elif args.dataset_name == "gpqa":
        problem = dataset["Question"][problem_id]
        options = {
            "A": dataset["Correct Answer"][problem_id],
            "B": dataset["Incorrect Answer 1"][problem_id],
            "C": dataset["Incorrect Answer 2"][problem_id],
            "D": dataset["Incorrect Answer 3"][problem_id],
        }
    return problem, options


def get_target_token_ids(model, tokenizer, messages):
    """Get the target series of token IDs for the given messages.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    num_input_tokens = model_inputs.input_ids.shape[1]
    print(f"num_input_tokens {num_input_tokens}, first eight tokens: {model_inputs.input_ids[0, :8].tolist()}")
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,  # was 512 in vanilla sd experiments
        # use greedy decoding, not sampling
        do_sample=False,
        # temperature=1.0,
        # top_p=1.0,
        # top_k=0.0,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    return generated_ids[0].tolist(), model_inputs


def get_next_n_tokens(model, orig_model_inputs, token_ids_so_far, n):
    """Get the next n tokens from the model given the token IDs so far.
    """
    new_tokens = torch.tensor(token_ids_so_far, device=orig_model_inputs['input_ids'].device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)  # attention mask = 1 for new tokens

    # Append along the sequence dimension (dim=1)
    new_model_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], new_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], new_mask], dim=1)
    }

    generated_ids = model.generate(
        **new_model_inputs,
        max_new_tokens=n,
        # use greedy decoding, not sampling
        do_sample=False,
        # temperature=1.0,
        # top_p=1.0,
        # top_k=0.0,
    )
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    
    return generated_ids.tolist()



def get_next_n_tokens_dllm(dllm, orig_model_inputs, token_ids_so_far, n, output_seqlen, small_block_size=8, threshold=0.9):
    """Get the next n tokens from the model given the token IDs so far.
    """
    num_tokens_in_prompt = orig_model_inputs.input_ids.shape[1]
    new_tokens = torch.tensor(token_ids_so_far, device=orig_model_inputs['input_ids'].device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)  # attention mask = 1 for new tokens

    # Append along the sequence dimension (dim=1)
    new_model_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], new_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], new_mask], dim=1)
    }

    generated_ids = dllm.generate(
        # **new_model_inputs,
        new_model_inputs["input_ids"],
        max_new_tokens=output_seqlen,  # NOTE(ruipan): setting this to 8 will not lead to new tokens hmm
        small_block_size=small_block_size,
        threshold=threshold,
        # use greedy decoding, not sampling
        do_sample=False,
        # temperature=1.0,
        # top_p=1.0,
        # top_k=0.0,
    )
    
    full_output_seqlen = generated_ids.shape[1]
    assert full_output_seqlen > num_tokens_in_prompt + len(token_ids_so_far), f"full_output_seqlen {full_output_seqlen}, num_tokens_in_prompt {num_tokens_in_prompt}, len(token_ids_so_far) {len(token_ids_so_far)}"
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    generated_ids = generated_ids.tolist()[:n]  # only take the next n tokens
    
    return generated_ids




# %%
parser = argparse.ArgumentParser(description="Profiles the acceptance rate of speculative decoding within a single query.")
parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa"], default="gpqa",
                    help="Dataset")
parser.add_argument("--output_dir", type=str, default="/data2/ruipan/diffspec", 
                    help="Where result pickle files (and output figures) will be written to")
parser.add_argument("--num_questions", type=int, default=1,
                    help="Number of questions to run profiling on")
args, _ = parser.parse_known_args()
args.output_dir_figures = os.path.join(args.output_dir, "figures", "acc_rate_within_query", args.dataset_name)
args.output_dir_pickles = os.path.join(args.output_dir, "pickles", "playground", "detailed_info", args.dataset_name)
for d in [args.output_dir_figures, args.output_dir_pickles]:
    os.makedirs(d, exist_ok=True)
dataset = get_dataset(args.dataset_name)


# %%
for problem_id in tqdm(range(args.num_questions), desc="Problems", position=0):
    problem, options = format_problem_and_options(args, problem_id)
    messages = [
        {"role": "user", "content": get_first_user_msg(problem, options)},
    ]
    
    target_ids, orig_model_inputs = get_target_token_ids(target_model, target_tokenizer, messages)
    num_target_tokens = len(target_ids)
    print(f"Target model generated {num_target_tokens} tokens")

    n = 5  # number of speculative tokens proposed each time
    accepted_tokens = 0
    rejected_tokens = 0
    current_token_ids = []  # prefix tokens generated so far
    acceptance_decisions = []
    pickled_data = {
        "orig_model_inputs": orig_model_inputs,
        "target_ids": target_ids,
        "problem": problem,
        "options": options,
        "num_target_tokens": num_target_tokens,
        "status_per_round": [],
    }

    if is_interactive():
        inner_bar = tqdm(total=num_target_tokens, miniters=25, desc=f"Verification (Problem {problem_id})",
                        position=1, leave=True, dynamic_ncols=False, file=sys.stdout)

    while len(current_token_ids) < len(target_ids):
        # Get next n speculative tokens from draft model
        # draft_proposal = get_next_n_tokens(draft_model, orig_model_inputs, current_token_ids, n=n)
        draft_proposal = get_next_n_tokens_dllm(dllm, orig_model_inputs, current_token_ids, n=n,
                                                output_seqlen=32)
        
        # The corresponding slice of ground-truth target tokens
        target_slice = target_ids[len(current_token_ids): len(current_token_ids) + n]

        info_this_round = {
            "current_token_ids": current_token_ids.copy(),
            "draft_proposal": draft_proposal,
            "target_slice": target_slice,
        }
        pickled_data["status_per_round"].append(info_this_round)

        # Compare draft proposal with target tokens one by one
        for draft_tok, target_tok in zip(draft_proposal, target_slice):
            if is_interactive():
                inner_bar.update(1)
            if draft_tok == target_tok:
                accepted_tokens += 1
                current_token_ids.append(draft_tok)
                acceptance_decisions.append(True)
            else:
                rejected_tokens += 1
                # replace with correct target token, sync with target model
                current_token_ids.append(target_tok)
                acceptance_decisions.append(False)
                # print(f"Rejection, current length: {len(current_token_ids)}, draft_tok {draft_tok}, target_tok {target_tok}")
                # print(f"draft token decoded: {draft_tokenizer.decode(draft_tok)}")
                # print(f"target token decoded: {target_tokenizer.decode(target_tok)}")
                break  # speculative generation diverged; go back to draft proposal step
            
                # FIXME(ruipan): math, question 1, len(target_ids) = 235, strange mismatch at len 148
            # print(f"Progress: len(current_token_ids) = {len(current_token_ids)}")

        # If we’ve already matched the full target sequence, stop
        if len(current_token_ids) >= len(target_ids):
            break

    if is_interactive():
        inner_bar.close()

    # Compute token acceptance rate
    acceptance_rate = accepted_tokens / (accepted_tokens + rejected_tokens)
    print(f"{Colors.YELLOW}Problem {problem_id} acceptance rate: {acceptance_rate:.3f}{Colors.RESET}")
    print(f"Accepted: {accepted_tokens}, Rejected: {rejected_tokens}, Total: {accepted_tokens + rejected_tokens}")

    # export
    visualize_boolean_series(acceptance_decisions, output_dir=args.output_dir_figures, problem_id=problem_id)
    
    with open(os.path.join(args.output_dir_pickles, f"{problem_id}.pickle"), "wb") as f:
        pickle.dump(pickled_data, f)
    with open(os.path.join(args.output_dir_pickles, f"{problem_id}.txt"), "w") as f:
        pprint.pprint(pickled_data, stream=f)

# %%
