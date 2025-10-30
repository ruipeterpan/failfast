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
from utils import (
    calculate_spec_decoding_speedup,
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


def get_target_token_ids(model, tokenizer, messages, max_new_tokens):
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
        max_new_tokens=max_new_tokens,  # was 512 in vanilla sd experiments
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



def get_next_n_tokens_dllm(dllm, orig_model_inputs, token_ids_so_far, n, output_seqlen, small_block_size, threshold, is_drafter):
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

    generated_ids, num_forward_passes, forward_pass_latencies = dllm.generate(
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
        is_drafter=is_drafter,
    )
    
    full_output_seqlen = generated_ids.shape[1]
    assert full_output_seqlen > num_tokens_in_prompt + len(token_ids_so_far), f"full_output_seqlen {full_output_seqlen}, num_tokens_in_prompt {num_tokens_in_prompt}, len(token_ids_so_far) {len(token_ids_so_far)}"
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    generated_ids = generated_ids.tolist()[:n]  # only take the next n tokens
    
    if any(x in generated_ids for x in [151665, 151645]):
        special_token = "MASK" if 151665 in generated_ids else "STOP"
        print(f"{Colors.RED}Generated ids contain {special_token} tokens! {generated_ids}{Colors.RESET}")
    
    return generated_ids, num_forward_passes, forward_pass_latencies




# %%
parser = argparse.ArgumentParser(description="Profiles the acceptance rate of speculative decoding within a single query.")
parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa"], default="math",
                    help="Dataset")
parser.add_argument("--output_dir", type=str, default="/data2/ruipan/diffspec", 
                    help="Where result pickle files (and output figures) will be written to")
parser.add_argument("--dllm_dir", type=str, default=None, 
                    help="Dir to the dLLM weights and (modified) modeling.py")
parser.add_argument("--num_questions", type=int, default=1,
                    help="Number of questions to run profiling on")
parser.add_argument("--max_new_tokens", type=int, default=512,
                    help="Max new tokens from the target model")
parser.add_argument("--veri_freq", type=int, default=5,
                    help="Frequency of verification steps (in number of tokens)")
parser.add_argument("--drafter_threshold", type=float, default=0.9,
                    help="Threshold for confidence-adaptive decoding of the dLLM drafter model")
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output pickles and figures')
args, _ = parser.parse_known_args()
args.output_dir_figures = os.path.join(args.output_dir, "figures", "acc_rate_within_query", args.dataset_name, str(args.drafter_threshold))
args.output_dir_pickles = os.path.join(args.output_dir, "pickles", "playground", "detailed_info", args.dataset_name, str(args.drafter_threshold))
for d in [args.output_dir_figures, args.output_dir_pickles]:
    os.makedirs(d, exist_ok=True)
dataset = get_dataset(args.dataset_name)

args.latency = {  # a6000, hf generate latencies
    "draft_fwd_pass": 28,  # ms; dLLM 1.5B drafter forward pass latency
    "target_tpt": 105,  # ms; Qwen2.5-32B, latency of short prefill pass (~=tpt)
}
# args.latency = {  # a6000, vllm latencies (assuming dllm latency is similar to 1.5b ar)
#     "draft_fwd_pass": 6.1,  # ms; dLLM 1.5B drafter forward pass latency
#     "target_tpt": 52.6,  # ms; Qwen2.5-32B, latency of short prefill pass (~=tpt)
# }

args.overwrite = False
args.drafter_threshold = 0.3
args.dllm_dir = "/data2/ruipan/Fast_dLLM_v2_1.5B"
args.max_new_tokens = 128

# %%
# draft_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
target_model_name = "Qwen/Qwen2.5-32B-Instruct"
# target_model_name = "Qwen/Qwen2.5-7B-Instruct"  # easier debugging
# target_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
dllm_name = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"

# draft_model = AutoModelForCausalLM.from_pretrained(
#     draft_model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
target_model = AutoModelForCausalLM.from_pretrained(
    target_model_name,
    torch_dtype="auto",
    device_map="auto"
)
target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
# NOTE(ruipan): maybe they should use the same tokenizer?
dllm = AutoModelForCausalLM.from_pretrained(
    args.dllm_dir if args.dllm_dir is not None else dllm_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
dllm_tokenizer = AutoTokenizer.from_pretrained(dllm_name, trust_remote_code=True)


# %%
for problem_id in tqdm(range(args.num_questions), desc="Problems", position=0):
    problem, options = format_problem_and_options(args, problem_id)
    messages = [
        {"role": "user", "content": get_first_user_msg(problem, options)},
    ]
    
    target_ids, orig_model_inputs = get_target_token_ids(target_model, target_tokenizer, messages, args.max_new_tokens)
    num_target_tokens = len(target_ids)
    print(f"Target model generated {num_target_tokens} tokens")

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

    num_speculation_rounds = 0
    total_num_forward_passes = 0
    while len(current_token_ids) < len(target_ids):
        print(f"Speculation round {num_speculation_rounds}")
        num_speculation_rounds += 1
        # Get next n speculative tokens from draft model
        # draft_proposal = get_next_n_tokens(draft_model, orig_model_inputs, current_token_ids, n=n)
        draft_proposal, num_forward_passes, forward_pass_latencies = get_next_n_tokens_dllm(dllm, orig_model_inputs, current_token_ids, 
                                                n=args.veri_freq,  # number of speculative tokens proposed each time
                                                output_seqlen=32,
                                                small_block_size=8,
                                                threshold=args.drafter_threshold,
                                                is_drafter=True,)
        total_num_forward_passes += num_forward_passes
        # print(f"forward_pass_latencies {forward_pass_latencies}")  # similar to TPT of 1.5B AR model
        
        # The corresponding slice of ground-truth target tokens
        target_slice = target_ids[len(current_token_ids): len(current_token_ids) + args.veri_freq]

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
                break  # speculative generation diverged; go back to draft proposal step
            
        # if all draft tokens are accepted, add one more token for free (from the verification prefill)
        if draft_proposal == target_slice:
            free_token_index = len(current_token_ids) + args.veri_freq
            print(f"{Colors.GREEN}All {len(draft_proposal)} speculative tokens accepted this round! free_token_index {free_token_index}{Colors.RESET}")
            if free_token_index >= len(target_ids):
                continue  # no more free tokens to add
            current_token_ids.append(target_ids[free_token_index])
            accepted_tokens += 1  # XXX(ruipan): is this correct? how is acceptance rate defined?
            acceptance_decisions.append(True)

        # If we’ve already matched the full target sequence, stop
        if len(current_token_ids) >= len(target_ids):
            break

    if is_interactive():
        inner_bar.close()

    # Compute token acceptance rate
    acceptance_rate = accepted_tokens / (accepted_tokens + rejected_tokens)
    print(f"{Colors.MAGENTA}drafter_threshold: {args.drafter_threshold}{Colors.RESET}")
    print(f"{Colors.MAGENTA}Problem {problem_id} acceptance rate: {acceptance_rate * 100:.1f}% ({accepted_tokens}/{num_target_tokens}){Colors.RESET}")
    print(f"{Colors.MAGENTA}Problem {problem_id} avg fwd passes/round: {total_num_forward_passes / num_speculation_rounds:.2f} ({total_num_forward_passes}/{num_speculation_rounds}){Colors.RESET}")
    if accepted_tokens + rejected_tokens != num_target_tokens:
        print(f"{Colors.RED}Warning: accepted + rejected != num_target_tokens!{Colors.RESET}")
    
    # compute e2e latency speedup
    latency_draft = total_num_forward_passes * args.latency["draft_fwd_pass"]  # ms
    latency_target = num_speculation_rounds * args.latency["target_tpt"]
    total_tpt = latency_draft + latency_target
    avg_tpt = total_tpt / num_target_tokens
    speedup = args.latency["target_tpt"] / avg_tpt
    theoretical_speedup = calculate_spec_decoding_speedup(
        alpha=0.9,  # offline-profiled acceptance rate of AR 1.5B drafter
        gamma=args.veri_freq,
        c=args.latency["draft_fwd_pass"] / args.latency["target_tpt"],
    )
    print(f"{Colors.MAGENTA}Avg TPT of SD: {avg_tpt:.2f}ms (Speedup: {speedup:.2f}x; Drafter latency ratio {latency_draft / total_tpt * 100:.1f}%){Colors.RESET}")
    print(f"{Colors.MAGENTA}Theoretical speedup of vanilla SD: {theoretical_speedup:.2f}x. Win: {speedup / theoretical_speedup:.3f}x.{Colors.RESET}")

    # export
    if args.overwrite:
        visualize_boolean_series(acceptance_decisions, output_dir=args.output_dir_figures, problem_id=problem_id)
    else:
        visualize_boolean_series(acceptance_decisions, output_dir=None, problem_id=None)
    
    pickled_data["num_speculation_rounds"] = num_speculation_rounds
    pickled_data["total_num_forward_passes"] = total_num_forward_passes
    pickled_data["accepted_tokens"] = accepted_tokens
    pickled_data["rejected_tokens"] = rejected_tokens
    
    if args.overwrite or (not os.path.exists(os.path.join(args.output_dir_pickles, f"{problem_id}.pickle"))):
        with open(os.path.join(args.output_dir_pickles, f"{problem_id}.pickle"), "wb") as f:
            pickle.dump(pickled_data, f)
        with open(os.path.join(args.output_dir_pickles, f"{problem_id}.txt"), "w") as f:
            pprint.pprint(pickled_data, stream=f)

# %%
