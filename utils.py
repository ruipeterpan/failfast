# %%
import os
import sys
import torch
import logging
from datasets import load_dataset, load_from_disk
from transformers.cache_utils import DynamicCache

system_prompt = """
Solve the following math problem efficiently and clearly. Please reason step by step, 
separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
Problem: {problem}
"""

class Colors:
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    STRIKETHROUGH = '\033[9m' # The code for a line across text
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

def format_drafter_name(args, drafter_config):
    draft_type, drafter_threshold, freq_scheme, lowconf_threshold, \
        max_spec_len, incr_len = drafter_config
    if draft_type == "ar":
        if freq_scheme == "sf":
            return f"ar_None_sf_{args.spec_len}"
        elif freq_scheme == "df":
            return f"ar_None_df_{lowconf_threshold}_{max_spec_len}_{incr_len}"
        else:
            raise ValueError(f"Unknown freq_scheme for AR drafter: {freq_scheme}")
    else:  # dllm
        if freq_scheme == "sf":  # Fast-dLLM, static frequency
            return f"dllm_{drafter_threshold}_sf_{args.spec_len}"
        elif freq_scheme == "df":  # FailFast
            if lowconf_threshold is None:
                return f"dllm_{drafter_threshold}_df"  # obsolete
            else:
                return f"dllm_{drafter_threshold}_df_{lowconf_threshold}_{max_spec_len}_{incr_len}"
        else:
            raise ValueError(f"Unknown freq_scheme for dLLM drafter: {freq_scheme}")



def get_rejected_overlap_info(last_round_rejected, curr_round_proposal):
    """
    Finds the longest suffix of curr_round_proposal that exists in last_round_rejected.
    
    Args:
        last_round_rejected (list): The list containing rejected tokens.
        curr_round_proposal (list): The list containing the current proposal tokens.

    Returns:
        tuple: (length_to_end, start_index_rejected, start_index_proposal)
        
        - length_to_end: Length from the match start in rejected to the end of the rejected list.
        - start_index_rejected: Index in last_round_rejected where the match begins.
        - start_index_proposal: Index in curr_round_proposal where the matching suffix begins.
        
        Returns (0, -1, -1) if no match is found.

    Example:
        >>> last_rejected = [
        ...     1077, 594, 1477, 400, 69, 4080, 16, 15087, 1447, 41306, 
        ...     4080, 16, 8, 284, 1124, 37018, 90, 18, 4080, 16, 7287, 
        ...     17, 15170, 12, 16, 12, 17, 92, 284, 1124, 37018, 19999, 
        ...     18, 12, 17, 15170, 12, 18, 92, 284, 1124, 37018, 19999, 
        ...     20, 15170, 12, 18, 92, 284, 1124, 37018, 90
        ... ]
        >>> curr_proposal = [11, 1077, 594, 1477, 400, 69, 4080, 16, 15087, 1447]
        >>> get_rejected_overlap_info(last_rejected, curr_proposal)
        (53, 0, 1)

        # Explanation:
        # The longest matching suffix is [1077, 594, ..., 1447].
        # It starts at index 1 in curr_proposal.
        # It is found at index 0 in last_rejected.
        # Length from index 0 to the end of last_rejected (len 53) is 53.
    """
    len_proposal = len(curr_round_proposal)
    len_rejected = len(last_round_rejected)
    
    # Iterate through curr_round_proposal to create suffixes.
    # i represents the start index in the proposal list.
    for i in range(len_proposal):
        # Create the suffix x
        suffix = curr_round_proposal[i:]
        len_suffix = len(suffix)
        
        # Slide through last_round_rejected to find this suffix
        for j in range(len_rejected - len_suffix + 1):
            
            # Check if the slice matches the suffix
            if last_round_rejected[j : j + len_suffix] == suffix:
                
                # Found the suffix.
                # j is the index in rejected.
                # i is the index in proposal.
                length_to_end = len_rejected - j
                return length_to_end, j, i
                
    # Return defaults if no suffix matches
    return 0, -1, -1


def get_proposal_str(args, spec_len, accepted_len, draft_proposal, final_token):
    proposed_tokens_str = ""
    for i in range(accepted_len):
        proposed_tokens_str += args.target_tokenizer.decode([draft_proposal[i]])
    proposed_tokens_str += f"{Colors.RED}{Colors.STRIKETHROUGH}"
    for i in range(spec_len - accepted_len):
        if i + accepted_len >= len(draft_proposal):
            break
        proposed_tokens_str += args.target_tokenizer.decode([draft_proposal[i + accepted_len]])
    proposed_tokens_str += f"{Colors.RESET}"
    proposed_tokens_str += f"{Colors.GREEN}{args.target_tokenizer.decode([final_token])}{Colors.RESET}"
    return proposed_tokens_str

def get_output_dir(args, problem_id, drafter_config):
    output_dir_pickles, output_dir_figures = [os.path.join(
        args.output_dir, 
        x,
        args.target_model_name_clean,
        args.dataset_name, 
        str(problem_id),
        format_drafter_name(args, drafter_config),
    ) for x in ["pickles", "figures"]]
    for d in [output_dir_pickles, output_dir_figures]:
        os.makedirs(d, exist_ok=True)
    return output_dir_pickles, output_dir_figures


def populate_dataset(args):
    if args.dataset_name == "aime":
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
    elif args.dataset_name == "math":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    elif args.dataset_name == "gpqa":
        # if os.getenv("HF_HUB_OFFLINE", "0") == "1":
        #     dataset = load_from_disk("/scratch/gpfs/RAVIAN/rp2773/hf_cache/datasets/gpqa")
        # else:    
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    elif args.dataset_name == "mmlu":
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")["validation"]
    elif args.dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")["test"]
    elif args.dataset_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval")["test"]
    else:
        raise NotImplementedError
    args.dataset = dataset

def format_problem_and_options(args, problem_id):
    """
    Returns a dictionary with the raw problem fields data.
    """
    if args.dataset_name in ["aime", "math"]:
        return {"problem": args.dataset["problem"][problem_id]}
    elif args.dataset_name == "gpqa":
        problem = args.dataset["Question"][problem_id]
        options = {
            "A": args.dataset["Correct Answer"][problem_id],
            "B": args.dataset["Incorrect Answer 1"][problem_id],
            "C": args.dataset["Incorrect Answer 2"][problem_id],
            "D": args.dataset["Incorrect Answer 3"][problem_id],
        }
        return {"problem": problem, "options": options}
    elif args.dataset_name == "mmlu":
        data = args.dataset[problem_id]
        return {
            "problem": data["question"], 
            "options": data["options"],
            "category": data["category"],
        }
    elif args.dataset_name == "gsm8k":
        return {"problem": args.dataset["question"][problem_id]}
    elif args.dataset_name == "humaneval":
        data = args.dataset[problem_id]
        return {"problem": data["prompt"]}
    else:
        raise NotImplementedError

def get_first_user_msg(args, raw_data):
    if args.dataset_name in ["aime", "math"]:
        system_prompt = """
        Solve the following math problem efficiently and clearly. Please reason step by step, 
        separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
        Problem: {problem}
        """
        return system_prompt.format(problem=raw_data["problem"])
    elif args.dataset_name == "gpqa":
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
            problem=raw_data["problem"],
            ans_a=raw_data["options"]["A"],
            ans_b=raw_data["options"]["B"],
            ans_c=raw_data["options"]["C"],
            ans_d=raw_data["options"]["D"],
        )
    elif args.dataset_name == "mmlu":
        system_prompt = """
        The following is multiple choice question (with answers) about {category}.
        Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.
        
        Question: {problem}
        Choices: {options}
        """
        return system_prompt.format(
            category=raw_data["category"],
            problem=raw_data["problem"],
            options=raw_data["options"],
        )
    elif args.dataset_name == "gsm8k":
        system_prompt = """
        Think step by step and then please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:
        ```
        {problem}
        ```
        """
        return system_prompt.format(problem=raw_data["problem"])
    elif args.dataset_name == "humaneval":
        system_prompt = """
        Think step by step and then please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:
        ```
        {problem}
        ```
        """
        return system_prompt.format(problem=raw_data["problem"])
    else:
        raise NotImplementedError

def merge_dynamic_caches(prev_cache, new_cache):
    merged = DynamicCache()

    num_layers = len(prev_cache.key_cache)
    # print(f"num_layers {num_layers}")
    for layer in range(num_layers):
        k1 = prev_cache.key_cache[layer]     # [b, h, t1, d]
        v1 = prev_cache.value_cache[layer]
        k2 = new_cache.key_cache[layer]      # [b, h, t2, d]
        v2 = new_cache.value_cache[layer]
        # print(f"k1.shape {str(k1.shape)}, k2.shape {str(k2.shape)}")
        merged_k = torch.cat([k1, k2], dim=2)
        merged_v = torch.cat([v1, v2], dim=2)

        merged.key_cache.append(merged_k)
        merged.value_cache.append(merged_v)

    return merged

def join_outputs(output, output_to_append):
    # 1. merge logits
    output.logits = torch.cat([
        output.logits, 
        output_to_append.logits]
    , dim=1)
    # 2. merge KVs
    output.past_key_values = merge_dynamic_caches(
        output.past_key_values,
        output_to_append.past_key_values,
    )
    return output

def get_output_tokens(stats_each_round):
    output_token_ids = []
    for round_id in range(len(stats_each_round)):
        accepted_len = stats_each_round[round_id]["accepted_len"]
        draft_proposal = stats_each_round[round_id]["~draft_proposal"]
        output_token_ids.extend(draft_proposal[:accepted_len])

        if stats_each_round[round_id]["bonus_token"] is not None:
            output_token_ids.append(stats_each_round[round_id]["bonus_token"])
        else:
            output_token_ids.append(stats_each_round[round_id]["final_token"])
    return output_token_ids

def print_sd_trajectory(pickled_data, tokenizer):
    logging.info(f"{Colors.BOLD}--- Input ---{Colors.RESET}")
    input_text = tokenizer.decode(pickled_data["orig_model_inputs"], skip_special_tokens=False)
    num_input_tokens = len(pickled_data["orig_model_inputs"])
    logging.info(input_text)
    logging.info(f"{Colors.BOLD}--- Output ---{Colors.RESET}")
    output_tokens = get_output_tokens(pickled_data["stats_each_round"])
    output_text = tokenizer.decode(output_tokens, skip_special_tokens=False)  # missing draft tokens in the last round
    logging.info(output_text)
    logging.info(f"{Colors.BOLD}--- Trajectory ---{Colors.RESET}")
    stats_each_round = pickled_data["stats_each_round"]
    output_str = ""
    for round_id in range(len(stats_each_round)):
        draft_proposal = stats_each_round[round_id]["~draft_proposal"]
        # target_tokens = stats_each_round[round_id]["target_tokens"]
        accepted_len = stats_each_round[round_id]["accepted_len"]
        proposal_len = len(draft_proposal)
        str_this_round = ""
        
        draft_accepted = draft_proposal[:accepted_len]
        draft_rejected = draft_proposal[accepted_len:]
        
        str_this_round += f"{tokenizer.decode(draft_accepted, skip_special_tokens=False)}"
        str_this_round += f"{Colors.RED}{Colors.STRIKETHROUGH}{tokenizer.decode(draft_rejected, skip_special_tokens=False)}{Colors.RESET}"
                
        if accepted_len < proposal_len:
            target_token = stats_each_round[round_id]["final_token"]
        elif accepted_len == proposal_len:  # get the bonus token
            target_token = stats_each_round[round_id]["bonus_token"]
        str_this_round += f"{Colors.GREEN}{tokenizer.decode([target_token], skip_special_tokens=False)}{Colors.RESET}"
        
        output_str += str_this_round
    logging.info(output_str)
        

def calculate_spec_decoding_speedup(alpha, gamma, c):
    """Calculate the speculative decoding speedup.

    Reference: Theorem 3.8 in https://arxiv.org/pdf/2211.17192
    
    Args:
        alpha (float): Avg per-token acceptance rate, between 0 and 1.
        gamma (int): The number of drafted tokens.
        c (float): The drafter-to-target per-token latency ratio.
    """
    numerator = 1 - alpha ** (gamma + 1)
    denominator = (1 - alpha) * (c * gamma + 1)
    speedup = numerator / denominator
    return speedup

def check_prefill_output_equivalence(output1, output2, round_idx):
    if not torch.equal(output1.logits, output2.logits):
        print(f"[Round {round_idx}] Logits are not equal!")
    output1_kvs = output1.past_key_values.to_legacy_cache()
    output2_kvs = output2.past_key_values.to_legacy_cache()
    
    for layer_idx, (layer_kvs1, layer_kvs2) in enumerate(zip(output1_kvs, output2_kvs)):
        for kv_idx, (kv1, kv2) in enumerate(zip(layer_kvs1, layer_kvs2)):
            if not torch.equal(kv1, kv2):
                print(f"[Round {round_idx}] Past key values are not equal at layer {layer_idx}, kv {kv_idx}!")

def check_prefill_output_list_equivalence(output1, output2):
    for round_idx in range(min(len(output1), len(output2))):
        o1 = output1[round_idx]
        o2 = output2[round_idx]
        check_prefill_output_equivalence(o1, o2, round_idx)
# %%
