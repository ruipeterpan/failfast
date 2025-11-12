# %%
import torch
import logging

class Colors:
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    STRIKETHROUGH = '\033[9m' # The code for a line across text
    RESET = '\033[0m'

def print_sd_trajectory(pickled_data, tokenizer):
    logging.info(f"{Colors.BOLD}--- Input ---{Colors.RESET}")
    input_text = tokenizer.decode(pickled_data["orig_model_inputs"]["input_ids"][0], skip_special_tokens=False)
    num_input_tokens = len(pickled_data["orig_model_inputs"]["input_ids"][0])
    logging.info(input_text)
    # logging.info(f"{Colors.BOLD}--- Output ---{Colors.RESET}")
    # output_text = tokenizer.decode(pickled_data["stats_per_round"][-1]["current_token_ids"], skip_special_tokens=False)
    # logging.info(output_text)
    logging.info(f"{Colors.BOLD}--- Trajectory ---{Colors.RESET}")
    stats_per_round = pickled_data["stats_per_round"]
    output_str = ""
    for round_id in range(len(stats_per_round)):
        draft_proposal = stats_per_round[round_id]["draft_proposal"]
        target_tokens = stats_per_round[round_id]["target_tokens"]
        accepted_len = stats_per_round[round_id]["accepted_len"]
        proposal_len = len(draft_proposal)
        str_this_round = ""
        
        draft_accepted = draft_proposal[:accepted_len]
        draft_rejected = draft_proposal[accepted_len:]
        
        str_this_round += f"{tokenizer.decode(draft_accepted, skip_special_tokens=False)}"
        str_this_round += f"{Colors.RED}{Colors.STRIKETHROUGH}{tokenizer.decode(draft_rejected, skip_special_tokens=False)}{Colors.RESET}"
                
        if accepted_len < proposal_len:
            target_token = target_tokens[accepted_len]
            str_this_round += f"{Colors.GREEN}{tokenizer.decode([target_token], skip_special_tokens=False)}{Colors.RESET}"
        elif accepted_len == proposal_len:
            if round_id + 1 < len(stats_per_round):  # get the bonus token
                next_round_current_tokens = stats_per_round[round_id + 1]["current_token_ids"]
                target_token = next_round_current_tokens[-1]
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
