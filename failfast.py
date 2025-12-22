# %%
import os
import sys
import copy
import time
import torch
import pickle
import pprint
import logging
import argparse
import transformers
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

# transformers.logging.set_verbosity_debug()
# transformers.logging.set_verbosity_info()
# transformers.logging.set_verbosity_warning()
transformers.logging.set_verbosity_error()

sys.path.insert(1, os.path.dirname(os.getcwd()))
from plotting import (
    visualize_acc_rate_over_time,
)
from utils import (
    Colors, is_interactive,
    populate_dataset, get_first_user_msg, 
    format_problem_and_options, format_drafter_name, get_proposal_str, get_output_tokens,
    get_output_dir,
    print_sd_trajectory,
)

# %%
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
    logging.debug(f"num_input_tokens {num_input_tokens}, first eight tokens: {model_inputs.input_ids[0, :8].tolist()}")
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,  # was 512 in vanilla sd experiments
        do_sample=False,  # use greedy decoding, not sampling; overrides all below sampling params
        # temperature=0.0, top_p=1.0, top_k=0.0,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    return generated_ids[0].tolist(), model_inputs


def get_next_n_tokens_ar(model, orig_model_inputs, token_ids_so_far, n):
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
        do_sample=False,  # use greedy decoding, not sampling; overrides all below sampling params
        # temperature=0.0, top_p=1.0, top_k=0.0,
    )
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    
    return generated_ids.tolist()



def get_next_n_tokens_dllm(dllm, args, orig_model_inputs, token_ids_so_far, spec_len, output_seqlen, small_block_size, threshold, is_drafter, prev_prefill_output=None):
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

    if args.disable_reusing_drafter_kvs:
        generated_ids, num_forward_passes, forward_pass_latencies = dllm.generate_draft_tokens(
            # **new_model_inputs,
            new_model_inputs["input_ids"],
            max_new_tokens=output_seqlen,  # NOTE(ruipan): setting this to 8 will not lead to new tokens hmm
            small_block_size=small_block_size,
            threshold=threshold,
            # use greedy decoding, not sampling
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            # use_block_cache=True,  # NOTE(ruipan): doesn't seem to make a difference in latency...
            is_drafter=is_drafter,
            spec_len=spec_len,
            return_prefill_kvs=False,
            args=args,
        )
    else:
        generated_ids, prefill_output, num_forward_passes, forward_pass_latencies = dllm.generate_draft_tokens(
            # **new_model_inputs,
            new_model_inputs["input_ids"],
            max_new_tokens=output_seqlen,  # NOTE(ruipan): setting this to 8 will not lead to new tokens hmm
            small_block_size=small_block_size,
            threshold=threshold,
            # use greedy decoding, not sampling
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            # use_block_cache=True,  # NOTE(ruipan): doesn't seem to make a difference in latency...
            is_drafter=is_drafter,
            spec_len=spec_len,
            return_prefill_kvs=True,
            prev_prefill_output=prev_prefill_output,
            args=args,
        )
    
    full_output_seqlen = generated_ids.shape[1]
    assert full_output_seqlen > num_tokens_in_prompt + len(token_ids_so_far), f"full_output_seqlen {full_output_seqlen}, num_tokens_in_prompt {num_tokens_in_prompt}, len(token_ids_so_far) {len(token_ids_so_far)}"
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    generated_ids = generated_ids.tolist()[:spec_len]  # only take the next n tokens
    
    if any(x in generated_ids for x in [151665, 151645]):
        special_token = "MASK" if 151665 in generated_ids else "STOP"
        logging.info(f"{Colors.RED}Generated ids contain {special_token} tokens! {generated_ids}{Colors.RESET}")
    
    if not args.disable_reusing_drafter_kvs:
        return generated_ids, prefill_output, num_forward_passes, forward_pass_latencies
    return generated_ids, num_forward_passes, forward_pass_latencies


def get_next_tokens_dllm(dllm, args, orig_model_inputs, token_ids_so_far, spec_len, output_seqlen, small_block_size, threshold, is_drafter, prev_prefill_output=None,
                        lowconf_threshold=None,
                        max_spec_len=None,
                        incr_len=None,
                        last_round_rejected=None,
    ):
    """Get the next few tokens from the model given the token IDs so far.
    Difference is that the dLLM drafter itself determines how many tokens to output based on model internal signals.
    """
    num_tokens_in_prompt = orig_model_inputs.input_ids.shape[1]
    new_tokens = torch.tensor(token_ids_so_far, device=orig_model_inputs['input_ids'].device, dtype=torch.long).unsqueeze(0)
    new_mask = torch.ones_like(new_tokens, dtype=torch.long)  # attention mask = 1 for new tokens

    # Append along the sequence dimension (dim=1)
    new_model_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], new_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], new_mask], dim=1)
    }

    if args.disable_reusing_drafter_kvs:
        generated_ids, actual_spec_len, num_forward_passes, forward_pass_latencies = dllm.generate_draft_tokens_arbitrary_length(
            # **new_model_inputs,
            new_model_inputs["input_ids"],
            max_new_tokens=output_seqlen,
            small_block_size=small_block_size,
            threshold=threshold,
            # use greedy decoding, not sampling
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            # use_block_cache=True,  # NOTE(ruipan): doesn't seem to make a difference in latency...
            is_drafter=is_drafter,
            spec_len=spec_len,
            return_prefill_kvs=False,
            args=args,
            lowconf_threshold=lowconf_threshold,
            max_spec_len=max_spec_len,
            incr_len=incr_len,
            last_round_rejected=last_round_rejected,
        )
    else:
        generated_ids, actual_spec_len, prefill_output, num_forward_passes, forward_pass_latencies = dllm.generate_draft_tokens_arbitrary_length(
            # **new_model_inputs,
            new_model_inputs["input_ids"],
            max_new_tokens=output_seqlen,
            small_block_size=small_block_size,
            threshold=threshold,
            # use greedy decoding, not sampling
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            # use_block_cache=True,  # NOTE(ruipan): doesn't seem to make a difference in latency...
            is_drafter=is_drafter,
            spec_len=spec_len,
            return_prefill_kvs=True,
            prev_prefill_output=prev_prefill_output,
            args=args,
            lowconf_threshold=lowconf_threshold,
            max_spec_len=max_spec_len,
            incr_len=incr_len,
            last_round_rejected=last_round_rejected,
        )
    
    full_output_seqlen = generated_ids.shape[1]
    assert full_output_seqlen > num_tokens_in_prompt + len(token_ids_so_far), f"full_output_seqlen {full_output_seqlen}, num_tokens_in_prompt {num_tokens_in_prompt}, len(token_ids_so_far) {len(token_ids_so_far)}"
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    generated_ids = generated_ids.tolist()[:actual_spec_len]  # only take the next n tokens
    
    if any(x in generated_ids for x in [151665, 151645]):
        special_token = "MASK" if 151665 in generated_ids else "STOP"
        logging.info(f"{Colors.RED}Generated ids contain {special_token} tokens! {generated_ids}{Colors.RESET}")
    
    if not args.disable_reusing_drafter_kvs:
        return generated_ids, actual_spec_len, prefill_output, num_forward_passes, forward_pass_latencies
    return generated_ids, actual_spec_len, num_forward_passes, forward_pass_latencies


def construct_drafter_configs(args):
    drafter_configs = []
    if args.run_ar:
        drafter_configs.extend([("ar", None, "sf", None, None, None)] )
    if args.run_dllm_sf:
        drafter_configs.extend([("dllm", thr, "sf", None, None, None) for thr in args.drafter_thresholds])
    if not args.baseline_sweep:
        drafter_configs.extend([("dllm", thr, "df", lowconf_threshold, max_spec_len, incr_len) 
                                for thr in args.drafter_thresholds
                                for lowconf_threshold in args.sweep_lowconf_threshold
                                for max_spec_len in args.sweep_max_spec_len
                                for incr_len in args.sweep_incr_len
                                ])
    args.drafter_configs = drafter_configs


# %%
parser = argparse.ArgumentParser(description="Profiles the acceptance rate of speculative decoding within a single query.")
parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa", "mmlu", "gsm8k", "humaneval"], default="math",
                    help="Dataset")
parser.add_argument("--output_dir", type=str, default="/data2/ruipan/diffspec", 
                    help="Where result pickle files (and output figures) will be written to")
parser.add_argument("--target_model_name", type=str, default="Qwen/Qwen2.5-32B-Instruct", 
                    help="Name of the base model to use")
parser.add_argument("--dllm_dir", type=str, default="/data2/ruipan/Fast_dLLM_v2_1.5B", 
                    help="Dir to the dLLM weights and (modified) modeling.py")
parser.add_argument("--num_questions", type=int, default=1,
                    help="Number of questions to run profiling on")
parser.add_argument("--max_new_tokens", type=int, default=1024,
                    help="Max new tokens from the target model")
parser.add_argument("--block_size", type=int, default=32,
                    help="Block size in Fast-dLLM")
parser.add_argument("--spec_len", type=int, default=10,
                    help="Frequency of verification steps (in number of tokens)")
parser.add_argument("--drafter_thresholds", type=float, nargs="+",  # one or more float thresholds
                    default=[0.05],
                    help="Threshold for confidence-adaptive decoding of the dLLM drafter model (e.g., --drafter_thresholds 0.1 0.5 0.9)")
parser.add_argument("--log_level",
                    type=str,
                    default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Set the logging level")
parser.add_argument("--sweep_lowconf_threshold", type=float, nargs="+",
                    default=[0.4],
                    help="XXX")
parser.add_argument("--sweep_max_spec_len", type=int, nargs="+",
                    default=[60],
                    help="XXX")
parser.add_argument("--sweep_incr_len", type=int, nargs="+",
                    default=[10],
                    help="XXX")
parser.add_argument('--run_ar', action='store_true', help='Run the AR drafter to compare speedups')
parser.add_argument('--run_dllm_sf', action='store_true', help='Run the dLLM drafter with static frequency (in param sweep for baselines)')
parser.add_argument('--baseline_sweep', action='store_true', help='Running a baseline sweep, don\'t run dynamic frequency')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output pickles and figures')
parser.add_argument('--disable_reusing_drafter_kvs', action='store_true', help='Disables reusing drafter KV cache across verification rounds')
parser.add_argument('--read_pickle', action='store_true', help='Use acceptance decisions from a cached pickle file rather than rerunning')
args, _ = parser.parse_known_args()
args.target_model_name_clean = args.target_model_name.split("/", 1)[1]


######custom fields for easier debugging######
# args.log_level = "DEBUG"
# args.disable_reusing_drafter_kvs = True
# args.dataset_name = "gpqa"
# args.overwrite = True
# args.max_new_tokens = 1024
# args.run_ar = True
# args.baseline_sweep = True
# args.spec_len = 8
# args.run_dllm_sf = True
# args.read_pickle = True  # XXX: read trajectory from pickle as well in future debugging
# args.target_model_name = "Qwen/Qwen2.5-7B-Instruct"  # for easier debugging
# args.sweep_lowconf_threshold = [0.4]
# args.sweep_max_spec_len = [50]
# args.sweep_incr_len = [10]
######custom fields for easier debugging######


logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%m%d %H:%M:%S",
    # datefmt="%m%d",
)

construct_drafter_configs(args)  # populates args.drafter_configs
populate_dataset(args)  # populates args.dataset

args.latency = {  # all in ms
    # "HuggingFace_A6000": {  # a6000, hf generate latencies
    #     "draft_fwd_pass": 28,  # dLLM 1.5B drafter forward pass latency
    #     "target_tpt": {
    #         "Qwen2.5-32B-Instruct": 105,  # Qwen2.5-32B, latency of short prefill pass (~=tpt)
    #     },
    # },
    "vLLM_A6000": {  # eager mode on for all numbers
        "draft_fwd_pass": 6.1,
        "target_tpt": {
            "Qwen2.5-7B-Instruct": 13.5,
            "Qwen2.5-14B-Instruct": 24.7,
            "Qwen2.5-32B-Instruct": 52.6,
        },
    },
    "vLLM_H100": {
        "draft_fwd_pass": 2.9,  # eager mode: 9.25
        "target_tpt": {  # eager mode on :P
            "Qwen2.5-14B-Instruct": 14.3,  # w/o eager mode: 14.3. 8.45??
            "Qwen2.5-32B-Instruct": 18.6,
            "Qwen2.5-72B-Instruct": 32.2,
        },
    },
}


target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
args.target_tokenizer = target_tokenizer

# %%
if not args.read_pickle:
    logging.info(f"{Colors.BOLD}=== Loading target model: {args.target_model_name} ==={Colors.RESET}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    dllm_name = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
    dllm = AutoModelForCausalLM.from_pretrained(
        args.dllm_dir if args.dllm_dir is not None else dllm_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    # NOTE(ruipan): drafter and target should probably share the same tokenizer?
    # dllm_tokenizer = AutoTokenizer.from_pretrained(dllm_name, trust_remote_code=True)
    dllm_tokenizer = target_tokenizer
    if args.run_ar:
        draft_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        draft_tokenizer = target_tokenizer

# %%
for problem_id in tqdm(range(args.num_questions), desc="Problems", position=0):
# for problem_id in [21]:
    transformers.set_seed(42)  # reproducibility for each question-model-model config pair
    raw_data = format_problem_and_options(args, problem_id)
    messages = [
        {"role": "user", "content": get_first_user_msg(args, raw_data)},
    ]
    text = args.target_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    if not args.read_pickle:
        orig_model_inputs = target_tokenizer([text], return_tensors="pt").to(target_model.device)
        num_target_tokens = args.max_new_tokens  # drafters will generate this many tokens
    
    # if not args.read_pickle:
    #     target_ids, orig_model_inputs = get_target_token_ids(target_model, target_tokenizer, messages, max_new_tokens=args.max_new_tokens)
    #     num_target_tokens = len(target_ids)
    #     logging.info(f"[Problem {problem_id}] Target (vanilla) generation length: {num_target_tokens} tokens")
    
    ar_drafter_speedup = {k: None for k in args.latency.keys()}
    for drafter_config in args.drafter_configs:
        transformers.set_seed(42)  # reproducibility for each question-model-model config pair
        draft_type, drafter_threshold, freq_scheme, lowconf_threshold, max_spec_len, incr_len = drafter_config
        drafter_name = format_drafter_name(args, drafter_config)
        
        # set up output dirs and export
        output_dir_pickles, output_dir_figures = get_output_dir(args, str(problem_id), drafter_config)
        
        if args.read_pickle:
            if not os.path.exists(os.path.join(output_dir_pickles, f"{args.max_new_tokens}.pickle")):
                logging.warning(f"{Colors.RED}No cached pickle found for {drafter_name} (token budget {args.max_new_tokens})!{Colors.RESET}")
            
            with open(os.path.join(output_dir_pickles, f"{args.max_new_tokens}.pickle"), "rb") as f:
                pickled_data = pickle.load(f)
            
            accepted_tokens = pickled_data["accepted_tokens"]
            drafted_tokens = pickled_data["drafted_tokens"]
            rejected_tokens = pickled_data["rejected_tokens"]
            num_speculation_rounds = pickled_data["num_speculation_rounds"]
            total_num_forward_passes = pickled_data["total_num_forward_passes"]
            current_token_ids = get_output_tokens(pickled_data["stats_each_round"])
        else:  # run the actual spec decoding pipeline
            logging.info(f"{Colors.BOLD}=== [Problem {problem_id}] Running drafter: {drafter_name} ==={Colors.RESET}")
            accepted_tokens = 0
            rejected_tokens = 0
            num_speculation_rounds = 0
            total_num_forward_passes = 0
            current_token_ids = []  # prefix tokens generated so far
            prev_prefill_output = None  # drafter's prefill KVs
            pickled_data = {
                "orig_model_inputs": orig_model_inputs["input_ids"][0].tolist(),
                # "target_ids": target_ids,  # not necessarily equal to current_token_ids at the end due to numerical differences/different kernels during rounds of prefill
                "raw_data": raw_data,
                "num_target_tokens": num_target_tokens,
                "stats_each_round": [],
            }

            if is_interactive():
                inner_bar = tqdm(total=num_target_tokens, miniters=25, desc=f"Verification (Problem {problem_id})",
                                position=1, leave=True, dynamic_ncols=False, file=sys.stdout)

            while len(current_token_ids) < num_target_tokens:
                logging.debug(f"--- [{drafter_name}_{freq_scheme}] Speculation round {num_speculation_rounds} ---")
                
                # A. PROPOSE: Get next n speculative tokens from draft model based on current accepted prefix
                if draft_type == "ar":
                    draft_proposal = get_next_n_tokens_ar(draft_model, orig_model_inputs, current_token_ids, n=args.spec_len)
                    num_forward_passes = args.spec_len  # 1 fwd pass per token for AR drafter
                    spec_len = args.spec_len
                elif draft_type == "dllm":
                    if freq_scheme == "sf":  # static frequency
                        spec_len = args.spec_len
                        if args.disable_reusing_drafter_kvs:
                            draft_proposal, num_forward_passes, forward_pass_latencies = get_next_n_tokens_dllm(dllm, args, orig_model_inputs, current_token_ids, 
                                                                    spec_len=spec_len,  # number of speculative tokens proposed each time
                                                                    output_seqlen=3*args.block_size,  # 2 blocks of 32. Ensures spec_len tokens are generated in case they span over two blocks
                                                                    small_block_size=8,
                                                                    threshold=drafter_threshold,
                                                                    is_drafter=True,)
                        else:
                            draft_proposal, prefill_output, num_forward_passes, forward_pass_latencies = get_next_n_tokens_dllm(dllm, args, orig_model_inputs, current_token_ids, 
                                                                    spec_len=spec_len,  # number of speculative tokens proposed each time
                                                                    output_seqlen=3*args.block_size,  # 2 blocks of 32. Ensures spec_len tokens are generated in case they span over two blocks
                                                                    small_block_size=8,
                                                                    threshold=drafter_threshold,
                                                                    is_drafter=True,
                                                                    prev_prefill_output=prev_prefill_output)
                            prev_prefill_output = prefill_output
                    else:  # dynamic frequency: drafter determines how many tokens to propose
                        
                        ###start of logic of reusing rejected drafts from the last round###
                        last_round_proposal = pickled_data["stats_each_round"][-1]["~draft_proposal"] if num_speculation_rounds > 0 else []
                        last_round_accepted_len = pickled_data["stats_each_round"][-1]["accepted_len"] if num_speculation_rounds > 0 else 0
                        if last_round_accepted_len < len(last_round_proposal) - 1:  # there are salvagable rejected tokens in the last round
                            # last_round_rejected = last_round_proposal[last_round_accepted_len+1:] if num_speculation_rounds > 0 else []
                            last_round_rejected = None
                        else:
                            last_round_rejected = None
                        ###end of logic of reusing rejected drafts from the last round###
                        
                        draft_proposal, actual_spec_len, prefill_output, num_forward_passes, forward_pass_latencies = get_next_tokens_dllm(dllm, args, orig_model_inputs, current_token_ids, 
                                                                    spec_len=args.spec_len,  # number of speculative tokens proposed each time
                                                                    output_seqlen=3*args.block_size,  # 2 blocks of 32. Ensures spec_len tokens are generated in case they span over two blocks
                                                                    small_block_size=8,
                                                                    threshold=drafter_threshold,
                                                                    is_drafter=True,
                                                                    prev_prefill_output=prev_prefill_output,
                                                                    lowconf_threshold=lowconf_threshold,
                                                                    max_spec_len=max_spec_len,
                                                                    incr_len=incr_len,
                                                                    last_round_rejected=last_round_rejected,
                                                                    )
                        prev_prefill_output = prefill_output
                        spec_len = actual_spec_len  # update spec_len to the actual number of tokens proposed
                        
                total_num_forward_passes += num_forward_passes
                # print(f"forward_pass_latencies {forward_pass_latencies}")  # NOTE(ruipan): seems to be similar to TPT of 1.5B AR model
                
                if not draft_proposal: # Stop if the draft model has nothing to say
                    logging.info(f"{Colors.RED}[Round {num_speculation_rounds}] Warning: Draft model returned no tokens{Colors.RESET}")
                    break
                
                if len(draft_proposal) < spec_len:
                    logging.info(f"{Colors.RED}[Round {num_speculation_rounds}] Warning: Draft model returned fewer tokens ({len(draft_proposal)}) than spec_len ({spec_len})!{Colors.RESET}")
                    # break
                
                # B. Verify proposed tokens
                prefix_len = len(current_token_ids)
                combined_ids = current_token_ids + draft_proposal
                verify_input_tensor = torch.tensor([combined_ids], device=target_model.device)
                full_input_ids = torch.cat([orig_model_inputs['input_ids'], verify_input_tensor], dim=1)

                with torch.no_grad():
                    outputs = target_model(input_ids=full_input_ids)
                    
                    # The logits for the draft tokens start after the prompt and the accepted prefix.
                    # The logit at sequence position 't' is the prediction for the token at 't+1'.
                    # So we need the logits from the token *before* the first draft token up to the one *before* the last.
                    start_index = orig_model_inputs['input_ids'].shape[1] + prefix_len - 1
                    end_index = start_index + len(draft_proposal)
                    verify_logits = outputs.logits[0, start_index:end_index]
                    target_tokens = torch.argmax(verify_logits, dim=-1).tolist()
                
                # # TODO: add stopping condition if drafter proposes EOS token?
                # if draft_proposal and draft_proposal[-1] == target_tokenizer.eos_token_id:
                #     logging.info(f"{Colors.YELLOW}Drafter proposed EOS token. Stopping speculation.{Colors.RESET}")
                #     break
                
                # C. ACCEPT/REJECT
                accepted_len = 0
                bonus_token = None
                for i in range(len(draft_proposal)):
                    # The target's prediction for position `i` is the argmax of the logits at position `i`
                    target_pred = torch.argmax(verify_logits[i, :], dim=-1).item()
                    # print(f"  Verifying draft token at position {i}: {draft_proposal[i]}. Target would have picked {target_pred}.")

                    # if target_pred == target_tokenizer.eos_token_id:
                    #     logging.info(f"{Colors.GREEN}Round {num_speculation_rounds} proposed EOS token at position {i}, draft_proposal {draft_proposal}.{Colors.RESET}")

                    if draft_proposal[i] == target_pred:
                        accepted_len += 1
                    else:
                        # Mismatch found. The correct token is the target's prediction.
                        final_token = target_pred
                        break
                else:
                    # All draft tokens were accepted. Get a "bonus" token from the final logit.
                    final_token_logits = outputs.logits[0, -1, :]
                    final_token = torch.argmax(final_token_logits, dim=-1).item()
                    bonus_token = final_token
                    # print(f"  All draft tokens accepted! Bonus token is {final_token}.")
                
                proposal_str = get_proposal_str(args, spec_len, accepted_len, draft_proposal, final_token)
                logging.debug(f"--- Speculation round {num_speculation_rounds} proposed token IDs {draft_proposal}, bonus token {bonus_token} ---")
                logging.debug(f"--- Speculation round {num_speculation_rounds} proposed str: {proposal_str} ---")
                logging.debug(f"--- Speculation round {num_speculation_rounds} acceptance rate {accepted_len / spec_len:.2f} ({accepted_len}/{spec_len}) ---")
                logging.debug(f"--- Speculation round {num_speculation_rounds} num_forward_passes: {num_forward_passes} ---")
                
                # D. UPDATE
                tokens_to_append = draft_proposal[:accepted_len] + [final_token]
                current_token_ids.extend(tokens_to_append)
                
                accepted_tokens += accepted_len
                rejected_tokens += len(draft_proposal) - accepted_len
                
                info_this_round = {
                    "target_tokens": target_tokens,  # useful up until the first rejection/mismatch
                    "prefix_len": prefix_len,  # number of decoded tokens before this round
                    "spec_len": spec_len,  # number of speculated tokens
                    "~draft_proposal": draft_proposal,  # list of speculated token IDs
                    "accepted_len": accepted_len,  # number of accepted tokens
                    "acceptance_rate": accepted_len / spec_len,  # acceptance rate this round
                    "num_forward_passes": num_forward_passes,  # number of drafter fwd passes this round
                    "final_token": final_token,  # the first corrected token or bonus token
                    "bonus_token": bonus_token,
                }
                pickled_data["stats_each_round"].append(info_this_round)
                
                num_speculation_rounds += 1
                
                if is_interactive():
                    inner_bar.update(len(tokens_to_append))

                if target_tokenizer.eos_token_id in tokens_to_append:
                    break

            if is_interactive():
                inner_bar.close()

        # Compute token acceptance rate
        # drafted_tokens = num_speculation_rounds * args.spec_len
        drafted_tokens = sum([x["spec_len"] for x in pickled_data["stats_each_round"]])
        acceptance_rate = accepted_tokens / drafted_tokens
        avg_spec_len = sum([x["spec_len"] for x in pickled_data["stats_each_round"]]) / num_speculation_rounds
        avg_acc_len = sum([x["accepted_len"] for x in pickled_data["stats_each_round"]]) / num_speculation_rounds
        max_spec_len = max([x["spec_len"] for x in pickled_data["stats_each_round"]])
        max_acc_len = max([x["accepted_len"] for x in pickled_data["stats_each_round"]])

        logging.info(f"{Colors.BOLD}--- [Problem {problem_id}, {drafter_name}] Statistics ---{Colors.RESET}")
        logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] Acceptance rate: {acceptance_rate * 100:.1f}% ({accepted_tokens}/{drafted_tokens}){Colors.RESET}")
        logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] Accepted/speculated: avg {avg_acc_len:.2f}/{avg_spec_len:.2f}, max {max_acc_len}/{max_spec_len}{Colors.RESET}")  # could be from different rounds!
        
        # compute e2e latency speedup
        total_output_tokens = len(current_token_ids)
        logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] Avg fwd passes/round: {total_num_forward_passes / num_speculation_rounds:.2f} ({total_num_forward_passes}/{num_speculation_rounds}) (total output tokens: {total_output_tokens}){Colors.RESET}")
        for hardware in args.latency.keys():
            latency_draft = total_num_forward_passes * args.latency[hardware]["draft_fwd_pass"]  # ms
            latency_target = num_speculation_rounds * args.latency[hardware]["target_tpt"][args.target_model_name_clean]
            total_tpt = latency_draft + latency_target
            avg_tpt = total_tpt / total_output_tokens
            speedup = args.latency[hardware]["target_tpt"][args.target_model_name_clean] / avg_tpt
            logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] [{hardware}] Speedup: {speedup:.2f}x (Drafter ratio {latency_draft / total_tpt * 100:.1f}% ({latency_draft:.1f}ms/{total_tpt:.1f}ms); Avg TPT of SD: {avg_tpt:.2f}ms) (num output tokens: {total_output_tokens}){Colors.RESET}")
            
            if draft_type == "ar" and ar_drafter_speedup[hardware] is None:
                ar_drafter_speedup[hardware] = speedup
            if ar_drafter_speedup[hardware] is not None:
                logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] [{hardware}] Win over AR drafter: {speedup / ar_drafter_speedup[hardware]:.3f}x.{Colors.RESET}")

        # save and visualize results
        stats_each_round = pickled_data["stats_each_round"]
        if args.overwrite:
            visualize_acc_rate_over_time(stats_each_round, spec_len=args.spec_len, acceptance_rate=acceptance_rate, output_dir=output_dir_figures, filename=f"{drafter_name}")
        else:
            visualize_acc_rate_over_time(stats_each_round, spec_len=args.spec_len, acceptance_rate=acceptance_rate, output_dir=None, filename=None)
        
        print_sd_trajectory(pickled_data, target_tokenizer)
        
        pickled_data["num_speculation_rounds"] = num_speculation_rounds
        pickled_data["total_num_forward_passes"] = total_num_forward_passes
        pickled_data["accepted_tokens"] = accepted_tokens
        pickled_data["drafted_tokens"] = drafted_tokens
        pickled_data["rejected_tokens"] = rejected_tokens
        pickled_data["acceptance_rate"] = acceptance_rate
        pickled_data["total_output_tokens"] = total_output_tokens  # not necessarily equal to num_target_tokens
        
        if (args.overwrite and not args.read_pickle) or (not os.path.exists(os.path.join(output_dir_pickles, f"{args.max_new_tokens}.pickle"))):
            with open(os.path.join(output_dir_pickles, f"{args.max_new_tokens}.pickle"), "wb") as f:
                pickle.dump(pickled_data, f)
            with open(os.path.join(output_dir_pickles, f"{args.max_new_tokens}.txt"), "w") as f:
                pp = pprint.PrettyPrinter(width=1000, stream=f)  # large enough to fit list
                pp.pprint(pickled_data)

# %%
