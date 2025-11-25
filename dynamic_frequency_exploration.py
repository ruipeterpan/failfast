# %%
import os
import sys
import copy
import time
import torch
import openai
import pickle
import pprint
import logging
import argparse
import transformers
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# transformers.logging.set_verbosity_debug()
# transformers.logging.set_verbosity_info()
# transformers.logging.set_verbosity_warning()
transformers.logging.set_verbosity_error()

sys.path.insert(1, os.path.dirname(os.getcwd()))
from plotting import (
    visualize_acc_rate_over_time,
    get_boolean_decision_from_stats_per_round,
)
from utils import (
    calculate_spec_decoding_speedup,
    print_sd_trajectory,
)

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
    logging.debug(f"num_input_tokens {num_input_tokens}, first eight tokens: {model_inputs.input_ids[0, :8].tolist()}")
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,  # was 512 in vanilla sd experiments
        # use greedy decoding, not sampling
        do_sample=False,  # overrides all below sampling params, but setting them just in case
        # temperature=0.0,
        # top_p=1.0,
        # top_k=0.0,
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
        # use greedy decoding, not sampling
        do_sample=False,
        # temperature=0.0,
        # top_p=1.0,
        # top_k=0.0,
    )
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    
    return generated_ids.tolist()



def get_next_n_tokens_dllm(dllm, args, orig_model_inputs, token_ids_so_far, veri_freq, output_seqlen, small_block_size, threshold, is_drafter, prev_prefill_output=None):
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
            veri_freq=veri_freq,
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
            veri_freq=veri_freq,
            return_prefill_kvs=True,
            prev_prefill_output=prev_prefill_output,
            args=args,
        )
    
    full_output_seqlen = generated_ids.shape[1]
    assert full_output_seqlen > num_tokens_in_prompt + len(token_ids_so_far), f"full_output_seqlen {full_output_seqlen}, num_tokens_in_prompt {num_tokens_in_prompt}, len(token_ids_so_far) {len(token_ids_so_far)}"
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    generated_ids = generated_ids.tolist()[:veri_freq]  # only take the next n tokens
    
    if any(x in generated_ids for x in [151665, 151645]):
        special_token = "MASK" if 151665 in generated_ids else "STOP"
        logging.info(f"{Colors.RED}Generated ids contain {special_token} tokens! {generated_ids}{Colors.RESET}")
    
    if not args.disable_reusing_drafter_kvs:
        return generated_ids, prefill_output, num_forward_passes, forward_pass_latencies
    return generated_ids, num_forward_passes, forward_pass_latencies


def get_next_tokens_dllm(dllm, args, orig_model_inputs, token_ids_so_far, veri_freq, output_seqlen, small_block_size, threshold, is_drafter, prev_prefill_output=None):
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
        generated_ids, spec_len, num_forward_passes, forward_pass_latencies = dllm.generate_draft_tokens_arbitrary_length(
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
            veri_freq=veri_freq,
            return_prefill_kvs=False,
            args=args,
        )
    else:
        generated_ids, spec_len, prefill_output, num_forward_passes, forward_pass_latencies = dllm.generate_draft_tokens_arbitrary_length(
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
            veri_freq=veri_freq,
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
        return generated_ids, spec_len, prefill_output, num_forward_passes, forward_pass_latencies
    return generated_ids, spec_len, num_forward_passes, forward_pass_latencies


def construct_drafter_configs(args):
    drafter_configs = []
    if args.run_ar:
        drafter_configs.extend([("ar", None, "sf")] )
    drafter_configs.extend([("dllm", thr, "sf") for thr in args.drafter_thresholds])
    drafter_configs.extend([("dllm", thr, "df") for thr in args.drafter_thresholds])
    args.drafter_configs = drafter_configs

def format_drafter_name(args, draft_type, drafter_threshold, freq_scheme):
    if draft_type == "ar":
        return "ar_None_sf"
    
    if freq_scheme == "sf":
        return f"dllm_{drafter_threshold}_sf"
    
    if freq_scheme == "df":
        return f"dllm_{drafter_threshold}_df"


# %%
parser = argparse.ArgumentParser(description="Profiles the acceptance rate of speculative decoding within a single query.")
parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gpqa"], default="math",
                    help="Dataset")
parser.add_argument("--output_dir", type=str, default="/data2/ruipan/diffspec", 
                    help="Where result pickle files (and output figures) will be written to")
parser.add_argument("--target_model_name", type=str, default="Qwen/Qwen2.5-32B-Instruct", 
                    help="Name of the base model to use")
parser.add_argument("--dllm_dir", type=str, default=None, 
                    help="Dir to the dLLM weights and (modified) modeling.py")
parser.add_argument("--num_questions", type=int, default=1,
                    help="Number of questions to run profiling on")
parser.add_argument("--max_new_tokens", type=int, default=512,
                    help="Max new tokens from the target model")
parser.add_argument("--block_size", type=int, default=32,
                    help="Block size in Fast-dLLM")
parser.add_argument("--veri_freq", type=int, default=10,
                    help="Frequency of verification steps (in number of tokens)")
parser.add_argument("--drafter_thresholds", type=float, nargs="+",  # one or more float thresholds
                    default=[0.05],
                    help="Threshold for confidence-adaptive decoding of the dLLM drafter model (e.g., --drafter_thresholds 0.1 0.5 0.9)")
parser.add_argument("--log_level",
                    type=str,
                    default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Set the logging level")
parser.add_argument("--v1_multiplicative_factors", type=float, nargs="+",
                    default=[1.8],
                    help="How much to bump up/down the frequency based on acceptance rate")
parser.add_argument("--v1_lower_bound_factors", type=float, nargs="+",
                    default=[0.8],
                    help="Lower bound factor for the frequency adjustment")
parser.add_argument('--run_ar', action='store_true', help='Run the AR drafter to compare speedups')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output pickles and figures')
parser.add_argument('--disable_reusing_drafter_kvs', action='store_true', help='Disables reusing drafter KV cache across verification rounds')
parser.add_argument('--read_pickle', action='store_true', help='Use acceptance decisions from a cached pickle file rather than rerunning')
args, _ = parser.parse_known_args()


######custom fields for easier debugging######
args.log_level = "DEBUG"
# args.overwrite = False
# # args.disable_reusing_drafter_kvs = True
# args.run_ar = True
# args.num_questions = 5
# args.veri_freq = 10
# args.read_pickle = True  # TODO: read trajectory from pickle as well
# # args.drafter_thresholds = [0.9, 0.7, 0.5, 0.3, 0.1, 0.01]
# args.drafter_thresholds = [0.05]
# args.drafter_thresholds = [0.05]
args.target_model_name = "Qwen/Qwen2.5-7B-Instruct"  # for easier debugging
args.dllm_dir = "/data2/ruipan/Fast_dLLM_v2_1.5B"
######custom fields for easier debugging######


logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%m%d %H:%M:%S",
    # datefmt="%m%d",
)

construct_drafter_configs(args)  # populates args.drafter_configs

dataset = get_dataset(args.dataset_name)
args.latency = {  # a6000, hf generate latencies
    "draft_fwd_pass": 28,  # ms; dLLM 1.5B drafter forward pass latency
    "target_tpt": 105,  # ms; Qwen2.5-32B, latency of short prefill pass (~=tpt)
}
args.latency_vllm = {  # a6000, vllm latencies (assuming dllm latency is similar to 1.5b ar)
    "draft_fwd_pass": 6.1,  # ms; dLLM 1.5B drafter forward pass latency
    "target_tpt": 52.6,  # ms; Qwen2.5-32B, latency of short prefill pass (~=tpt)
}


target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
args.target_tokenizer = target_tokenizer

# %%
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
# for problem_id in tqdm(range(args.num_questions), desc="Problems", position=0):
for problem_id in [12]:
    transformers.set_seed(42)  # reproducibility for each question-model-model config pairing
    problem, options = format_problem_and_options(args, problem_id)
    messages = [
        {"role": "user", "content": get_first_user_msg(problem, options)},
    ]
    
    if not args.read_pickle:
        target_ids, orig_model_inputs = get_target_token_ids(target_model, target_tokenizer, messages, max_new_tokens=args.max_new_tokens)
        num_target_tokens = len(target_ids)
        logging.info(f"[Problem {problem_id}] Target (vanilla) generation length: {num_target_tokens} tokens")
    
    ar_drafter_speedup = None
    for draft_type, drafter_threshold, freq_scheme in args.drafter_configs:
        transformers.set_seed(42)  # reproducibility for each question-model-model config pairing
        
        # set up output dirs and export
        if draft_type == "ar":
            output_dir_figures = os.path.join(args.output_dir, "figures", "acc_rate_within_query", args.dataset_name, str(problem_id))
            output_dir_pickles = os.path.join(args.output_dir, "pickles", "detailed_info", args.dataset_name, str(problem_id), "ar")
        elif draft_type == "dllm":
            output_dir_figures = os.path.join(args.output_dir, "figures", "acc_rate_within_query", args.dataset_name, str(problem_id))
            output_dir_pickles = os.path.join(args.output_dir, "pickles", "detailed_info", args.dataset_name, str(problem_id), f"{draft_type}_{drafter_threshold}")
        for d in [output_dir_figures, output_dir_pickles]:
            os.makedirs(d, exist_ok=True)
        
        if args.read_pickle:
            if not os.path.exists(os.path.join(output_dir_pickles, f"{draft_type}_{drafter_threshold}.pickle")):
                logging.warning(f"{Colors.RED}No cached pickle found for {draft_type}_{drafter_threshold}!{Colors.RESET}")
            
            with open(os.path.join(output_dir_pickles, f"{draft_type}_{drafter_threshold}.pickle"), "rb") as f:
                pickled_data = pickle.load(f)
            
            accepted_tokens = pickled_data["accepted_tokens"]
            rejected_tokens = pickled_data["rejected_tokens"]
            num_speculation_rounds = pickled_data["num_speculation_rounds"]
            total_num_forward_passes = pickled_data["total_num_forward_passes"]
            current_token_ids = pickled_data["stats_per_round"][-1]["current_token_ids"]
        else:  # run the actual spec decoding pipeline
            drafter_name = format_drafter_name(args, draft_type, drafter_threshold, freq_scheme)
            logging.info(f"{Colors.BOLD}=== [Problem {problem_id}] Running drafter: {drafter_name} ==={Colors.RESET}")
            accepted_tokens = 0
            rejected_tokens = 0
            num_speculation_rounds = 0
            total_num_forward_passes = 0
            current_token_ids = []  # prefix tokens generated so far
            prev_prefill_output = None  # drafter's prefill KVs
            pickled_data = {
                "orig_model_inputs": orig_model_inputs,
                "target_ids": target_ids,
                "problem": problem,
                "options": options,
                "num_target_tokens": num_target_tokens,
                "stats_per_round": [],
            }

            if is_interactive():
                inner_bar = tqdm(total=num_target_tokens, miniters=25, desc=f"Verification (Problem {problem_id})",
                                position=1, leave=True, dynamic_ncols=False, file=sys.stdout)

            freq_so_far = []
            acc_rate_so_far = []

            while len(current_token_ids) < len(target_ids):
                logging.debug(f"--- [{draft_type}_{drafter_threshold}_{freq_scheme}] Speculation round {num_speculation_rounds} ---")
                current_token_ids_snapshot = copy.deepcopy(current_token_ids)
                
                # A. PROPOSE: Get next n speculative tokens from draft model based on current accepted prefix
                if draft_type == "ar":
                    draft_proposal = get_next_n_tokens_ar(draft_model, orig_model_inputs, current_token_ids, n=args.veri_freq)
                    num_forward_passes = args.veri_freq  # 1 fwd pass per token for AR drafter
                    veri_freq = args.veri_freq
                elif draft_type == "dllm":
                    if freq_scheme == "sf":  # static frequency
                        veri_freq = args.veri_freq
                        if args.disable_reusing_drafter_kvs:
                            draft_proposal, num_forward_passes, forward_pass_latencies = get_next_n_tokens_dllm(dllm, args, orig_model_inputs, current_token_ids, 
                                                                    veri_freq=veri_freq,  # number of speculative tokens proposed each time
                                                                    output_seqlen=2*args.block_size,  # 2 blocks of 32. Ensures veri_freq tokens are generated in case they span over two blocks
                                                                    small_block_size=8,
                                                                    threshold=drafter_threshold,
                                                                    is_drafter=True,)
                        else:
                            draft_proposal, prefill_output, num_forward_passes, forward_pass_latencies = get_next_n_tokens_dllm(dllm, args, orig_model_inputs, current_token_ids, 
                                                                    veri_freq=veri_freq,  # number of speculative tokens proposed each time
                                                                    output_seqlen=2*args.block_size,  # 2 blocks of 32. Ensures veri_freq tokens are generated in case they span over two blocks
                                                                    small_block_size=8,
                                                                    threshold=drafter_threshold,
                                                                    is_drafter=True,
                                                                    prev_prefill_output=prev_prefill_output)
                            prev_prefill_output = prefill_output
                    else:  # dynamic frequency: drafter determines how many tokens to propose
                        # TODO
                        draft_proposal, spec_len, prefill_output, num_forward_passes, forward_pass_latencies = get_next_tokens_dllm(dllm, args, orig_model_inputs, current_token_ids, 
                                                                    veri_freq=veri_freq,  # number of speculative tokens proposed each time
                                                                    output_seqlen=2*args.block_size,  # 2 blocks of 32. Ensures veri_freq tokens are generated in case they span over two blocks
                                                                    small_block_size=8,
                                                                    threshold=drafter_threshold,
                                                                    is_drafter=True,
                                                                    prev_prefill_output=prev_prefill_output)
                        prev_prefill_output = prefill_output
                        veri_freq = spec_len  # update veri_freq to the actual number of tokens proposed
                        
                        
                total_num_forward_passes += num_forward_passes
                # print(f"forward_pass_latencies {forward_pass_latencies}")  # NOTE(ruipan): seems to be similar to TPT of 1.5B AR model
                
                if not draft_proposal: # Stop if the draft model has nothing to say
                    logging.info(f"{Colors.RED}Warning: Draft model returned no tokens{Colors.RESET}")
                    break
                
                if len(draft_proposal) < veri_freq:
                    logging.info(f"{Colors.RED}Warning: Draft model returned fewer tokens ({len(draft_proposal)}) than veri_freq ({veri_freq})!{Colors.RESET}")
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
                
                # C. ACCEPT/REJECT
                accepted_len = 0
                for i in range(len(draft_proposal)):
                    # The target's prediction for position `i` is the argmax of the logits at position `i`
                    target_pred = torch.argmax(verify_logits[i, :], dim=-1).item()
                    # print(f"  Verifying draft token at position {i}: {draft_proposal[i]}. Target would have picked {target_pred}.")

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
                    # print(f"  All draft tokens accepted! Bonus token is {final_token}.")
                
                proposed_tokens_str = ""
                for i in range(accepted_len):
                    proposed_tokens_str += args.target_tokenizer.decode([draft_proposal[i]])
                proposed_tokens_str += f"{Colors.RED}{Colors.STRIKETHROUGH}"
                for i in range(veri_freq - accepted_len):
                    if i + accepted_len >= len(draft_proposal):
                        break
                    proposed_tokens_str += args.target_tokenizer.decode([draft_proposal[i + accepted_len]])
                proposed_tokens_str += f"{Colors.RESET}"
                proposed_tokens_str += f"{Colors.GREEN}{args.target_tokenizer.decode([final_token])}{Colors.RESET}"
                if accepted_len != veri_freq:
                    logging.debug(f"--- Speculation round {num_speculation_rounds} proposed token IDs {draft_proposal} ---")
                else:
                    logging.debug(f"--- Speculation round {num_speculation_rounds} proposed token IDs {draft_proposal}, bonus token {final_token} ---")
                logging.debug(f"--- Speculation round {num_speculation_rounds} proposed str: {proposed_tokens_str} ---")
                logging.debug(f"--- Speculation round {num_speculation_rounds}: acceptance rate {accepted_len / veri_freq:.2f} ({accepted_len}/{veri_freq}) ---")
                logging.debug(f"--- Speculation round {num_speculation_rounds} num_forward_passes: {num_forward_passes} ---")
                
                acc_rate_so_far.append(accepted_len / veri_freq)
                
                # D. UPDATE
                tokens_to_append = draft_proposal[:accepted_len] + [final_token]
                current_token_ids.extend(tokens_to_append)
                
                accepted_tokens += accepted_len
                rejected_tokens += len(draft_proposal) - accepted_len
                
                info_this_round = {
                    "current_token_ids": current_token_ids_snapshot,
                    "target_tokens": target_tokens,
                    "draft_proposal": draft_proposal,
                    "accepted_len": accepted_len,
                    "prefix_len": prefix_len,
                    "veri_freq": veri_freq,
                }
                pickled_data["stats_per_round"].append(info_this_round)
                
                num_speculation_rounds += 1
                
                if is_interactive():
                    inner_bar.update(len(tokens_to_append))

                if target_tokenizer.eos_token_id in tokens_to_append:
                    break

            if is_interactive():
                inner_bar.close()

        # Compute token acceptance rate
        # drafted_tokens = num_speculation_rounds * args.veri_freq
        drafted_tokens = sum([x["veri_freq"] for x in pickled_data["stats_per_round"]])
        acceptance_rate = accepted_tokens / drafted_tokens
        logging.info(f"{Colors.BOLD}--- [Problem {problem_id}, {drafter_name}] Statistics ---{Colors.RESET}")
        logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] Acceptance rate: {acceptance_rate * 100:.1f}% ({accepted_tokens}/{drafted_tokens}){Colors.RESET}")
        
        # compute e2e latency speedup
        logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] Avg fwd passes/round: {total_num_forward_passes / num_speculation_rounds:.2f} ({total_num_forward_passes}/{num_speculation_rounds}) (total output tokens: {len(current_token_ids)}){Colors.RESET}")
        
        latency_draft = total_num_forward_passes * args.latency["draft_fwd_pass"]  # ms
        latency_target = num_speculation_rounds * args.latency["target_tpt"]
        total_tpt = latency_draft + latency_target
        avg_tpt = total_tpt / len(current_token_ids)
        speedup = args.latency["target_tpt"] / avg_tpt
        logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] [HuggingFace] Speedup: {speedup:.2f}x (Drafter latency ratio {latency_draft / total_tpt * 100:.1f}%; Avg TPT of SD: {avg_tpt:.2f}ms){Colors.RESET}")
        
        if draft_type == "ar" and ar_drafter_speedup is None:
            ar_drafter_speedup = speedup
        if ar_drafter_speedup is not None:
            logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] [HuggingFace] Win over AR drafter: {speedup / ar_drafter_speedup:.3f}x.{Colors.RESET}")
        
        latency_draft = total_num_forward_passes * args.latency_vllm["draft_fwd_pass"]  # ms
        latency_target = num_speculation_rounds * args.latency_vllm["target_tpt"]
        total_tpt = latency_draft + latency_target
        avg_tpt = total_tpt / len(current_token_ids)
        speedup = args.latency_vllm["target_tpt"] / avg_tpt
        logging.info(f"{Colors.CYAN}[Problem {problem_id}, {drafter_name}] [vLLM] Speedup: {speedup:.2f}x (Drafter latency ratio {latency_draft / total_tpt * 100:.1f}%; Avg TPT of SD: {avg_tpt:.2f}ms){Colors.RESET}")

        # save and visualize results
        stats_per_round = pickled_data["stats_per_round"]
        if args.overwrite:
            visualize_acc_rate_over_time(stats_per_round, veri_freq=args.veri_freq, acceptance_rate=acceptance_rate, output_dir=output_dir_figures, filename=f"{drafter_name}")
        else:
            visualize_acc_rate_over_time(stats_per_round, veri_freq=args.veri_freq, acceptance_rate=acceptance_rate, output_dir=None, filename=None)
        
        # print_sd_trajectory(pickled_data, target_tokenizer)
        
        pickled_data["num_speculation_rounds"] = num_speculation_rounds
        pickled_data["total_num_forward_passes"] = total_num_forward_passes
        pickled_data["accepted_tokens"] = accepted_tokens
        pickled_data["rejected_tokens"] = rejected_tokens
        
        if args.overwrite or (not os.path.exists(os.path.join(output_dir_pickles, f"{draft_type}_{drafter_threshold}.pickle"))):
            with open(os.path.join(output_dir_pickles, f"{draft_type}_{drafter_threshold}.pickle"), "wb") as f:
                pickle.dump(pickled_data, f)
            with open(os.path.join(output_dir_pickles, f"{draft_type}_{drafter_threshold}.txt"), "w") as f:
                pp = pprint.PrettyPrinter(width=1000, stream=f)  # large enough to fit list
                pp.pprint(pickled_data)

# %%
