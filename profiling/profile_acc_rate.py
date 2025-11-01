# %%
import os
import sys
import time
import torch
import openai
import transformers
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
# transformers.logging.set_verbosity_info()
transformers.logging.set_verbosity_error()
transformers.set_seed(42)

class Colors:
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def is_notebook():
    """Detect if running inside Jupyter or IPython kernel."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ('ZMQInteractiveShell',)  # Jupyter/IPython
    except Exception:
        return False

def is_interactive():
    return is_notebook() or sys.stdout.isatty()
    
system_prompt = """
Solve the following math problem efficiently and clearly. Please reason step by step, 
separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
Problem: {problem}
"""

dataset_name = "aime"

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
draft_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
target_model_name = "Qwen/Qwen2.5-32B-Instruct"
# target_model_name = "Qwen/Qwen2.5-7B-Instruct"
dllm_name = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"

target_model = AutoModelForCausalLM.from_pretrained(
    target_model_name,
    torch_dtype="auto",
    device_map="auto"
)
target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_name,
    torch_dtype="auto",
    device_map="auto"
)
# draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
draft_tokenizer = target_tokenizer  # assume they use the same tokenizer
# NOTE(ruipan): maybe they should use the same tokenizer?
dllm = AutoModelForCausalLM.from_pretrained(
    dllm_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
dllm_tokenizer = AutoTokenizer.from_pretrained(dllm_name, trust_remote_code=True)


# %%
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
        max_new_tokens=16,  # was 512 in vanilla sd experiments
        # use greedy decoding, not sampling
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0.0,
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
        temperature=0.0,
        top_p=1.0,
        top_k=0.0,
    )
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    
    return generated_ids.tolist()



def get_next_n_tokens_dllm(dllm, orig_model_inputs, token_ids_so_far, n, output_seqlen=512, small_block_size=8, threshold=0.9):
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
        # max_new_tokens=output_seqlen,
        max_new_tokens=32,  # NOTE(ruipan): setting this to 8 will not lead to new tokens hmm
        small_block_size=small_block_size,
        threshold=threshold,
        # use greedy decoding, not sampling
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0.0,
    )
    
    full_output_seqlen = generated_ids.shape[1]
    assert full_output_seqlen > num_tokens_in_prompt + len(token_ids_so_far), f"full_output_seqlen {full_output_seqlen}, num_tokens_in_prompt {num_tokens_in_prompt}, len(token_ids_so_far) {len(token_ids_so_far)}"
    generated_ids = generated_ids[0][len(new_model_inputs["input_ids"][0]):]
    generated_ids = generated_ids.tolist()[:n]  # only take the next n tokens
    
    return generated_ids


# %%
# ChatGPT generated. Looks a bit sus? Total generation length on
# q0 is ~512, whereas the original target generation length is ~200...
total_accepted_tokens = 0
total_rejected_tokens = 0
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

    # === LIVE VERIFICATION CHANGE START ===
    # Build initial input for both models (we no longer pre-generate the target sequence)
    text = target_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    orig_model_inputs = target_tokenizer([text], return_tensors="pt").to(target_model.device)

    # We will terminate either on EOS or after this many target tokens (safety cap)
    max_target_tokens = 512

    current_token_ids = []  # prefix tokens accepted so far
    accepted_tokens = 0
    rejected_tokens = 0

    if is_interactive():
        inner_bar = tqdm(total=max_target_tokens, miniters=25,
                         desc=f"Verification (Problem {problem_id})",
                         position=1, leave=True, dynamic_ncols=False, file=sys.stdout)

    n = 5  # speculative length proposed by draft model each round
    num_speculation_rounds = 0

    # Main speculative loop: propose n tokens from draft, verify them live with target model
    while len(current_token_ids) < max_target_tokens:
        num_speculation_rounds += 1

        # Get next n speculative tokens from draft model based on current accepted prefix
        draft_proposal = get_next_n_tokens(draft_model, orig_model_inputs, current_token_ids, n=n)

        print(f"\nSpeculation round {num_speculation_rounds}, current length: {len(current_token_ids)}")
        print(f"draft_proposal {draft_proposal}")

        # Verify each proposed token by querying the target model's next-token argmax (live)
        for i, draft_tok in enumerate(draft_proposal):
            # Build prefix (original prompt + already-accepted tokens)
            prefix = orig_model_inputs["input_ids"][0].tolist() + current_token_ids
            input_ids = torch.tensor([prefix], device=target_model.device)

            with torch.no_grad():
                # compute target model's next-token logits given the current accepted prefix
                outputs = target_model(input_ids=input_ids)
                logits = outputs.logits[:, -1, :]
                target_next_tok = int(torch.argmax(logits, dim=-1).item())

            print(f"Spec round {num_speculation_rounds}, token index {i}, draft_tok {draft_tok}, target_tok {target_next_tok}")

            # Compare draft token to target model's argmax for the next token
            if draft_tok == target_next_tok:
                accepted_tokens += 1
                current_token_ids.append(draft_tok)
                if is_interactive():
                    inner_bar.update(1)
                # If target predicted EOS, stop
                if draft_tok == target_tokenizer.eos_token_id:
                    print("Target predicted EOS — finishing early.")
                    break
            else:
                # On first mismatch, we consider all remaining tokens in the proposal rejected
                rejected_tokens += (n - i)
                # Sync by appending the target model's actual next token
                current_token_ids.append(target_next_tok)
                if is_interactive():
                    inner_bar.update(1)
                # Stop verifying this draft proposal; go back to asking draft for next block
                break

        # Termination conditions:
        # - we hit EOS (detected above),
        # - or we've reached the safety cap (max_target_tokens),
        # - or we have no progress (defensive, unlikely)
        if len(current_token_ids) >= max_target_tokens:
            print(f"Reached max_target_tokens={max_target_tokens}; stopping.")
            break
        if (len(current_token_ids) > 0) and (current_token_ids[-1] == target_tokenizer.eos_token_id):
            # last appended token was EOS
            break

    if is_interactive():
        inner_bar.close()

    # Compute token acceptance rate for this problem
    denom = (accepted_tokens + rejected_tokens)
    if denom == 0:
        acceptance_rate = 0.0
    else:
        acceptance_rate = accepted_tokens / denom

    print(f"{Colors.YELLOW}Problem {problem_id} acceptance rate: {acceptance_rate:.3f}{Colors.RESET}")
    print(f"Accepted: {accepted_tokens}, Rejected: {rejected_tokens}, Total considered: {denom}")
    print(f"Number of speculation rounds: {num_speculation_rounds}")

    total_accepted_tokens += accepted_tokens
    total_rejected_tokens += rejected_tokens

# Overall acceptance rate across problems
overall_denom = (total_accepted_tokens + total_rejected_tokens)
if overall_denom == 0:
    overall_acceptance_rate = 0.0
else:
    overall_acceptance_rate = total_accepted_tokens / overall_denom
print(f"{Colors.CYAN}Overall acceptance rate: {overall_acceptance_rate:.3f}{Colors.RESET}")
# === LIVE VERIFICATION CHANGE END ===






# %%
total_accepted_tokens = 0
total_rejected_tokens = 0

# for problem_id in tqdm(range(50), desc="Problems", position=0):
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
    
    # Build initial input for both models
    text = target_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    orig_model_inputs = target_tokenizer([text], return_tensors="pt").to(target_model.device)
    current_token_ids = []


    n = 5  # number of speculative tokens proposed each time
    accepted_tokens = 0
    rejected_tokens = 0
    current_token_ids = []  # prefix tokens generated so far

    num_speculation_rounds = 0
    while len(current_token_ids) < len(target_ids):
        num_speculation_rounds += 1
        
        # Get next n speculative tokens from draft model
        draft_proposal = get_next_n_tokens(draft_model, orig_model_inputs, current_token_ids, n=n)
        # draft_proposal = get_next_n_tokens_dllm(dllm, orig_model_inputs, current_token_ids, n=n)
        
        # The corresponding slice of ground-truth target tokens
        target_slice = target_ids[len(current_token_ids): len(current_token_ids) + n]
        
        print(f"Speculation round {num_speculation_rounds}, current length: {len(current_token_ids)}")
        print(f"target_slice {target_slice}, draft_proposal {draft_proposal}")

        # Compare draft proposal with target tokens one by one
        for i, (draft_tok, target_tok) in enumerate(zip(draft_proposal, target_slice)):
            print(f"Spec round {num_speculation_rounds}, token index {i}, draft_tok {draft_tok}, target_tok {target_tok}")
            if draft_tok == target_tok:
                accepted_tokens += 1
                if is_interactive():
                    inner_bar.update(1)
                current_token_ids.append(draft_tok)
            else:
                rejected_tokens += (n - i)  # all remaining tokens in this proposal are rejected
                # replace with correct target token, sync with target model
                current_token_ids.append(target_tok)
                # print(f"Rejection, current length: {len(current_token_ids)}, draft_tok {draft_tok}, target_tok {target_tok}")
                # print(f"draft token decoded: {draft_tokenizer.decode(draft_tok)}")
                # print(f"target token decoded: {target_tokenizer.decode(target_tok)}")
                break  # speculative generation diverged; go back to draft proposal step
            
                # FIXME(ruipan): math, question 1, len(target_ids) = 235, strange mismatch at len 148
            # print(f"Progress: len(current_token_ids) = {len(current_token_ids)}")
        
        if draft_proposal == target_slice:  # this logic seems to be throwing off the acceptance rate by a lot
            free_token_index = len(current_token_ids) + n
            if free_token_index >= len(target_ids):
                continue  # no more free tokens to add
            current_token_ids.append(target_ids[free_token_index])
            # accepted_tokens += 1  # NOTE(ruipan): should this free lunch token count as an accepted token?
            if is_interactive():
                inner_bar.update(1)

        # If we’ve already matched the full target sequence, stop
        if len(current_token_ids) >= len(target_ids):
            break

    if is_interactive():
        inner_bar.close()

    # Compute token acceptance rate
    acceptance_rate = accepted_tokens / (accepted_tokens + rejected_tokens)
    print(f"{Colors.YELLOW}Problem {problem_id} acceptance rate: {acceptance_rate:.3f}{Colors.RESET}")
    print(f"Accepted: {accepted_tokens}, Rejected: {rejected_tokens}, Total: {accepted_tokens + rejected_tokens}")
    print(f"Number of speculation rounds: {num_speculation_rounds}")
    total_accepted_tokens += accepted_tokens
    total_rejected_tokens += rejected_tokens

overall_acceptance_rate = total_accepted_tokens / (total_accepted_tokens + total_rejected_tokens)
print(f"Overall acceptance rate: {overall_acceptance_rate:.3f}")

# %%
