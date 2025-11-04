# %%
import os
import sys
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk

# Set seeds and logging for reproducibility
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
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False

def is_interactive():
    return is_notebook() or sys.stdout.isatty()

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
# --- Helper Functions ---

# %%
def get_full_target_generation(model, tokenizer, messages, target_len):
    """Generates the ground-truth token sequence using the target model's standard .generate() method."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print(f"Number of prompt tokens: {model_inputs.input_ids.shape[1]}")
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=target_len,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0.0,
    )
    return generated_ids[0][model_inputs.input_ids.shape[1]:].tolist(), model_inputs

def get_speculative_tokens(model, tokenizer, orig_inputs, prefix_ids, n):
    """Generates n speculative tokens from the draft model given the current context."""
    if not prefix_ids:
        # Use the original prompt if the prefix is empty
        input_ids = orig_inputs['input_ids']
    else:
        prefix_tensor = torch.tensor([prefix_ids], device=model.device, dtype=torch.long)
        input_ids = torch.cat([orig_inputs['input_ids'], prefix_tensor], dim=1)

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=n,
        do_sample=False,
        # pad_token_id=tokenizer.eos_token_id,
        temperature=0.0,
        top_p=1.0,
        top_k=0.0,
    )
    return generated_ids[0][input_ids.shape[1]:].tolist()

# --- Main Logic ---
for problem_id in range(1):
    problem = dataset["problem"][problem_id]
    messages = [{"role": "user", "content": system_prompt.format(problem=problem)}]

    # 1. Generate the ground-truth answer for verification
    # Let's generate a shorter sequence for faster debugging
    TARGET_LEN = 512
    target_ids, orig_model_inputs = get_full_target_generation(target_model, target_tokenizer, messages, target_len=TARGET_LEN)
    print(f"Target (vanilla) generation length: {len(target_ids)} tokens")
    print(f"Target token IDs: {target_ids}")

    # 2. Begin the speculative decoding loop
    current_token_ids = []
    accepted_tokens = 0
    rejected_tokens = 0
    num_speculation_rounds = 0
    n = 5 # Number of tokens to speculate each round

    if is_interactive():
        progress_bar = tqdm(total=TARGET_LEN, desc=f"Speculative Generation (Problem {problem_id})")

    while len(current_token_ids) < TARGET_LEN:
        num_speculation_rounds += 1
        
        # if num_speculation_rounds == 2:
        #     break  # debugging

        # A. PROPOSE: Get n speculative tokens from the draft model
        draft_proposal = get_speculative_tokens(draft_model, target_tokenizer, orig_model_inputs, current_token_ids, n)
        
        if not draft_proposal: # Stop if the draft model has nothing to say
            break
        
        print(f"\nRound {num_speculation_rounds}: Draft proposal is {draft_proposal}")

        # B. VERIFY (Single Forward Pass)
        # The input for verification includes the prompt, the accepted prefix, AND the draft proposal
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

        # C. ACCEPT/REJECT
        accepted_len = 0
        for i in range(len(draft_proposal)):
            # The target's prediction for position `i` is the argmax of the logits at position `i`
            target_pred = torch.argmax(verify_logits[i, :], dim=-1).item()
            print(f"  Verifying draft token {draft_proposal[i]} at position {i}. Target would have picked {target_pred}.")

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
            print(f"  All draft tokens accepted! Bonus token is {final_token}.")

        # D. UPDATE
        tokens_to_append = draft_proposal[:accepted_len] + [final_token]
        current_token_ids.extend(tokens_to_append)
        
        accepted_tokens += accepted_len
        rejected_tokens += len(draft_proposal) - accepted_len
        
        if is_interactive():
            progress_bar.update(len(tokens_to_append))

        if target_tokenizer.eos_token_id in tokens_to_append:
            break
            
    if is_interactive():
        progress_bar.close()

    # --- Verification and Stats ---
    final_spec_ids = current_token_ids[:len(target_ids)]

    print(f"\n{Colors.BOLD}--- Verification ---{Colors.RESET}")
    print(f"Target (vanilla) IDs: {target_ids}")
    print(f"Final speculative IDs:  {final_spec_ids}")
    if final_spec_ids != target_ids:
        print(f"{Colors.RED}Warning: Mismatch between speculative and vanilla decoding!{Colors.RESET}")
    else:
        print(f"{Colors.GREEN}Success: Speculative decoding output exactly matches vanilla decoding output.{Colors.RESET}")

    denom = num_speculation_rounds * n
    acceptance_rate = accepted_tokens / denom if denom > 0 else 0.0

    print(f"\n{Colors.BOLD}--- Statistics ---{Colors.RESET}")
    print(f"{Colors.YELLOW}Problem {problem_id} draft token acceptance rate: {acceptance_rate:.3f}. Accepted: {accepted_tokens}, Rejected: {rejected_tokens}, Total Drafted: {denom}.{Colors.RESET}")
    print(f"Number of speculation rounds: {num_speculation_rounds}")

# %%
print(target_ids)
print(target_tokenizer.decode(target_ids))
# %%
print(current_token_ids)
print(target_tokenizer.decode(current_token_ids))
# %%
