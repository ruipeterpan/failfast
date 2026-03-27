# %%
import os
import sys
import io
import copy
import time
import torch
import pickle
import pprint
import logging
import argparse
import transformers
import importlib
import importlib.util
from argparse import Namespace
from contextlib import redirect_stdout
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

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
        max_new_tokens=max_new_tokens,
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


def get_next_tokens_ar(
    model,
    orig_model_inputs,
    token_ids_so_far,
    n,
    lowconf_threshold,
    max_spec_len,
    incr_len,
):
    """AR version of dynamic-length drafting.

    Spiritually similar to `get_next_tokens_dllm` (dynamic frequency), but implemented
    using an AR model's greedy next-token probability:
    - Greedily speculate tokens in chunks of `incr_len`
    - Stop early if any speculated token has confidence < `lowconf_threshold`
    - Stop once `max_spec_len` is reached (or `n` if `max_spec_len` is None)
    """
    if incr_len is None or incr_len <= 0:
        raise ValueError(f"incr_len must be a positive int, got {incr_len}")
    if max_spec_len is not None and max_spec_len <= 0:
        raise ValueError(f"max_spec_len must be a positive int or None, got {max_spec_len}")
    if lowconf_threshold is None:
        raise ValueError("lowconf_threshold must not be None for get_next_tokens_ar")

    # Back-compat: if caller doesn't provide max_spec_len, use `n` as the cap.
    cap = n if max_spec_len is None else max_spec_len
    if cap <= 0:
        return [], []

    device = orig_model_inputs["input_ids"].device
    drafted = []
    confidences = []

    # Build initial input: prompt + accepted prefix
    current_tokens = torch.tensor(token_ids_so_far, device=device, dtype=torch.long).unsqueeze(0)
    current_mask = torch.ones_like(current_tokens, dtype=torch.long)
    current_inputs = {
        'input_ids': torch.cat([orig_model_inputs['input_ids'], current_tokens], dim=1),
        'attention_mask': torch.cat([orig_model_inputs['attention_mask'], current_mask], dim=1)
    }

    with torch.no_grad():
        while len(drafted) < cap:
            # Generate next chunk of tokens
            chunk_size = min(incr_len, cap - len(drafted))
            
            # Use generate() with output_scores to get logits for confidence checking
            generate_output = model.generate(
                **current_inputs,
                max_new_tokens=chunk_size,
                do_sample=False,  # use greedy decoding
                output_scores=True,
                return_dict_in_generate=True,
            )
            
            # Extract generated token IDs (excluding the input)
            generated_ids = generate_output.sequences[0][len(current_inputs["input_ids"][0]):]
            generated_ids = generated_ids.tolist()
            
            # Check confidence of each generated token
            # scores is a tuple of tensors, one per generated position
            scores = generate_output.scores
            found_lowconf = False
            for i, (token_id, score_logits) in enumerate(zip(generated_ids, scores)):
                probs = torch.softmax(score_logits, dim=-1)
                conf = probs[0, token_id].item()
                drafted.append(token_id)
                confidences.append(conf)
                
                if conf < lowconf_threshold:
                    found_lowconf = True
            
            # If we found a low-confidence token in this chunk, return after processing the whole chunk
            if found_lowconf:
                return drafted, confidences
            
            # Update current_inputs for next iteration (if we haven't hit the cap)
            if len(drafted) < cap:
                # Append all generated tokens to current_inputs for next generate() call
                new_tokens = torch.tensor(generated_ids, device=device, dtype=torch.long).unsqueeze(0)
                new_mask = torch.ones_like(new_tokens, dtype=torch.long)
                current_inputs = {
                    'input_ids': torch.cat([current_inputs['input_ids'], new_tokens], dim=1),
                    'attention_mask': torch.cat([current_inputs['attention_mask'], new_mask], dim=1)
                }

    return drafted, confidences


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
            new_model_inputs["input_ids"],
            max_new_tokens=output_seqlen,
            small_block_size=small_block_size,
            block_size=args.block_size,
            threshold=threshold,
            # use greedy decoding, not sampling
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            # use_block_cache=True,  # NOTE(ruipan): doesn't seem to make a difference in latency, prob because we are running a 1.5B model, which is memory-bound
            is_drafter=is_drafter,
            spec_len=spec_len,
            return_prefill_kvs=False,
            args=args,
        )
    else:
        generated_ids, prefill_output, num_forward_passes, forward_pass_latencies = dllm.generate_draft_tokens(
            # **new_model_inputs,
            new_model_inputs["input_ids"],
            max_new_tokens=output_seqlen,
            small_block_size=small_block_size,
            block_size=args.block_size,
            threshold=threshold,
            # use greedy decoding, not sampling
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            # use_block_cache=True,
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
            block_size=args.block_size,
            threshold=threshold,
            # use greedy decoding, not sampling
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=0.0,
            # use_block_cache=True,
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
            # use_block_cache=True,
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


_FAST_DLLM_V1_MODULES = None


def _load_fast_dllm_v1_modules(args):
    """Lazily import Fast-dLLM v1 generate/modeling modules together."""
    global _FAST_DLLM_V1_MODULES
    if _FAST_DLLM_V1_MODULES is not None:
        return _FAST_DLLM_V1_MODULES

    llada_dir = os.path.join(args.dllm_v1_dir, "llada")
    generate_py = os.path.join(llada_dir, "generate.py")
    modeling_py = os.path.join(llada_dir, "model", "modeling_llada.py")
    if not os.path.exists(generate_py):
        raise FileNotFoundError(
            f"Could not find Fast-dLLM-v1 generate.py at: {generate_py}. "
            "Please set --dllm_v1_dir to the Fast-dLLM repo root."
        )
    if not os.path.exists(modeling_py):
        raise FileNotFoundError(
            f"Could not find Fast-dLLM-v1 modeling file at: {modeling_py}. "
            "Please set --dllm_v1_dir to the Fast-dLLM repo root."
        )

    if llada_dir not in sys.path:
        sys.path.insert(0, llada_dir)

    generate_spec = importlib.util.spec_from_file_location("fast_dllm_v1_generate", generate_py)
    generate_module = importlib.util.module_from_spec(generate_spec)
    generate_spec.loader.exec_module(generate_module)

    modeling_module = importlib.import_module("model.modeling_llada")

    _FAST_DLLM_V1_MODULES = (generate_module, modeling_module.LLaDAModelLM)
    return _FAST_DLLM_V1_MODULES


def _resolve_local_snapshot_path(model_name_or_path):
    """Resolve Hub model id to local cache path when available."""
    if os.path.exists(model_name_or_path):
        return model_name_or_path
    try:
        return snapshot_download(repo_id=model_name_or_path, local_files_only=True)
    except Exception:
        return model_name_or_path


def _sanitize_tokens_for_dllm_v1(token_ids, vocab_size, pad_token_id=126081):
    """Replace out-of-range token ids with pad token id for LLaDA v1."""
    if vocab_size is None:
        return token_ids, 0
    if vocab_size <= 0:
        return token_ids, 0

    safe_pad_id = pad_token_id if 0 <= pad_token_id < vocab_size else (vocab_size - 1)
    invalid_mask = (token_ids < 0) | (token_ids >= vocab_size)
    num_invalid = int(invalid_mask.sum().item())
    if num_invalid == 0:
        return token_ids, 0

    token_ids = token_ids.clone()
    token_ids[invalid_mask] = safe_pad_id
    return token_ids, num_invalid


def get_next_n_tokens_dllm_v1(dllm_v1, args, orig_model_inputs, token_ids_so_far, spec_len, small_block_size, threshold):
    """Get next n speculative tokens from Fast-dLLM-v1 (LLaDA backend)."""
    generator_mod, _ = _load_fast_dllm_v1_modules(args)
    if args.dllm_v1_generate_mode == "generate":
        generator_fn = generator_mod.generate
    elif args.dllm_v1_generate_mode == "prefix_cache":
        generator_fn = generator_mod.generate_with_prefix_cache
    elif args.dllm_v1_generate_mode == "dual_cache":
        generator_fn = generator_mod.generate_with_dual_cache
    else:
        raise ValueError(f"Unknown --dllm_v1_generate_mode: {args.dllm_v1_generate_mode}")

    device = dllm_v1.device
    new_tokens = torch.tensor(token_ids_so_far, device=device, dtype=torch.long).unsqueeze(0)
    prompt = torch.cat([orig_model_inputs["input_ids"].to(device), new_tokens], dim=1)
    prompt, num_sanitized = _sanitize_tokens_for_dllm_v1(
        prompt,
        getattr(dllm_v1.config, "vocab_size", None),
        pad_token_id=126081,
    )
    if num_sanitized > 0:
        logging.warning(
            f"[dllm_v1] Replaced {num_sanitized} out-of-range token ids with pad token id 126081."
        )
    prompt_len = prompt.shape[1]

    generated_ids, num_forward_passes = generator_fn(
        dllm_v1,
        prompt,
        gen_length=args.block_size, # total number of tokens to generate this call (the drafted length budget).
        block_length=small_block_size, # size of each semi-AR block.
        temperature=0.0,
        remasking="low_confidence",
        threshold=threshold,
    )

    generated_ids = generated_ids[0][prompt_len:prompt_len + spec_len].tolist()
    forward_pass_latencies = []
    return generated_ids, num_forward_passes, forward_pass_latencies


# ---------------------------------------------------------------------------
# DiffuLLaMA drafter (diffusionfamily/diffullama)
# ---------------------------------------------------------------------------

_DIFFULLAMA_MODULES = None


def _load_diffullama_modules(args):
    """Lazily import DiffuLLaMA attention_patch + model modules.

    The attention patch monkey-patches LlamaModel.forward to accept 4D masks.
    It is backward-compatible: for 2D masks (used by the target model) the
    original causal behaviour is preserved.
    """
    global _DIFFULLAMA_MODULES
    if _DIFFULLAMA_MODULES is not None:
        return _DIFFULLAMA_MODULES

    diffullama_dir = args.diffullama_dir
    model_py = os.path.join(diffullama_dir, "model.py")
    attention_patch_py = os.path.join(diffullama_dir, "attention_patch.py")
    for path in (model_py, attention_patch_py):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find DiffuLLaMA file at: {path}. "
                "Please set --diffullama_dir to the DiffuLLaMA repo root."
            )

    if diffullama_dir not in sys.path:
        sys.path.insert(0, diffullama_dir)

    # Load the attention patch module
    attn_spec = importlib.util.spec_from_file_location("diffullama_attention_patch", attention_patch_py)
    attn_module = importlib.util.module_from_spec(attn_spec)
    attn_spec.loader.exec_module(attn_module)

    # Save originals BEFORE patching so we can restore them after DiffuLLaMA is built.
    orig_llama_model_fwd = transformers.models.llama.modeling_llama.LlamaModel.forward
    orig_decoder_fwd = transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward

    # Replace attention_patch.replace_attention_mask with a version that tolerates
    # transformers ≥4.46 where LlamaFlashAttention2 no longer exists.
    # Must happen before model.py is imported, since model.py calls it at load time.
    # NOTE: patch is applied class-level temporarily (for model.py import); after
    # DiffuLLaMA is constructed the caller restores originals and re-binds instance-level.
    def _safe_replace_attention_mask():
        transformers.models.llama.modeling_llama.LlamaModel.forward = attn_module.forward_llama2
        if hasattr(transformers.models.llama.modeling_llama, "LlamaFlashAttention2"):
            transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = attn_module.forward_llama2fa2
        transformers.models.gpt2.modeling_gpt2.GPT2Model.forward = attn_module.forward_gpt2
        # NOTE: Do NOT wrap LlamaDecoderLayer at class level here — done per-instance
        # after DiffuLLaMA construction to avoid breaking the target model.

    attn_module.replace_attention_mask = _safe_replace_attention_mask
    _safe_replace_attention_mask()
    # Register under the canonical name so model.py's `from attention_patch import ...`
    # picks up our patched module instead of the original.
    sys.modules["attention_patch"] = attn_module

    # Load model module
    model_spec = importlib.util.spec_from_file_location("diffullama_model", model_py)
    model_module = importlib.util.module_from_spec(model_spec)
    model_spec.loader.exec_module(model_module)

    _DIFFULLAMA_MODULES = (
        model_module.DiscreteDiffusionModel,
        model_module.generate_samples,
        attn_module,
        orig_llama_model_fwd,
        orig_decoder_fwd,
    )
    return _DIFFULLAMA_MODULES


def get_next_n_tokens_diffullama(diffullama, args, orig_model_inputs, token_ids_so_far, spec_len, diffusion_steps):
    """Draft `spec_len` tokens using DiffuLLaMA (static-frequency).

    DiffuLLaMA is a discrete diffusion LM that fills all masked positions
    simultaneously via iterative denoising.  The calling convention mirrors
    ``get_next_n_tokens_dllm_v1``:

    Returns:
        (draft_tokens: list[int], num_forward_passes: int, forward_pass_latencies: list)
    """
    _, generate_samples_fn, *_ = _load_diffullama_modules(args)
    diff_args = Namespace(
        shift=args.diffullama_args.shift,
        diffusion_steps=diffusion_steps,
        logits_temp=args.diffullama_args.logits_temp,
        topp_temp=args.diffullama_args.topp_temp,
    )
    tokenizer = args.diffullama_tokenizer

    # Build full prefix: original prompt tokens + accepted tokens so far
    prefix_ids = orig_model_inputs["input_ids"][0].tolist() + token_ids_so_far
    gen_len = spec_len

    x0 = prefix_ids + [0] * gen_len
    src_mask = [1] * len(prefix_ids) + [0] * gen_len

    inputs = {
        "input_ids": torch.tensor([x0], dtype=torch.long),
        "src_mask": torch.tensor([src_mask], dtype=torch.long),
    }

    # Suppress the "*** Start sampling..." print inside generate_samples
    buf = io.StringIO()
    with torch.no_grad(), redirect_stdout(buf):
        result = generate_samples_fn(diffullama, diff_args, tokenizer, inputs)

    # After shift removal the output has shape (1, len(prefix_ids) + gen_len - 1).
    # The generated tokens occupy indices [len(prefix_ids)-1 : len(prefix_ids)-1+gen_len].
    start = len(prefix_ids) - 1
    draft_tokens = result[0][start : start + gen_len].tolist()

    # Warn if any token exceeds the target vocab (should not happen in practice)
    target_vocab_size = args.target_tokenizer.vocab_size
    for tok in draft_tokens:
        if tok >= target_vocab_size:
            logging.warning(
                f"[diffullama] Out-of-range token {tok} (target vocab_size={target_vocab_size})"
            )

    # Each generate_samples call performs diffusion_steps forward passes
    num_forward_passes = diff_args.diffusion_steps
    return draft_tokens, num_forward_passes, []


def construct_drafter_configs(args):
    drafter_configs = []
    if args.run_ar:  # AR Drafter
        drafter_configs.extend([("ar", None, "sf", None, None, None)])
        if args.ar_dynamic:  # AR Drafter (dynamic frequency)
            drafter_configs.extend([
                ("ar", None, "df", lowconf_threshold, max_spec_len, incr_len)
                for lowconf_threshold in args.sweep_lowconf_threshold
                for max_spec_len in args.sweep_max_spec_len
                for incr_len in args.sweep_incr_len
            ])
    if args.run_dllm_sf:  # Fast-dLLM Drafter
        drafter_configs.extend([("dllm", thr, "sf", None, None, None) for thr in args.drafter_thresholds])
    if args.run_dllm_v1_sf:  # Fast-dLLM-v1 Drafter (LLaDA backend)
        drafter_configs.extend([("dllm_v1", thr, "sf", None, None, None) for thr in args.drafter_thresholds])
    if args.run_diffullama_sf:  # DiffuLLaMA Drafter (diffusionfamily/diffullama)
        drafter_configs.extend([("diffullama", steps, "sf", None, None, None) for steps in args.diffullama_diffusion_steps])
    if not args.baseline_sweep:  # FailFast Drafter
        drafter_configs.extend([("dllm", thr, "df", lowconf_threshold, max_spec_len, incr_len) 
                                for thr in args.drafter_thresholds
                                for lowconf_threshold in args.sweep_lowconf_threshold
                                for max_spec_len in args.sweep_max_spec_len
                                for incr_len in args.sweep_incr_len
                                ])
    args.drafter_configs = drafter_configs


# %%
parser = argparse.ArgumentParser(description="Profiles the acceptance rate of speculative decoding within a single query.")
parser.add_argument("--dataset_name", type=str, choices=["aime", "math", "gsm8k", "gpqa", "humaneval"], default="math",
                    help="Dataset")
parser.add_argument("--output_dir", type=str, default="/data2/ruipan/diffspec", 
                    help="Where result pickle files (and output figures) will be written to")
parser.add_argument("--target_model_name", type=str, default="Qwen/Qwen2.5-32B-Instruct", 
                    help="Name of the base model to use")
parser.add_argument("--dllm_dir", type=str, default="/data2/ruipan/Fast_dLLM_v2_1.5B", 
                    help="Dir to the dLLM weights and (modified) modeling.py")
parser.add_argument("--dllm_v1_dir", type=str, default="/scratch/gpfs/RAVIAN/zhuofuc/Fast-dLLM",
                    help="Path to Fast-dLLM repo root (expects llada/generate.py inside)")
parser.add_argument("--dllm_v1_generate_mode", type=str, default="generate", choices=["generate", "prefix_cache", "dual_cache"],
                    help="Fast-dLLM-v1 generation backend")
parser.add_argument("--num_questions", type=int, default=1,
                    help="Number of questions to run profiling on")
parser.add_argument("--max_new_tokens", type=int, default=1024,
                    help="Max new tokens from the target model")
parser.add_argument("--block_size", type=int, default=32,
                    help="Block size in Fast-dLLM")
parser.add_argument("--small_block_size", type=int, default=8,
                    help="Small block size in Fast-dLLM")
parser.add_argument("--spec_len", type=int, default=10,
                    help="Frequency of verification steps (in number of tokens)")
parser.add_argument("--drafter_thresholds", type=float, nargs="+",  # one or more float thresholds
                    default=[0.05],
                    help="Threshold for confidence-adaptive decoding of the dLLM drafter model (e.g., --drafter_thresholds 0.1 0.5 0.9 runs a sweep of the three)")
parser.add_argument("--log_level",
                    type=str,
                    default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Set the logging level")
parser.add_argument("--sweep_lowconf_threshold", type=float, nargs="+",
                    default=[0.45],
                    help="τ in FailFast Alg. 1")
parser.add_argument("--sweep_max_spec_len", type=int, nargs="+",
                    default=[60],
                    help="N_max in FailFast Alg. 1")
parser.add_argument("--sweep_incr_len", type=int, nargs="+",
                    default=[10],
                    help="N in FailFast Alg. 1")
parser.add_argument('--run_ar', action='store_true', help='Run the AR drafter to compare speedups')
parser.add_argument('--ar_dynamic', action='store_true', help='Also run the AR drafter in dynamic mode (FailFast-style stop condition)')
parser.add_argument('--run_dllm_sf', action='store_true', help='Run the dLLM drafter with static frequency (in param sweep for baselines)')
parser.add_argument('--run_dllm_v1_sf', action='store_true', help='Run Fast-dLLM-v1 (LLaDA) drafter with static frequency')
parser.add_argument('--run_diffullama_sf', action='store_true', help='Run DiffuLLaMA (diffusionfamily/diffullama) drafter with static frequency')
parser.add_argument('--diffullama_dir', type=str, default="/scratch/gpfs/RAVIAN/zhuofuc/DiffuLLaMA",
                    help="Path to DiffuLLaMA repo root (expects model.py, attention_patch.py inside)")
parser.add_argument('--diffullama_diffusion_steps', type=int, nargs="+", default=[64],
                    help="Diffusion denoising steps per draft call; multiple values run a sweep (e.g., --diffullama_diffusion_steps 10 32 64)")
parser.add_argument('--baseline_sweep', action='store_true', help='Running a baseline sweep, don\'t run dynamic frequency')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output pickles and figures')
parser.add_argument('--reuse_drafts', action='store_true', help='Reuses drafted tokens from previous rounds if possible -- see Appendix E.')
parser.add_argument('--disable_reusing_drafter_kvs', action='store_true', help='Disables reusing drafter KV cache across verification rounds')
parser.add_argument('--read_pickle', action='store_true', help='Use acceptance decisions from a cached pickle file rather than rerunning')
args, _ = parser.parse_known_args()



######custom fields for easier debugging########
# args.log_level = "DEBUG"
# args.disable_reusing_drafter_kvs = True
# args.dataset_name = "gpqa"
# args.overwrite = True
# args.max_new_tokens = 1024
# args.run_ar = True
# args.ar_dynamic = True
# args.baseline_sweep = True
# args.spec_len = 16
# args.block_size = 32
# args.small_block_size = 16
# args.run_dllm_sf = True
# args.read_pickle = True  # XXX: read trajectory from pickle as well in future debugging
# args.target_model_name = "Qwen/Qwen2.5-7B-Instruct"  # for easier debugging
# args.sweep_lowconf_threshold = [0.4]
# args.sweep_max_spec_len = [96]
# args.sweep_incr_len = [16]
######custom fields for easier debugging######

args.target_model_name_clean = args.target_model_name.split("/", 1)[1]
logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%m%d %H:%M:%S",
    # datefmt="%m%d",
)

construct_drafter_configs(args)  # populates args.drafter_configs
populate_dataset(args)  # populates args.dataset

args.latency = {  # all in ms
    "vLLM_A6000": {
        "draft_fwd_pass": 6.1,
        "target_tpt": {
            "Qwen2.5-7B-Instruct": 13.5,
            "Qwen2.5-14B-Instruct": 24.7,
            "Qwen2.5-32B-Instruct": 52.6,
            # TODO: get the actual TPT for following models
            "Meta-Llama-3-70B-Instruct": 1000,
            "Llama-2-13b-chat-hf": 1000,
            "Llama-2-70b-chat-hf": 1000,
        },
    },
}

target_model_path = _resolve_local_snapshot_path(args.target_model_name)
target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)
args.target_tokenizer = target_tokenizer

# %%
if not args.read_pickle:
    logging.info(f"{Colors.BOLD}=== Loading target model: {args.target_model_name} ==={Colors.RESET}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    dllm = None
    dllm_v1 = None
    diffullama = None

    need_dllm = any(x[0] == "dllm" for x in args.drafter_configs)
    need_dllm_v1 = any(x[0] == "dllm_v1" for x in args.drafter_configs)
    need_diffullama = any(x[0] == "diffullama" for x in args.drafter_configs)

    if need_dllm:
        dllm_name = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
        dllm = AutoModelForCausalLM.from_pretrained(
            args.dllm_dir if args.dllm_dir is not None else dllm_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
    if need_dllm_v1:
        _, LLaDAModelLM = _load_fast_dllm_v1_modules(args)
        dllm_v1_name = "GSAI-ML/LLaDA-8B-Instruct"
        dllm_v1_path = _resolve_local_snapshot_path(dllm_v1_name)
        dllm_v1_device = "cuda" if torch.cuda.is_available() else "cpu"
        dllm_v1 = LLaDAModelLM.from_pretrained(
            dllm_v1_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(dllm_v1_device).eval()
    # NOTE(ruipan): drafter and target should probably share the same tokenizer?
    # dllm_tokenizer = AutoTokenizer.from_pretrained(dllm_name, trust_remote_code=True)
    dllm_tokenizer = target_tokenizer
    # dllm_v1_tokenizer = AutoTokenizer.from_pretrained(dllm_v1_path, trust_remote_code=True)
    dllm_v1_tokenizer = target_tokenizer
    if need_diffullama:
        # Load DiffuLLaMA modules (applies attention patch as a side-effect)
        DiscreteDiffusionModel, _, _diffullama_attn_module, _orig_llama_model_fwd, _orig_decoder_fwd = _load_diffullama_modules(args)
        from transformers import AutoConfig, LlamaForCausalLM
        diffullama_model_name = "diffusionfamily/diffullama"
        diffullama_path = _resolve_local_snapshot_path(diffullama_model_name)
        logging.info(f"{Colors.BOLD}=== Loading DiffuLLaMA drafter: {diffullama_model_name} ==={Colors.RESET}")
        diffullama_config = AutoConfig.from_pretrained(diffullama_path)
        diffullama_tokenizer = AutoTokenizer.from_pretrained(diffullama_path)
        diffullama_base = LlamaForCausalLM.from_pretrained(
            diffullama_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager",
            # Do NOT use device_map — accelerate would split layers across GPUs and
            # generate_samples's single-device torch.cat calls would break.
            # Load to CPU, then move the whole model to GPU 0 below.
        )
        diffullama = DiscreteDiffusionModel(
            model=diffullama_base,
            config=diffullama_config,
            tokenizer=diffullama_tokenizer,
            device="cuda:0",
        ).to("cuda:0")
        diffullama.eval()
        # Disable KV cache — DiffuLLaMA uses bidirectional attention and never needs it.
        diffullama.denoise_model.config.use_cache = False

        # --- Instance-level patching (transformers ≥4.57 compatibility) ---
        # The class-level forward_llama2 patch was needed for model.py import.
        # Now restore the original class-level LlamaModel.forward so the target
        # model uses the standard transformers forward (no _update_causal_mask call).
        import types as _types
        transformers.models.llama.modeling_llama.LlamaModel.forward = _orig_llama_model_fwd
        # Bind forward_llama2 to diffullama.denoise_model instance only.
        diffullama.denoise_model.forward = _types.MethodType(
            _diffullama_attn_module.forward_llama2, diffullama.denoise_model
        )
        # In transformers ≥4.57, LlamaDecoderLayer.forward returns a plain Tensor.
        # Wrap each of DiffuLLaMA's decoder layers individually so layer_outputs[0]
        # gives full 3D hidden_states without stripping the batch dimension.
        def _make_tuple_fwd(orig_fwd):
            def _wrapped(self, *args, **kwargs):
                out = orig_fwd(self, *args, **kwargs)
                return out if isinstance(out, (tuple, list)) else (out,)
            return _wrapped
        _tuple_dec_fwd = _make_tuple_fwd(_orig_decoder_fwd)
        for _layer in diffullama.denoise_model.layers:
            _layer.forward = _types.MethodType(_tuple_dec_fwd, _layer)
        # Store in args for access inside get_next_n_tokens_diffullama
        args.diffullama_tokenizer = diffullama_tokenizer
        # logits_temp=1.0 / topp_temp=1.0: no temperature scaling → deterministic greedy drafts
        args.diffullama_args = Namespace(
            shift=True,
            diffusion_steps=None,  # set per-call from drafter_threshold in the main loop
            logits_temp=1.0,
            topp_temp=1.0,
        )
        logging.info(
            f"DiffuLLaMA loaded. diffusion_steps sweep={args.diffullama_diffusion_steps}, "
            f"mask_token_id={diffullama_tokenizer.mask_token_id}"
        )
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
# for problem_id in [2]:
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
                    if freq_scheme == "sf":
                        draft_proposal = get_next_n_tokens_ar(draft_model, orig_model_inputs, current_token_ids, n=args.spec_len)
                        spec_len = args.spec_len
                    else:
                        draft_proposal, confidences = get_next_tokens_ar(
                            draft_model,
                            orig_model_inputs,
                            current_token_ids,
                            n=args.spec_len,
                            lowconf_threshold=lowconf_threshold,
                            max_spec_len=max_spec_len,
                            incr_len=incr_len,
                        )
                        spec_len = len(draft_proposal)
                        logging.debug(f"[Round {num_speculation_rounds}] AR drafter proposed {spec_len} tokens: {draft_proposal}")
                        logging.debug(f"[Round {num_speculation_rounds}] AR drafter confidences: {[round(r, 2) for r in confidences]}")
                    num_forward_passes = spec_len  # 1 fwd pass per token for AR drafter
                elif draft_type == "dllm":
                    if freq_scheme == "sf":  # static frequency
                        spec_len = args.spec_len
                        if args.disable_reusing_drafter_kvs:
                            draft_proposal, num_forward_passes, forward_pass_latencies = get_next_n_tokens_dllm(dllm, args, orig_model_inputs, current_token_ids, 
                                                                    spec_len=spec_len,  # number of speculative tokens proposed each time
                                                                    output_seqlen=3*args.block_size,  # 2 blocks of 32. Ensures spec_len tokens are generated in case they span over two blocks
                                                                    small_block_size=args.small_block_size,
                                                                    threshold=drafter_threshold,
                                                                    is_drafter=True,)
                        else:
                            draft_proposal, prefill_output, num_forward_passes, forward_pass_latencies = get_next_n_tokens_dllm(dllm, args, orig_model_inputs, current_token_ids, 
                                                                    spec_len=spec_len,  # number of speculative tokens proposed each time
                                                                    output_seqlen=3*args.block_size,  # 2 blocks of 32. Ensures spec_len tokens are generated in case they span over two blocks
                                                                    small_block_size=args.small_block_size,
                                                                    threshold=drafter_threshold,
                                                                    is_drafter=True,
                                                                    prev_prefill_output=prev_prefill_output)
                            prev_prefill_output = prefill_output
                    else:  # dynamic frequency: drafter determines how many tokens to propose
                        ###start of logic of reusing rejected drafts from the last round###
                        last_round_proposal = pickled_data["stats_each_round"][-1]["~draft_proposal"] if num_speculation_rounds > 0 else []
                        last_round_accepted_len = pickled_data["stats_each_round"][-1]["accepted_len"] if num_speculation_rounds > 0 else 0
                        if last_round_accepted_len < len(last_round_proposal) - 1:  # there are salvagable rejected tokens in the last round
                            if args.reuse_drafts:
                                last_round_rejected = last_round_proposal[last_round_accepted_len+1:] if num_speculation_rounds > 0 else []
                            else:
                                last_round_rejected = None
                        else:
                            last_round_rejected = None
                        ###end of logic of reusing rejected drafts from the last round###
                        
                        draft_proposal, actual_spec_len, prefill_output, num_forward_passes, forward_pass_latencies = get_next_tokens_dllm(dllm, args, orig_model_inputs, current_token_ids, 
                                                                    spec_len=args.spec_len,  # number of speculative tokens proposed each time
                                                                    output_seqlen=3*args.block_size,  # 3 blocks of 32. Ensures spec_len tokens are generated in case they span over two blocks
                                                                    small_block_size=args.small_block_size,
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
                elif draft_type == "dllm_v1":
                    if freq_scheme != "sf":
                        raise NotImplementedError("Fast-dLLM-v1 currently supports static-frequency drafting only.")
                    spec_len = args.spec_len
                    draft_proposal, num_forward_passes, forward_pass_latencies = get_next_n_tokens_dllm_v1(
                        dllm_v1,
                        args,
                        orig_model_inputs,
                        current_token_ids,
                        spec_len=spec_len,
                        small_block_size=args.small_block_size,
                        threshold=drafter_threshold,
                    )
                elif draft_type == "diffullama":
                    if freq_scheme != "sf":
                        raise NotImplementedError("DiffuLLaMA currently supports static-frequency drafting only.")
                    spec_len = args.spec_len
                    draft_proposal, num_forward_passes, forward_pass_latencies = get_next_n_tokens_diffullama(
                        diffullama,
                        args,
                        orig_model_inputs,
                        current_token_ids,
                        spec_len=spec_len,
                        diffusion_steps=drafter_threshold,
                    )

                total_num_forward_passes += num_forward_passes
                # print(f"forward_pass_latencies {forward_pass_latencies}")  # NOTE(ruipan): TPT of 1.5B dLLM is similar to 1.5B AR model
                
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
                
                # # add stopping condition if drafter proposes EOS token?
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
        
        pickle_path = os.path.join(output_dir_pickles, f"{args.max_new_tokens}.pickle")
        if (args.overwrite and not args.read_pickle) or (not os.path.exists(pickle_path)):
            with open(pickle_path, "wb") as f:
                pickle.dump(pickled_data, f)
                logging.info(f"Saved pickled data to {pickle_path}")
            with open(os.path.join(output_dir_pickles, f"{args.max_new_tokens}.txt"), "w") as f:
                pp = pprint.PrettyPrinter(width=1000, stream=f)  # large enough to fit list
                pp.pprint(pickled_data)
        else:
            logging.info(f"Skipping save for pickled data to {pickle_path}")

# %%
