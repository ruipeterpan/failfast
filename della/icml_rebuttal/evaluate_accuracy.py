# %%
import re
import os
from datasets import load_dataset, load_from_disk

# 1. Variable that specifies the path to the text file
file_path = "/data2/ruipan/failfast_icml_rebuttal/accuracy_check/32b/2025_12_22_21_16_gsm8k.ansi"

# 2. Variable that extracts which dataset is being evaluated
dataset_choices =["aime", "math", "gsm8k", "gpqa", "humaneval"]
# Finds the first matching dataset name in the filename, or defaults to "unknown"
dataset_name = next((d for d in dataset_choices if d in file_path), "unknown")

print(f"Detected dataset: {dataset_name}")


def populate_dataset(dataset_name):
    if dataset_name == "aime":
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
    elif dataset_name == "math":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    elif dataset_name == "gpqa":
        # if os.getenv("HF_HUB_OFFLINE", "0") == "1":
        #     dataset = load_from_disk("/scratch/gpfs/RAVIAN/rp2773/hf_cache/datasets/gpqa")
        # else:    
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
    elif dataset_name == "mmlu":
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")["validation"]
    elif dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")["test"]
    elif dataset_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval")["test"]
    else:
        raise NotImplementedError
    return dataset

dataset = populate_dataset(dataset_name)



def extract_boxed_content_from_str(s):
    start_token = r'\boxed{'
    start = s.find(start_token)
    if start == -1:
        return None

    i = start + len(start_token)
    brace_count = 1
    content = []

    while i < len(s):
        char = s[i]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1

        if brace_count == 0:
            break

        content.append(char)
        i += 1

    return ''.join(content) if brace_count == 0 else None

def extract_trajectories(filepath):
    # Dictionary to store the target data
    # Key: Problem ID (int), Value: Full output trajectory (str)
    trajectories_dict = {}

    # Regex to clean up terminal ANSI escape codes (colors, formatting)
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    # Regex to clean up the standard logger prefix (e.g., "[0327 00:52:07 INFO] ")
    log_prefix = re.compile(r'^\[\d{4} \d{2}:\d{2}:\d{2} INFO\]\s*')

    with open(filepath, 'r', encoding='utf-8') as file:
        current_problem_id = None
        is_target_drafter = False
        capture_output = False
        current_output =[]

        for line in file:
            # Remove ANSI color/formatting codes to safely match text
            clean_line = ansi_escape.sub('', line)

            # Look for the start of a problem block and identify the drafter
            # Example: ===[Problem 0] Running drafter: ar_None_sf_10 ===
            drafter_match = re.search(r'===\s*\[Problem\s+(\d+)\]\s+Running drafter:\s*(.*?)\s*===', clean_line)
            
            # Alternatively, catch it if logging lacks the '===' header but prints statistics:
            # Example: ---[Problem 0, ar_None_sf_10] Statistics ---
            stat_match = re.search(r'\[Problem\s+(\d+),\s*(ar_None_sf_10)\]', clean_line)

            if drafter_match:
                current_problem_id = int(drafter_match.group(1))
                drafter_name = drafter_match.group(2)
                
                # We only want the specific "ar_None_sf_10" drafter
                is_target_drafter = (drafter_name == "ar_None_sf_10")
                capture_output = False
                continue
            elif stat_match and not is_target_drafter:
                current_problem_id = int(stat_match.group(1))
                is_target_drafter = True

            # If we are currently inside the target drafter's block for a problem
            if is_target_drafter:
                # Start capturing when hitting "--- Output ---"
                if '--- Output ---' in clean_line:
                    capture_output = True
                    current_output =[]
                    continue
                
                # Stop capturing when hitting "--- Trajectory ---"
                elif '--- Trajectory ---' in clean_line:
                    capture_output = False
                    if current_problem_id is not None:
                        # Join captured lines and save to dictionary
                        # .strip() removes trailing/leading whitespace but maintains inner newlines
                        trajectories_dict[current_problem_id] = "".join(current_output).strip()
                    
                    # Reset state machine for the next block
                    is_target_drafter = False
                    current_problem_id = None
                    continue

                # Append lines directly into our buffer while in the "Output" section
                if capture_output:
                    # Strip out the logger timestamp at the start of the line (if present)
                    text = log_prefix.sub('', clean_line)
                    current_output.append(text)

    return trajectories_dict

# Execute the extraction
total_correct = 0
total_problems = 0
if os.path.exists(file_path):
    problem_trajectories = extract_trajectories(file_path)

    # Example verification output
    for problem_id, trajectory in problem_trajectories.items():
        total_problems += 1

        extracted_answer = extract_boxed_content_from_str(trajectory)
        if dataset_name in ["aime", "math"]:
            correct_answer = dataset[problem_id]["answer"]
        elif dataset_name == "gsm8k":
            correct_answer = dataset[problem_id]["answer"].split("\n####")[-1].strip()
        elif dataset_name == "gpqa":
            correct_answer = "A"  # dataset[problem_id]["Pre-Revision Correct Answer"]
        em = (extracted_answer == correct_answer)
        if em: 
            total_correct += 1
        print(f"Problem ID: {problem_id}. EM: {em}. Extracted Answer: {extracted_answer}. Correct Answer: {correct_answer}.")
        print("Trajectory Output:")
        print(trajectory)
else:
    print(f"File {file_path} not found. Please ensure the path is correct.")

print(f"Accuracy: {total_correct}/{total_problems} = {total_correct/total_problems:.2%}. Remember to manually check all answers!")

# %%
