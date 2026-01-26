# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pickle

# Reused from difficulty_intuition.py
target_model = "Qwen2.5-32B-Instruct"
datasets = ["math", "aime", "gsm8k", "gpqa", "humaneval"]
DATASET_DISPLAY_NAMES = {
    "math": "MATH",
    "aime": "AIME",
    "gsm8k": "GSM8K",
    "gpqa": "GPQA",
    "humaneval": "HumanEval",
}
drafter_name = "ar_None_sf_8"
question_ids = list(range(30))  # 0 to 29, all thirty questions
pickles_base = "/data2/ruipan/diffspec/pickles"


def get_boolean_decision_from_stats_each_round(stats_each_round):
    """Reused from difficulty_intuition.py: converts round stats to a boolean array."""
    decisions = []
    for round_id, round_info in enumerate(stats_each_round):
        accepted_len = round_info["accepted_len"]
        spec_len = len(round_info["~draft_proposal"])

        if accepted_len == spec_len:
            decisions.extend([True] * (spec_len + 1))  # all accepted plus one bonus token
        else:
            decisions.extend([True] * accepted_len)
            decisions.append(False)  # first rejected token
    return decisions


def binary_to_consecutive_true_lengths(binary_list):
    """
    Convert a binary list to lengths of consecutive True's.
    E.g., [True, True, False, True, False, True, True, True] -> [2, 1, 3].
    """
    lengths = []
    count = 0
    for b in binary_list:
        if b:
            count += 1
        else:
            if count > 0:
                lengths.append(count)
            count = 0
    if count > 0:
        lengths.append(count)
    return lengths


def values_to_cdf(values):
    """
    Reused from acc_pro_len_cdf.py (lines 79–89).
    Sorts the values in-place and returns the Y-axis probabilities.
    """
    cdf_list = []
    values.sort()  # sorts in place
    count = 0
    for v in values:
        count += 1
        cdf_list.append(count / len(values))
    return cdf_list


# Load data and compute consecutive-easy lengths per dataset
dataset_consecutive_lengths = {}
dataset_all_num_tokens = {}
for dataset in datasets:
    all_lengths = []
    total_num_tokens = 0
    for qid in question_ids:
        stats_path = f"{pickles_base}/{target_model}/{dataset}/{qid}/{drafter_name}/1024.pickle"
        if not os.path.exists(stats_path):
            continue
        with open(stats_path, "rb") as f:
            pickle_file = pickle.load(f)
        stats_each_round = pickle_file["stats_each_round"]
        decisions = get_boolean_decision_from_stats_each_round(stats_each_round)
        lengths = binary_to_consecutive_true_lengths(decisions)
        all_lengths.extend(lengths)
        total_num_tokens += len(decisions)
    dataset_consecutive_lengths[dataset] = all_lengths
    dataset_all_num_tokens[dataset] = total_num_tokens

# %%
cutoffs = [50]
for cutoff in cutoffs:
    for dataset in datasets:
        num_tokens_in_easy_blks = sum([x for x in dataset_consecutive_lengths[dataset] \
            if x > cutoff])
        ratio = num_tokens_in_easy_blks / dataset_all_num_tokens[dataset]
        print(f"Cutoff {cutoff}, dataset {dataset}, ratio {ratio}")




# # # %%
# # Plot: 5 subplots in one row, one per dataset, shared y-axis (0 to 1)
# fig, axes = plt.subplots(1, 5, figsize=(12, 2.5), sharey=True)
# axes = axes.flatten()

# for i, dataset in enumerate(datasets):
#     ax = axes[i]
#     lengths = dataset_consecutive_lengths[dataset]
#     y_vals = values_to_cdf(lengths)  # sorts lengths in place; returns CDF y-values
#     ax.plot(lengths, y_vals, color="#2D6A4F", linewidth=2)
#     ax.set_title(DATASET_DISPLAY_NAMES[dataset], fontsize=12)
#     ax.set_ylim(0, 1)
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
#     ax.grid(color="lightgrey", linestyle="dashed", linewidth=0.6)
#     ax.set_axisbelow(True)

# # Single x-label for the figure; shared y-label on the left
# fig.supxlabel("Consecutive Easy Length", fontsize=12)
# fig.supylabel("CDF", fontsize=12, x=-0.02)
# plt.tight_layout()
# plt.show()

# # Optional: save
# fig.savefig("../figures/consecutive_easy_cdf.pdf", dpi=300, bbox_inches="tight")
# %%
# %%
# Plot: 5 subplots in one row, shared y-axis (0 to 1)
# X-axis: Number of tokens in an easy block (Cutoff)
# Y-axis: Ratio of tokens that belong to an easy block > cutoff
fig, axes = plt.subplots(1, 5, figsize=(12, 2.5), sharey=True)
axes = axes.flatten()

for i, dataset in enumerate(datasets):
    ax = axes[i]
    lengths = dataset_consecutive_lengths[dataset]
    total_tokens = dataset_all_num_tokens[dataset]
    
    # Determine the range for x-axis based on the longest block in this dataset
    max_len = max(lengths) if lengths else 0
    # Create x values from 0 up to max_len (inclusive)
    x_vals = range(max_len + 2)
    
    y_vals = []
    for cutoff in x_vals:
        # Logic derived from your provided snippet:
        # Sum of lengths for blocks strictly longer than the cutoff
        num_tokens_in_easy_blks = sum([x for x in lengths if x > cutoff])
        ratio = num_tokens_in_easy_blks / total_tokens
        y_vals.append(ratio)

    ax.plot(x_vals, y_vals, color="#2D6A4F", linewidth=2)
    ax.set_title(DATASET_DISPLAY_NAMES[dataset], fontsize=12)
    ax.set_ylim(0, 1)
    # Optional: Set x-limit if tails are very long, or let matplotlib autoscal
    # ax.set_xlim(0, 50) 
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
    ax.grid(color="lightgrey", linestyle="dashed", linewidth=0.6)
    ax.set_axisbelow(True)

# Single x-label and shared y-label
fig.supxlabel("Easy Block Length", fontsize=12)
fig.supylabel("Ratio of Tokens that\nBelong to an Easy Block", fontsize=12, x=-0.02)
plt.tight_layout()
plt.show()
# %%
# %%
# Define a dictionary mapping dataset names to colors
dataset_colors = {
    datasets[0]: "#2D6A4F",  # Dark Blue-Green
    datasets[1]: "#40916C",  # Teal
    datasets[2]: "#52B788",  # Yellow-Ochre
    datasets[3]: "#74C69D",  # Sandy Orange
    datasets[4]: "#95D5B2"   # Burnt Sienna
}

# Define linestyles to differentiate further
linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

fig, ax = plt.subplots(figsize=(9, 4.7))

for i, dataset in enumerate(datasets):
    lengths = dataset_consecutive_lengths[dataset]
    total_tokens = dataset_all_num_tokens[dataset]
    
    # Calculate X and Y values
    max_len = max(lengths) if lengths else 0
    x_vals = list(range(max_len + 2))
    y_vals = []
    
    for cutoff in x_vals:
        # Logic: ratio of tokens belonging to blocks strictly longer than the cutoff
        num_tokens_in_easy_blks = sum([x for x in lengths if x > cutoff])
        ratio = num_tokens_in_easy_blks / total_tokens
        y_vals.append(ratio)

    # Plot
    ax.plot(
        x_vals, 
        y_vals, 
        color=dataset_colors[dataset], 
        linestyle=linestyles[i % len(linestyles)], 
        linewidth=2.5, 
        label=DATASET_DISPLAY_NAMES[dataset]
    )

# Formatting
ax.set_ylim(0, 1)
ticks = list(range(0, 210, 10))
ax.set_xticks(ticks)
# Set labels: only show string if divisible by 20, else empty string
ax.set_xticklabels([str(t) if t % 20 == 0 else "" for t in ticks])
ax.set_xlim(0, 200)
# ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(color="lightgrey", linestyle="dashed", linewidth=1.5)
ax.set_axisbelow(True)

# ax.set_xlabel("Length of Blocks of Consecutive Easy Tokens", fontsize=20)
ax.set_xlabel("Easy Region Length", fontsize=20)
ax.set_ylabel("Ratio of Tokens in Easy Regions", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(loc="upper right", fontsize=20)
ax.yaxis.set_label_coords(-0.08, 0.43)  # x, y

plt.tight_layout()
plt.show()
fig.savefig("../figures/consecutive_easy_ratio.pdf", dpi=300, bbox_inches="tight")

# %%
