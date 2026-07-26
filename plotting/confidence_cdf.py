# %%
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Data Loading (From <script1>)
# ==========================================

def read_pickle(data_dir, target_model, dataset, question_id, drafter_config):
    file_name = f"{target_model}/{dataset}/{question_id}/{drafter_config}/1024.pickle"
    file_path = os.path.join(data_dir, file_name)
    
    # Check if file exists to prevent crashing if data isn't present
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return None

    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

data_dir = "/home/ruipan/data2/failfast_icml_rebuttal/pickles"
target_model = "Qwen2.5-7B-Instruct"
datasets = ["math", "aime", "gsm8k", "gpqa", "humaneval"]
# datasets = ["math"]
num_questions = 30
# drafter_config = "ar_None_sf_10"
drafter_config = "dllm_0.05_df_0.45_60_10"
percentiles = [25, 50, 75]

correct_conf_list = []
incorrect_conf_list = []

print("Loading data...")
for dataset in datasets:
    for question_id in range(num_questions):
        ar_data = read_pickle(data_dir, target_model, dataset, question_id, drafter_config)
        
        if ar_data is not None:
            stats_each_round = ar_data["stats_each_round"]
            for r in range(len(stats_each_round)):
                stats = stats_each_round[r]
                conf = stats["confidences"]
                accepted_len = stats["accepted_len"]
                print(f"Question {question_id}, Round {r}: accepted_len={accepted_len}, conf={conf}")
                correct_conf_list.extend([(dataset, x) for x in conf[:accepted_len]])
                if accepted_len < len(conf):
                    incorrect_conf_list.append((dataset, conf[accepted_len]))


print(f"Data loaded. Points - correct_conf_list: {len(correct_conf_list)}, incorrect_conf_list: {len(incorrect_conf_list)}")



# %%
# ==========================================
# 1.5 Calibration Plot (P(easy | confidence)) — Improved Binning
# This is the one we are using for NeurIPS
# ==========================================

# Combine data
all_data = []
all_data.extend([(x[1], 1) for x in correct_conf_list])
all_data.extend([(x[1], 0) for x in incorrect_conf_list])

# Convert to arrays
conf_all = np.array([x[0] for x in all_data])
labels_all = np.array([x[1] for x in all_data])

# Sort by confidence
sorted_idx = np.argsort(conf_all)
conf_all = conf_all[sorted_idx]
labels_all = labels_all[sorted_idx]

# ---- Quantile bins ----
num_bins = 20  # try 30–50 depending on data size
quantiles = np.linspace(0, 1, num_bins + 1)

bin_edges = np.quantile(conf_all, quantiles)

bin_centers = []
bin_acc = []
bin_counts = []

for i in range(num_bins):
    left, right = bin_edges[i], bin_edges[i+1]
    
    # Include right edge in last bin
    if i == num_bins - 1:
        mask = (conf_all >= left) & (conf_all <= right)
    else:
        mask = (conf_all >= left) & (conf_all < right)
    
    bin_labels = labels_all[mask]
    bin_conf = conf_all[mask]
    
    if len(bin_labels) > 0:
        acc = np.mean(bin_labels)
        center = np.mean(bin_conf)
        
        bin_centers.append(center)
        bin_acc.append(acc)
        bin_counts.append(len(bin_labels))

# Plot
fig, ax = plt.subplots(figsize=(4.5, 2.4))

ax.plot([0, 1], [0, 1], linestyle='--', color='#74C69D', label="Perfect correlation")
ax.plot(bin_centers, bin_acc, marker='o', color='#2D6A4F', label="Empirical correlation", markersize=6)

ax.set_xlabel("Drafter Token Confidence", fontsize=12)
# ax.set_ylabel("P(Token Accepted)")
ax.set_ylabel("Acceptance Rate", fontsize=12)
# ax.set_title("Confidence is a Strong Signal for Acceptance\n(Quantile Binning)")
ax.legend(fontsize=10)

ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# Styling
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Optional diagnostics
print("Avg samples per bin:", np.mean(bin_counts))
print("Min samples in a bin:", np.min(bin_counts))

fig.savefig(f"../figures/conf_vs_acc_rate.pdf", dpi=500, bbox_inches='tight')

# %%
# ==========================================
# 1.6 Calibration Plot — Per-Dataset (Row of 5 Subplots)
# ==========================================

def compute_calibration_bins(correct_list, incorrect_list, num_bins=20):
    """Compute quantile-binned calibration curve for a given dataset's data points."""
    all_data = [(x, 1) for x in correct_list] + [(x, 0) for x in incorrect_list]
    if len(all_data) == 0:
        return [], [], []

    conf_arr = np.array([x[0] for x in all_data])
    labels_arr = np.array([x[1] for x in all_data])

    sorted_idx = np.argsort(conf_arr)
    conf_arr = conf_arr[sorted_idx]
    labels_arr = labels_arr[sorted_idx]

    quantiles = np.linspace(0, 1, num_bins + 1)
    bin_edges = np.quantile(conf_arr, quantiles)

    centers, accs, counts = [], [], []
    for i in range(num_bins):
        left, right = bin_edges[i], bin_edges[i + 1]
        mask = (conf_arr >= left) & (conf_arr <= right if i == num_bins - 1 else conf_arr < right)
        bin_labels = labels_arr[mask]
        bin_conf = conf_arr[mask]
        if len(bin_labels) > 0:
            centers.append(np.mean(bin_conf))
            accs.append(np.mean(bin_labels))
            counts.append(len(bin_labels))

    return centers, accs, counts


# Build per-dataset lookup from the already-loaded lists
dataset_correct = {ds: [] for ds in datasets}
dataset_incorrect = {ds: [] for ds in datasets}

for ds, conf in correct_conf_list:
    dataset_correct[ds].append(conf)
for ds, conf in incorrect_conf_list:
    dataset_incorrect[ds].append(conf)

# Plot
fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 2.8), sharey=True)

dataset_display_names = {
    "math": "MATH",
    "aime": "AIME",
    "gsm8k": "GSM8K",
    "gpqa": "GPQA",
    "humaneval": "HumanEval",
}

for ax, ds in zip(axes, datasets):
    centers, accs, counts = compute_calibration_bins(
        dataset_correct[ds], dataset_incorrect[ds], num_bins=20
    )

    ax.plot([0, 1], [0, 1], linestyle='--', color='#74C69D', label="Perfect correlation")
    if centers:
        ax.plot(centers, accs, marker='o', color='#2D6A4F', label="Empirical correlation", markersize=5)

    ax.set_title(dataset_display_names.get(ds, ds), fontsize=15)
    ax.set_xlabel("Drafter Token Confidence", fontsize=15)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

axes[0].set_ylabel("Acceptance Rate", fontsize=15)

# Single shared legend to the right of the last subplot
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=15,
           bbox_to_anchor=(0.5, 1.15), frameon=True)

plt.tight_layout()
plt.show()

fig.savefig(f"../figures/conf_vs_acc_rate_per_dataset.pdf", dpi=500, bbox_inches='tight')
# %%
