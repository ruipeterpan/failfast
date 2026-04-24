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
# 1.5 Calibration Plot (P(easy | confidence))
# ==========================================

# Combine data into (confidence, label)
# label = 1 if easy (accepted), 0 if hard (rejected)
all_data = []
all_data.extend([(x[1], 1) for x in correct_conf_list])
all_data.extend([(x[1], 0) for x in incorrect_conf_list])

# Sort by confidence (not strictly necessary, but clean)
all_data.sort(key=lambda x: x[0])

# Binning
num_bins = 10
bins = np.linspace(0.0, 1.0, num_bins + 1)

bin_centers = []
bin_acc = []
bin_counts = []

for i in range(num_bins):
    left, right = bins[i], bins[i+1]
    
    # Include right edge in last bin
    if i == num_bins - 1:
        bin_data = [label for conf, label in all_data if left <= conf <= right]
    else:
        bin_data = [label for conf, label in all_data if left <= conf < right]
    
    if len(bin_data) > 0:
        acc = np.mean(bin_data)  # P(easy | confidence in bin)
        center = (left + right) / 2
        
        bin_centers.append(center)
        bin_acc.append(acc)
        bin_counts.append(len(bin_data))

# Plot
fig, ax = plt.subplots(figsize=(5, 3))

ax.plot(bin_centers, bin_acc, marker='o', label="Empirical P(easy | confidence)")
ax.plot([0, 1], [0, 1], linestyle='--', label="Perfect calibration")

ax.set_xlabel("Drafter Token Confidence")
ax.set_ylabel("P(Token Accepted)")
# ax.set_title("Calibration Plot")
ax.legend()

plt.tight_layout()
plt.show()

# Optional: print bin stats (useful for rebuttal text)
print("Bin centers:", bin_centers)
print("P(easy | bin):", bin_acc)
print("Counts per bin:", bin_counts)



# %%
# NeurIPS-style binned plot
# --- Configuration ---
NUM_BINS = 20
COLOR_ACCEPT = '#5cb85c'  # Green
COLOR_REJECT = '#d9534f'  # Red
COLOR_LINE = '#428bca'    # Blue
# ---------------------

# 1. Extract confidences from the parsed data
accept_confs = [x[1] for x in correct_conf_list]
reject_confs = [x[1] for x in incorrect_conf_list]

# 2. Define bins based on the 0 to 1 confidence range
bins = np.linspace(0.0, 1.0, NUM_BINS + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2
bin_width = (1.0 / NUM_BINS) * 0.75  # 0.75 gives a nice gap between bars

# 3. Calculate counts per bin
accept_counts, _ = np.histogram(accept_confs, bins=bins)
reject_counts, _ = np.histogram(reject_confs, bins=bins)
total_counts = accept_counts + reject_counts

# 4. Calculate acceptance rate (handle division by zero)
with np.errstate(divide='ignore', invalid='ignore'):
    # Use np.nan so the line skips empty bins instead of dropping to 0
    acceptance_rate = np.where(total_counts > 0, accept_counts / total_counts, np.nan)

# 5. Setup the plot with dual axes
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# 6. Plot the stacked bars on the left axis (ax1)
# Notice `bottom=accept_counts` on the reject bar to stack it on top
bar_accept = ax1.bar(bin_centers, accept_counts, width=bin_width, 
                     color=COLOR_ACCEPT, label='accept')
bar_reject = ax1.bar(bin_centers, reject_counts, width=bin_width, bottom=accept_counts, 
                     color=COLOR_REJECT, label='reject')

# 7. Plot the acceptance rate line on the right axis (ax2)
line_acc, = ax2.plot(bin_centers, acceptance_rate * 100, 
                     color=COLOR_LINE, linewidth=2.5, label='acceptance rate')

# 8. Formatting X-axis (place ticks at bin edges like the reference image)
ax1.set_xlabel("Confidence")
ax1.set_xticks(bins)
ax1.set_xticklabels([f"{x:.2f}" for x in bins], rotation=45, ha='right')
ax1.set_xlim(0 - (bin_width/2), 1 + (bin_width/2))

# 9. Formatting Y-axes
ax1.set_ylabel("Count")
ax2.set_ylabel("Acceptance Rate")
ax2.set_ylim(0, 105) # Add a tiny bit of padding above 100%
ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())

# 10. Aesthetics and Legend
# Add light horizontal grid lines
ax1.grid(axis='y', linestyle='-', alpha=0.5)

# Remove top spines
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Combine legends into a single box at the top, matching the image order
handles = [line_acc, bar_reject, bar_accept]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), 
           ncol=3, frameon=False, handletextpad=0.5)

plt.tight_layout()
plt.show()


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
# 1.7 Histogram Overlap (Density View)
# ==========================================

correct_conf = np.array([x[1] for x in correct_conf_list])
incorrect_conf = np.array([x[1] for x in incorrect_conf_list])

# Binning
num_bins = 30
bins = np.linspace(0.0, 1.0, num_bins + 1)

fig, ax = plt.subplots(figsize=(5, 3))

# Plot normalized histograms (density=True)
ax.hist(
    correct_conf,
    bins=bins,
    density=True,
    alpha=0.6,
    label="\"Easier\" Tokens",
    color='#2D6A4F'
)

ax.hist(
    incorrect_conf,
    bins=bins,
    density=True,
    alpha=0.6,
    label="\"Harder\" Tokens",
    color='#74C69D'
)

ax.set_xlabel("Drafter Token Confidence")
ax.set_ylabel("Density")
ax.set_title("Confidence Distribution Overlap")
ax.legend()

plt.tight_layout()
plt.show()

# Optional: quantify overlap (useful for rebuttal)
# Overlap = integral of min(densities)
hist_easy, _ = np.histogram(correct_conf, bins=bins, density=True)
hist_hard, _ = np.histogram(incorrect_conf, bins=bins, density=True)

bin_width = bins[1] - bins[0]
overlap = np.sum(np.minimum(hist_easy, hist_hard)) * bin_width

print(f"Distribution overlap (area): {overlap:.3f}")

# %%

# ==========================================
# 2. Plotting Configuration (Knobs)
# ==========================================

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

# Font settings for PDF export (from <script2>)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42



fig, ax = plt.subplots(figsize=(5, 3))


correct_conf = [x[1] for x in correct_conf_list]
incorrect_conf = [x[1] for x in incorrect_conf_list]
ax.plot(
    correct_conf,
    values_to_cdf(correct_conf),
    label="\"Easier\" Tokens",
    color='#2D6A4F',
)
ax.plot(
    incorrect_conf,
    values_to_cdf(incorrect_conf),
    label="\"Harder\" Tokens",
    color='#74C69D',
    linestyle="dashed"
)
ax.set_xlabel("Drafter Token Confidence")
ax.set_ylabel("CDF")
ax.legend()
plt.tight_layout()
plt.show()



# Report percentiles of correct_conf_list and incorrect_conf_list
if correct_conf_list:
    correct_percentiles = np.percentile([x[1] for x in correct_conf_list], percentiles)
    print(f"Percentiles for correct_conf_list ({percentiles}): {correct_percentiles}")
else:
    print("correct_conf_list is empty, no percentiles to report.")

if incorrect_conf_list:
    incorrect_percentiles = np.percentile([x[1] for x in incorrect_conf_list], percentiles)
    print(f"Percentiles for incorrect_conf_list ({percentiles}): {incorrect_percentiles}")
else:
    print("incorrect_conf_list is empty, no percentiles to report.")

# %%
fig, axs = plt.subplots(1, 5, figsize=(30, 4))

for dataset in datasets:
    correct_conf = [x[1] for x in correct_conf_list if x[0] == dataset]
    incorrect_conf = [x[1] for x in incorrect_conf_list if x[0] == dataset]
    ax = axs[datasets.index(dataset)]
    ax.plot(
        correct_conf,
        values_to_cdf(correct_conf),
        label="\"Easier\" Tokens",
        color='#2D6A4F',
    )
    ax.plot(
        incorrect_conf,
        values_to_cdf(incorrect_conf),
        label="\"Harder\" Tokens",
        color='#74C69D',
        linestyle="dashed"
    )
    ax.set_xlabel("Drafter Token Confidence")
    ax.set_ylabel("CDF")
    ax.set_title(f"Dataset: {dataset}")
    ax.legend()
plt.show()

fig.savefig(f"../figures/rebuttal/conf_cdf_by_dataset.pdf", dpi=500, bbox_inches='tight')

# Report percentiles of correct_conf_list and incorrect_conf_list
if correct_conf_list:
    correct_percentiles = np.percentile([x[1] for x in correct_conf_list], percentiles)
    print(f"Percentiles for correct_conf_list ({percentiles}): {correct_percentiles}")
else:
    print("correct_conf_list is empty, no percentiles to report.")

if incorrect_conf_list:
    incorrect_percentiles = np.percentile([x[1] for x in incorrect_conf_list], percentiles)
    print(f"Percentiles for incorrect_conf_list ({percentiles}): {incorrect_percentiles}")
else:
    print("incorrect_conf_list is empty, no percentiles to report.")
# %%
