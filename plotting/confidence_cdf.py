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
