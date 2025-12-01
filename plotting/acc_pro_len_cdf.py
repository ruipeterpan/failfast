# %%
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt

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

data_dir = "/home/ruipan/data2/diffspec/pickles"
target_model = "Qwen2.5-32B-Instruct"
dataset = "math"
num_questions = 30
ar_baseline = "ar_None_sf_10"  # ar_None_sf_8 is the best one
failfast_config = "dllm_0.05_df_0.4_60_10"

ar_spec_len_list = []
ar_acc_len_list = []
ff_spec_len_list = []
ff_acc_len_list = []

print("Loading data...")
for question_id in range(num_questions):
    ar_data = read_pickle(data_dir, target_model, dataset, question_id, ar_baseline)
    ff_data = read_pickle(data_dir, target_model, dataset, question_id, failfast_config)
    
    if ar_data:
        ar_spec_len_list.extend([ar_data["stats_each_round"][r]["spec_len"] for r in range(len(ar_data["stats_each_round"]))])
        ar_acc_len_list.extend([ar_data["stats_each_round"][r]["accepted_len"] for r in range(len(ar_data["stats_each_round"]))])
    
    if ff_data:
        ff_spec_len_list.extend([ff_data["stats_each_round"][r]["spec_len"] for r in range(len(ff_data["stats_each_round"]))])
        ff_acc_len_list.extend([ff_data["stats_each_round"][r]["accepted_len"] for r in range(len(ff_data["stats_each_round"]))])

print(f"Data loaded. Points - AR: {len(ar_spec_len_list)}, FF: {len(ff_spec_len_list)}")

# ==========================================
# 2. Plotting Configuration (Knobs)
# ==========================================

# Font settings for PDF export (from <script2>)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# --- KNOBS FOR COLORS ---
# Set your two preferred colors here
color_ar = '#4285F4'  # Example: Dark Green for AR
color_ff = '#EA4335'  # Example: Red for FailFast

# --- KNOBS FOR LINESTYLES ---
linestyle_spec = 'solid'   # Style for spec_len
linestyle_acc = 'dotted'   # Style for acc_len
linewidth = 2

# ==========================================
# 3. Helper Functions (From <script2>)
# ==========================================

def values_to_cdf(values):
    """
    Sorts the values in-place and returns the Y-axis probabilities.
    """
    cdf_list = []
    values.sort() # Sorts in place
    count = 0
    for v in values:
        count += 1
        cdf_list.append(count / len(values))
    return cdf_list

# ==========================================
# 4. Plotting Execution
# ==========================================

fig, ax = plt.subplots(figsize=(6, 3))

# -- Prepare Data for AR --
# Note: values_to_cdf sorts the list in place, so the list itself becomes the X-axis
y_ar_spec = values_to_cdf(ar_spec_len_list)
y_ar_acc = values_to_cdf(ar_acc_len_list)

# -- Prepare Data for FF --
y_ff_spec = values_to_cdf(ff_spec_len_list)
y_ff_acc = values_to_cdf(ff_acc_len_list)

# -- Plot Lines --

# 1. AR Spec Len (Solid, Color 1) -> Labelled for Legend
# ax.plot(ar_spec_len_list, y_ar_spec, 
#         color=color_ar, linestyle=linestyle_spec, linewidth=linewidth, 
#         label="AR Drafter")

# 2. AR Acc Len (Dashed, Color 1) -> No label
ax.plot(ar_acc_len_list, y_ar_acc, 
        color=color_ar, linestyle=linestyle_acc, linewidth=linewidth,
        label="AR Drafter")

# 3. FF Spec Len (Solid, Color 2) -> Labelled for Legend
ax.plot(ff_spec_len_list, y_ff_spec, 
        color=color_ff, linestyle=linestyle_spec, linewidth=linewidth, 
        )
        # label="FailFast")

# 4. FF Acc Len (Dashed, Color 2) -> No label
ax.plot(ff_acc_len_list, y_ff_acc, 
        color=color_ff, linestyle=linestyle_acc, linewidth=linewidth,
        label="FailFast")


# -- Aesthetics --
ax.set_ylabel("CDF", fontsize=14)
ax.set_xlabel("Number of Tokens", fontsize=14)
ax.set_ylim(0, 1.05)

# Grid styling
ax.set_axisbelow(True)
ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)

# Legend settings
# We only labelled the solid lines, so the legend will show 
# "AR drafter" (Solid Color A) and "FailFast" (Solid Color B)
ax.legend(fontsize=12, loc="best", frameon=False)

plt.tight_layout()
plt.show()

# Optional: Save figure
fig.savefig("cdf_len_distribution.pdf", dpi=300, bbox_inches='tight')
# %%
