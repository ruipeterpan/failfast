# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import os
import pickle

target_model = "Qwen2.5-32B-Instruct"
dataset = "math"
drafter_name = "ar_None_sf_8"
# question_ids = [0, 3, 4, 5, 6, 8]
# question_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# question_ids = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
question_ids = [0, 4, 15, 18, 21]
stats_list = []


for qid in question_ids:
    stats_path = f"/data2/ruipan/diffspec/pickles/{target_model}/{dataset}/{qid}/{drafter_name}/1024.pickle"
    with open(stats_path, "rb") as f:
        pickle_file = pickle.load(f)
    stats_each_round = pickle_file["stats_each_round"]
    stats_list.append(stats_each_round)



def get_boolean_decision_from_stats_each_round(stats_each_round):
    """Reused from original script: converts round stats to a boolean array."""
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

def visualize_multiple_sessions_raster(stats_list, labels=None, figsize=None, output_dir=None, filename=None):
    """
    Plots multiple raster plots (one for each stats_each_round in stats_list)
    with a unified X-axis matching the longest sequence.
    """
    num_sessions = len(stats_list)
    
    # 1. Pre-calculate all decisions and find the global maximum length
    all_decisions = [get_boolean_decision_from_stats_each_round(s) for s in stats_list]
    max_len = max(len(d) for d in all_decisions)
    
    custom_cmap = ListedColormap(["white", "#2D6A4F"])
    
    if figsize is None:
        figsize = (4.5, 0.15 * num_sessions + 1)
        
    fig, axes = plt.subplots(
        num_sessions, 1, 
        figsize=figsize, 
        sharex=True,
        gridspec_kw={'hspace': 1}  # Adjust this value (0.0 is no space, 0.5 is lots of space)
    )
    
    if num_sessions == 1:
        axes = [axes]

    for i, decisions in enumerate(all_decisions):
        ax = axes[i]
        arr = np.array(decisions, dtype=int)
        
        # 2. Use 'extent' to ensure the pixels align with the token indices correctly
        # [left, right, bottom, top]
        curr_len = len(decisions)
        ax.imshow(arr[np.newaxis, :], 
                #   cmap='Greens', 
                  cmap=custom_cmap, 
                  aspect='auto', 
                  interpolation='nearest',
                  extent=[-0.5, curr_len - 0.5, -0.5, 0.5])
        
        ax.set_yticks([])
        if labels and i < len(labels):
            ax.set_ylabel(labels[i], rotation=0, ha='right', va='center')
        
        # Optional: Grid lines logic
        if max_len < 200:
            ax.set_xticks(np.arange(-.5, curr_len, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)

    # 3. Explicitly set the x-limit to the global maximum
    # imshow centers pixel 0 at coordinate 0.0, so the edge is -0.5
    axes[0].set_xlim(-0.5, max_len - 0.5)

    # Label the bottom subplot
    axes[-1].set_xlabel("Output Token Index", fontsize=10)
    # Add a common y-label for the entire figure
    fig.supylabel("Query ID", fontsize=10)
    
    plt.tight_layout()
    
    if output_dir is not None and filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{filename}.png"))
        print(f"Saved multiple session plot to {output_dir}/{filename}.png")
    
    plt.show()
    fig.savefig("../figures/difficulty_intuition.pdf", dpi=500, bbox_inches='tight')

# Example Usage:
# visualize_multiple_sessions_raster(stats_list)
visualize_multiple_sessions_raster(stats_list, labels=question_ids)
# %%
