# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches # <--- Added for custom legend handle

# ---------------------------------------------------------
# 1. INPUT DATA
# ---------------------------------------------------------
# Groups are now "32B" and "14B".
# Added "fast-dllm+" with placeholder data (slightly improved over fast-dllm).
data = {
    "Qwen2.5-32B": {
        "ar": [4466, 5712, 10178],         # [spec, ver, total]
        "fast-dllm": [3423, 5571, 8995],
        # "fast-dllm+": [4013, 4924, 8937], # <--- New Scheme
        "failfast": [1899, 5119, 7019],
    },
    "Qwen2.5-14B": {
        "ar": [3537, 3299, 6837],
        "fast-dllm": [2855, 3259, 6115],
        # "fast-dllm+": [3250, 2770, 6020],  # <--- New Scheme
        "failfast": [1872, 2417, 4289],
    }
}

# ---------------------------------------------------------
# 2. CONFIGURATION
# ---------------------------------------------------------

# --- COLOR MODE KNOB ---
USE_CUSTOM_LIGHT_COLORS = True

# Base Colors (Verification Latency / Bottom Bar)
# Added Green for fast-dllm+
color_dict = {  
    "ar": "#52B788",         # Google Blue
    "fast-dllm": "#40916C",  # Google Red
    # "fast-dllm+": "#34A853", # Google Green (New)
    "failfast": "#2D6A4F"    # Google Yellow
} # https://coolors.co/palette/d8f3dc-b7e4c7-95d5b2-74c69d-52b788-40916c-2d6a4f

# Lighter Colors (Speculation Latency / Top Bar)
light_color_dict = {
    "ar": "#95D5B2",         # Lighter Blue
    "fast-dllm": "#95D5B2",  # Lighter Red
    # "fast-dllm+": "#CEEAD6", # Lighter Green (New)
    "failfast": "#95D5B2"    # Lighter Yellow
}

# Opacity
SPECULATION_OPACITY = 0.7

# Display names for the legend
label_map = {
    "ar": "AR Drafter",
    "fast-dllm": "Fast-dLLM",
    # "fast-dllm+": "Fast-dLLM+",
    "failfast": "FailFast"
}

# ---------------------------------------------------------
# 3. PLOTTING LOGIC
# ---------------------------------------------------------
def plot_grouped_latency(data, base_colors, light_colors, use_light_mode, opacity):
    # Extract model names (groups) and method names
    models = list(data.keys())   # ['32B', '14B']
    # Ensure methods are in the order defined in the dict
    methods = list(data[models[0]].keys()) # ['ar', 'fast-dllm', 'fast-dllm+', 'failfast']
    
    x = np.arange(len(models))    # the label locations
    total_width = 0.8             # width of the entire group of bars
    bar_width = total_width / len(methods)  # width of individual bars
    
    fig, ax = plt.subplots(figsize=(4.5, 2.7))

    # Iterate through each method to plot their bars
    for i, method in enumerate(methods):
        # Collect data for this method across all models
        spec_latencies = []
        ver_latencies = []
        
        for model in models:
            # Data structure is [speculation, verification, total]
            vals = data[model][method]
            spec_latencies.append(vals[0] / 1e3)
            ver_latencies.append(vals[1] / 1e3)
        
        # Calculate x position for this specific bar in the group
        bar_x_positions = x + (i * bar_width) - (total_width / 2) + (bar_width / 2)
        
        # Determine colors based on the mode
        c_base = base_colors.get(method, 'gray')
        
        if use_light_mode:
            c_top = light_colors.get(method, 'lightgray')
            alpha_top = 1.0
        else:
            c_top = c_base
            alpha_top = opacity

        lbl = label_map.get(method, method)

        # 1. Plot Verification Latency (Bottom)
        ax.bar(bar_x_positions, ver_latencies, width=bar_width, 
               label=lbl, color=c_base, edgecolor='white', linewidth=0.5)

        # 2. Plot Speculation Latency (Top)
        ax.bar(bar_x_positions, spec_latencies, width=bar_width, 
               bottom=ver_latencies, color=c_top, alpha=alpha_top, 
               edgecolor='white', linewidth=0.5)

    # Formatting
    ax.set_ylabel('Latency (s)', fontsize=10)
    ax.set_xlabel('Target Model', fontsize=10)
    # ax.set_title('Latency Breakdown by Model and Method', fontsize=14)
    ax.set_xticks(x)
    # ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.set_xticklabels(models, fontsize=10)
    
    # Legend
    # ax.legend(title="Method", loc='upper right')
    # ax.legend(loc="upper center", ncols=3, fontsize=12, bbox_to_anchor=(0.5, 1.2), columnspacing=0.8, handlelength=0.8, frameon=False, borderaxespad=0)  # , mode="expand"

    # Legend
    # Retrieve existing handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Create the custom handle for "spec. latency"
    spec_patch = mpatches.Patch(color='#95D5B2', label='Speculation')
    # handles.append(spec_patch)
    # labels.append('Speculation Latency')
    handles.insert(0, spec_patch)
    labels.insert(0, 'Speculation')

    # Pass modified handles/labels to legend and adjust layout (ncols=2 for 2x2 grid)
    ax.legend(handles=handles, labels=labels, loc="upper center", ncols=4, fontsize=10, 
            bbox_to_anchor=(0.5, 1.2), columnspacing=0.8, handlelength=0.8, frameon=False, borderaxespad=0)



    # Grid and Layout
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    
    # Show and Save
    plt.show()
    # fig.savefig("latency_breakdown_models.png", dpi=300)
    fig.savefig(f"../figures/latency_breakdown_models.pdf", dpi=500, bbox_inches='tight')



if __name__ == "__main__":
    plot_grouped_latency(
        data, 
        color_dict, 
        light_color_dict, 
        USE_CUSTOM_LIGHT_COLORS, 
        SPECULATION_OPACITY
    )
# %%