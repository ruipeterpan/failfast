# %%
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. INPUT DATA
# ---------------------------------------------------------
data = {
    "vLLM_A6000": {
        "math": {
            "ar": [4569.3, 4925.1, 9494.4],        # [spec, ver, total]
            "fast-dllm": [3140.5, 4756.8, 7897.3],
            "failfast": [1660.4, 4136.1, 5796.5],
        },
        "aime": {
            "ar": [6773.4, 7300.9, 14074.3],
            "fast-dllm": [5082.7, 7229, 12311.7],
            "failfast": [2629.5, 6831, 9460.5],
        },
        "gpqa": {
            "ar": [4217.1, 7272.8, 11490],
            "fast-dllm": [3595.3, 7115, 10710],
            "failfast": [2234.4, 6261.2, 8495.6],
        },
    }
}["vLLM_A6000"]

# ---------------------------------------------------------
# 2. CONFIGURATION
# ---------------------------------------------------------

# --- COLOR MODE KNOB ---
# If True: Uses light_color_dict for speculation bars (opacity 1.0).
# If False: Uses color_dict with SPECULATION_OPACITY.
USE_CUSTOM_LIGHT_COLORS = False

# Base Colors (Verification Latency / Bottom Bar)
color_dict = {  # https://partnermarketinghub.withgoogle.com/brands/google-news/visual-identity/color-palette/
    "ar": "#4285F4",        # Google Blue
    "fast-dllm": "#EA4335", # Google Red
    "failfast": "#FBBC04"   # Google Yellow
}

# Lighter Colors (Speculation Latency / Top Bar)
# Used only if USE_CUSTOM_LIGHT_COLORS is True
light_color_dict = {
    "ar": "#D2E3FC",        # Lighter Blue
    "fast-dllm": "#FAD2CF", # Lighter Red
    "failfast": "#FEEFC3"   # Lighter Yellow
}

# Opacity (Only used if USE_CUSTOM_LIGHT_COLORS is False)
SPECULATION_OPACITY = 0.4 

# Display names for the legend
label_map = {
    "ar": "AR Drafter",
    "fast-dllm": "Fast-dLLM",
    "failfast": "FailFast"
}

# ---------------------------------------------------------
# 3. PLOTTING LOGIC
# ---------------------------------------------------------
def plot_grouped_latency(data, base_colors, light_colors, use_light_mode, opacity):
    # Extract dataset names (groups) and method names
    datasets = list(data.keys())  # ['math', 'aime', 'gpqa']
    methods = list(data[datasets[0]].keys()) # ['ar', 'fast-dllm', 'failfast']
    
    x = np.arange(len(datasets))  # the label locations
    total_width = 0.8             # width of the entire group of bars
    bar_width = total_width / len(methods)  # width of individual bars
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate through each method to plot their bars
    for i, method in enumerate(methods):
        # Collect data for this method across all datasets
        spec_latencies = []
        ver_latencies = []
        
        for ds in datasets:
            # Data structure is [speculation, verification, total]
            vals = data[ds][method]
            spec_latencies.append(vals[0])
            ver_latencies.append(vals[1])
        
        # Calculate x position for this specific bar in the group
        bar_x_positions = x + (i * bar_width) - (total_width / 2) + (bar_width / 2)
        
        # Determine colors based on the mode
        c_base = base_colors.get(method, 'gray')
        
        if use_light_mode:
            # Use specific light hex, fully opaque
            c_top = light_colors.get(method, 'lightgray')
            alpha_top = 1.0
        else:
            # Use base hex, lower opacity
            c_top = c_base
            alpha_top = opacity

        lbl = label_map.get(method, method)

        # 1. Plot Verification Latency (Bottom)
        ax.bar(bar_x_positions, ver_latencies, width=bar_width, 
               label=lbl, color=c_base, edgecolor='white', linewidth=0.5)

        # 2. Plot Speculation Latency (Top)
        # Note: We do NOT assign a label here to avoid duplicate legend entries
        ax.bar(bar_x_positions, spec_latencies, width=bar_width, 
               bottom=ver_latencies, color=c_top, alpha=alpha_top, 
               edgecolor='white', linewidth=0.5)

    # Formatting
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Latency Breakdown by Dataset and Method', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets], fontsize=11)
    
    # Legend
    ax.legend(title="Method", loc='upper left')

    # Optional: Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    
    # Show and Save
    plt.show()
    fig.savefig("latency_breakdown.png", dpi=300)


if __name__ == "__main__":
    plot_grouped_latency(
        data, 
        color_dict, 
        light_color_dict, 
        USE_CUSTOM_LIGHT_COLORS, 
        SPECULATION_OPACITY
    )
# %%