# %%
import matplotlib.pyplot as plt
import numpy as np

# Data from the log
maxlens = [10, 20, 30, 40, 50, 60, 70]
speedup = [4.355, 4.746, 4.858, 4.891, 4.891, 4.906, 4.902]

# Sort data by 'compute' (X-axis) so the line moves consistently from left to right
data = sorted(zip(maxlens, speedup))
x_values, y_values = zip(*data)

# Create the plot
fig, ax = plt.subplots(figsize=(4.5, 2.4))
ax.plot(x_values, y_values, marker='o', linestyle='-', color='#2D6A4F', markersize=6)

# --- Updated Ticks Logic ---
# # Define all tick positions (0.15, 0.20, ..., 0.70)
# ticks = np.arange(0.15, 0.71, 0.05)
# # Define labels: only show labels for 0.2, 0.3, ..., 0.7
# # We use round(t*100)%10 == 0 to identify the "0.x" values vs the "0.x5" values
# labels = [f"{t:.1f}" if round(t * 100) % 10 == 0 else "" for t in ticks]

# ax.set_xticks(ticks)
# ax.set_xticklabels(labels)
# ---------------------------

# Labeling
ax.set_xlabel("Max Spec Length $N_{max}$", fontsize=12)
ax.set_ylabel("Speedup ($\\times$)", fontsize=12)
# ax.set_ylim(4.25, 5)
ax.set_ylim(3, 5)
# ax.set_xlim(0.1, 0.75)
ax.tick_params(axis='both', which='major', labelsize=10)

# Styling
ax.grid(True, linestyle='--', alpha=0.7)
fig.tight_layout()

# Display the plot
plt.show()
fig.savefig(f"../figures/micro_maxlen.pdf", dpi=500, bbox_inches='tight')
# %%