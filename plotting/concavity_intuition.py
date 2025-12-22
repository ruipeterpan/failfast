# %%
import matplotlib.pyplot as plt

# Data from the log
compute = [2.33, 2.41, 2.55, 2.71, 2.87, 3.20, 3.38, 3.04, 3.57, 3.79, 4.00, 4.23, 4.40, 4.62, 4.85, 5.09, 5.37, 5.67, 6.13]
acc_rate = [53.2, 53.3, 53.6, 54.2, 54.6, 56.3, 57.1, 55.1, 57.8, 58.8, 59.3, 59.4, 59.7, 60.3, 60.0, 60.3, 60.4, 60.5, 60.6]

# Sort data by 'compute' (X-axis) so the line moves consistently from left to right
data = sorted(zip(compute, acc_rate))
x_values, y_values = zip(*data)

# Create the plot
fig, ax = plt.subplots(figsize=(4.5, 2.4))
# fig, ax = plt.subplots(figsize=(3, 2.7))
ax.plot(x_values, y_values, marker='o', linestyle='-', color='#2D6A4F', markersize=6)

# Labeling
# plt.title("Acceptance Rate vs. Num Drafter Passes", fontsize=14)
ax.set_xlabel("Drafter Forward Passes", fontsize=12)
ax.set_ylabel("Acceptance Rate (%)", fontsize=12)
ax.set_ylim(25, 62.5)
ax.set_xlim(2, 6.3)
ax.tick_params(axis='both', which='major', labelsize=12)

# Styling
ax.grid(True, linestyle='--', alpha=0.7)
fig.tight_layout()

# Display the plot
plt.show()
fig.savefig(f"../figures/concavity.pdf", dpi=500, bbox_inches='tight')
# %%
