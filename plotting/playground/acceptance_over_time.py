# %%
import matplotlib.pyplot as plt
import numpy as np

data = np.array(acceptance_decisions)

plt.figure(figsize=(16, 1))
plt.imshow(data[np.newaxis, :], cmap='Greens', aspect='auto')
plt.yticks([])
plt.xlabel("Token Index")
plt.title("True (green) vs False (white)")
plt.show()

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Example data
data = acceptance_decisions

# Convert to numeric
series = pd.Series(data, dtype=int)

# Compute rolling mean (change window size to control smoothness)
window = 5
rolling_rate = series.rolling(window, center=True).mean()

plt.figure(figsize=(16, 1))
plt.plot(series.index, rolling_rate, label=f"Rolling True Rate (window={window})", linewidth=2)
plt.scatter(series.index, series, alpha=0.3, label="Raw (True=1, False=0)")
plt.ylim(-0.1, 1.1)
plt.xlabel("Index (Time)")
plt.ylabel("True Rate")
plt.legend()
plt.title("True Rate Over Time")
plt.show()

# %%
