# %%
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_boolean_series(data, window=5, figsize=(10, 3), output_dir=None, problem_id=None):
    """
    Visualize a boolean time series with:
      - Top: raster (True = green, False = white)
      - Bottom: rolling average of True rate

    Args:
        data (list or np.ndarray): list of bools
        window (int): rolling window size for smoothing (default: 5)
        figsize (tuple): figure size (default: (10, 5))
    """
    # Convert to numeric array
    arr = np.array(data, dtype=int)
    series = pd.Series(arr)

    # Compute rolling mean
    rolling_rate = series.rolling(window, center=True).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [1, 2]}, 
                                   sharex=True)

    # --- Top: Raster plot ---
    ax1.imshow(arr[np.newaxis, :], cmap='Greens', aspect='auto')
    ax1.set_yticks([])
    ax1.set_title("Accepted (green) vs Rejected (white)")
    
    # --- Bottom: Rolling average ---
    ax2.plot(series.index, rolling_rate, label=f"Rolling Acceptance Rate (window={window})", linewidth=2)
    ax2.scatter(series.index, arr, alpha=0.1, s=10, label="Raw (True=1, False=0)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Token Index")
    ax2.set_ylabel("True Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # fig.suptitle("Boolean Series Visualization", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    if output_dir is not None and problem_id is not None:
        fig.savefig(os.path.join(output_dir, f"{problem_id}.png"))
        plt.close(fig)

# %%
