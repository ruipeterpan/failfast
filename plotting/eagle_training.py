# %%
import re
import matplotlib.pyplot as plt

def plot_training_accuracy(file_path):
    # Dictionary to store {(epoch, step): accuracy}
    # This automatically handles duplicates by keeping the last seen value
    data_points = {}

    # Regex to capture: Epoch, Current Step, and Accuracy
    # Matches: "Epoch 7", "31795/51967", and "acc=0.80"
    pattern = re.compile(r"Epoch\s+(\d+):.*?\s+(\d+)/\d+.*acc=([0-9.]+)")

    print(f"Processing {file_path} (this may take a moment for 800k lines)...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                step = int(match.group(2))
                acc = float(match.group(3))
                
                # The dictionary key is a tuple. 
                # If (7, 31795) appears again, it overwrites the previous value.
                data_points[(epoch, step)] = acc

    if not data_points:
        print("No data found.")
        return

    # Sort the data by epoch then by step to ensure a continuous line
    sorted_keys = sorted(data_points.keys())
    accuracies = [data_points[k] for k in sorted_keys]
    
    print(f"Extracted {len(accuracies)} unique steps. Plotting...")

    # Plotting
    fig, ax = plt.subplots(figsize=(4.5, 2.4))
    
    # We use a simple range for X axis (Global Training Steps)
    # ax.plot(range(len(accuracies)), accuracies, linewidth=0.5, alpha=0.8, label='Step Accuracy')

    # Since 800k points is very noisy, let's add a moving average (Window of 500 steps)
    if len(accuracies) > 1000:
        import numpy as np
        window = 250
        moving_avg = np.convolve(accuracies, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(accuracies)), moving_avg, color='#2D6A4F', linewidth=1.5)

    # plt.title("Training Accuracy (Duplicates Removed)")
    ax.set_xlabel("Num Training Steps", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    # plt.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    fig.savefig(f"../figures/eagle3_32b.pdf", dpi=500, bbox_inches='tight')

if __name__ == "__main__":
    # Replace 'training_log.txt' with your actual filename
    # plot_training_accuracy('/data2/ruipan/diffspec/logs/eagle_training/7b.log')
    # plot_training_accuracy('/data2/ruipan/diffspec/logs/eagle_training/14b.log')
    plot_training_accuracy('/data2/ruipan/diffspec/logs/eagle_training/32b.log')

# %%
