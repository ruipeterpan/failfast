import re
import matplotlib.pyplot as plt

def plot_training_accuracy(file_path):
    accuracies = []
    
    # Regex pattern to find 'acc=' followed by a decimal number
    # Matches "acc=0.80" and captures "0.80"
    acc_pattern = re.compile(r"acc=([0-9.]+)")

    print(f"Reading {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Search for the accuracy pattern in the current line
                match = acc_pattern.search(line)
                if match:
                    # Convert the captured string to a float
                    acc_value = float(match.group(1))
                    accuracies.append(acc_value)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    if not accuracies:
        print("No accuracy data found in the log file.")
        return

    print(f"Extracted {len(accuracies)} data points. Plotting...")

    # Plotting the data
    plt.figure(figsize=(12, 6))
    plt.plot(accuracies, linewidth=0.5, color='blue', label='Accuracy per Step')
    
    # Optional: Calculate a moving average for 800k lines to see trends clearly
    # if len(accuracies) > 1000:
    #     import pandas as pd
    #     smooth_acc = pd.Series(accuracies).rolling(window=100).mean()
    #     plt.plot(smooth_acc, color='red', label='Moving Average (100 steps)')

    plt.title("Training Accuracy Over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace 'training_log.txt' with your actual filename
    plot_training_accuracy('~/data/failfast_eagle/logs/2025_12_22_13_34.log')
