# %%
import re
import matplotlib.pyplot as plt

# --- regex to strip ANSI escape sequences (unchanged structure) ---
ansi_escape = re.compile(r'\x1B\[[0-9;]*[mK]')

# --- regex to capture speculation acceptance lines ---
spec_pattern = re.compile(
    r"Speculation round\s+(\d+).*?\((\d+)/(\d+)\)"
)

def parse_log(filename):
    """
    New parser: extracts speculation round ID, accepted tokens, and speculation length.
    Preserves overall structure from previous script.
    """
    rounds = []
    accepted = []
    lengths = []

    with open(filename, "r") as f:
        for line in f:
            clean = ansi_escape.sub("", line)  # strip ANSI colors
            m = spec_pattern.search(clean)
            if m:
                round_id = int(m.group(1))
                acc = int(m.group(2))
                total = int(m.group(3))

                rounds.append(round_id)
                accepted.append(acc)
                lengths.append(total)

    return rounds, accepted, lengths


def plot_rounds(rounds, accepted, lengths):
    """
    Plot accepted tokens vs speculation length over speculation rounds.
    """
    if not rounds:
        print("No valid speculation data found.")
        return

    # Sort by round ID
    data = sorted(zip(rounds, accepted, lengths), key=lambda x: x[0])
    rounds, accepted, lengths = zip(*data)

    plt.figure(figsize=(16, 5))
    plt.plot(rounds, accepted, marker="o", color="tab:green", label="Accepted tokens")
    plt.plot(rounds, lengths, marker="s", color="tab:blue", label="Speculation length")
    ymax = max(lengths)
    plt.yticks(range(0, ymax + 5, 5))
    xmax = max(rounds)
    plt.xticks(range(0, xmax + 5, 5))
    plt.xlabel("Speculation round ID")
    plt.ylabel("Number of tokens")
    plt.title("Speculation Acceptance per Round")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    filename = "../logs/dynamic_frequency.ansi"   # change if needed
    rounds, accepted, lengths = parse_log(filename)
    plot_rounds(rounds, accepted, lengths)

# %%
