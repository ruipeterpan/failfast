# %%
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# --- Regex patterns ---
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")  # strip ANSI escapes
RUNNING_RE = re.compile(r"Running drafter:\s*(\S+)")
PROB_DRAFTER_BRACKET_RE = re.compile(r"\[Problem\s+(\d+),\s*([^\]]+)\]")
PROB_ONLY_RE = re.compile(r"\[Problem\s+(\d+)\]")
ACCEPT_RE = re.compile(r"Acceptance rate:\s*([\d.]+)%")
FWD_RE = re.compile(r"Avg fwd passes/round:\s*([\d.]+)")
SPEED_RE = re.compile(r"Speedup:\s*([\d.]+)x")

# Threshold order (x-axis)
ORDER = [f"dllm_{vlen}" for vlen in range(3, 26)]

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

def parse_log(filename):
    """Parse the log into {problem_id: {drafter: [accept, speed, fwd]}}"""
    data = defaultdict(dict)
    cur_prob = None
    cur_drafter = None

    with open(filename, "r", encoding="utf-8") as f:
        for raw in f:
            line = strip_ansi(raw.rstrip("\n"))

            m_run = RUNNING_RE.search(line)
            if m_run:
                m_prob = PROB_ONLY_RE.search(line)
                if m_prob:
                    cur_prob = int(m_prob.group(1))
                cur_drafter = m_run.group(1)
                continue

            m_pd = PROB_DRAFTER_BRACKET_RE.search(line)
            if m_pd:
                cur_prob = int(m_pd.group(1))
                cur_drafter = m_pd.group(2)

            if cur_prob is None or cur_drafter is None:
                continue

            a = ACCEPT_RE.search(line)
            s = SPEED_RE.search(line)
            f = FWD_RE.search(line)

            if a:
                data[cur_prob].setdefault(cur_drafter, [None, None, None])[0] = float(a.group(1))
            if s:
                data[cur_prob].setdefault(cur_drafter, [None, None, None])[1] = float(s.group(1))
            if f:
                data[cur_prob].setdefault(cur_drafter, [None, None, None])[2] = float(f.group(1))
    return data


def compute_avg(data):
    """Compute average acceptance rate, speedup, and forward passes per drafter."""
    sums = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # acc_sum, spd_sum, fwd_sum, count
    for _, drafter_data in data.items():
        for drafter, (acc, spd, fwd) in drafter_data.items():
            if acc is not None and spd is not None and fwd is not None:
                sums[drafter][0] += acc
                sums[drafter][1] += spd
                sums[drafter][2] += fwd
                sums[drafter][3] += 1

    avg_acc, avg_spd, avg_fwd = {}, {}, {}
    for drafter, (a_sum, s_sum, f_sum, cnt) in sums.items():
        if cnt > 0:
            avg_acc[drafter] = a_sum / cnt
            avg_spd[drafter] = s_sum / cnt
            avg_fwd[drafter] = f_sum / cnt
    return avg_acc, avg_spd, avg_fwd


def extract_threshold(drafter_name):
    """Extract verification length from drafter name, e.g., dllm_0.05_25 -> 25.
    Returns an int or None if not parseable.
    """
    try:
        parts = drafter_name.split("_")
        # Last token is expected to be the verification length (integer)
        vlen = int(parts[-1])
        return vlen
    except Exception:
        return None


def plot_thresholds(avg_acc, avg_spd, avg_fwd):
    """Plot global average stats vs verification length (3..25).

    This version iterates over the actual drafter keys produced by the parser
    (e.g., 'dllm_0.05_25'), extracts the verification length suffix, and
    aggregates points for those with lengths in [3, 25].
    """
    vlengths, accs, spds, fwds = [], [], [], []

    # Iterate over the actual averaged results keys (these are the full drafter names)
    for drafter in avg_acc.keys():
        vlen = extract_threshold(drafter)
        if vlen is None:
            continue
        # Only include verification lengths in the desired range 3..25
        if 3 <= vlen <= 25:
            vlengths.append(vlen)
            accs.append(avg_acc.get(drafter, 0.0))
            spds.append(avg_spd.get(drafter, 0.0))
            fwds.append(avg_fwd.get(drafter, 0.0))

    if not vlengths:
        print("No valid data to plot.")
        return

    # Sort by verification length ascending
    vlengths, accs, spds, fwds = zip(*sorted(zip(vlengths, accs, spds, fwds)))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]}
    )

    # --- Top: Speedup ---
    ax_top.plot(vlengths, spds, marker="o", label="Speedup (x)")
    ax_top.set_ylabel("Speedup (x)", color="tab:blue")
    ax_top.grid(True, linestyle="--", alpha=0.4)
    ax_top.legend(loc="best")
    ax_top.set_title("Average Speedup and Acceptance vs Verification Length (All Problems)")

    # --- Bottom: Acceptance rate (left) + fwd passes (right) ---
    ax_bot.plot(vlengths, accs, marker="o", color="tab:green", label="Acceptance rate (%)")
    ax2 = ax_bot.twinx()
    ax2.plot(vlengths, fwds, marker="s", color="tab:red", label="Avg # fwd passes")

    ax_bot.set_ylabel("Acceptance Rate (%)", color="tab:green")
    ax2.set_ylabel("Avg # Forward Passes", color="tab:red")
    ax_bot.grid(True, linestyle="--", alpha=0.4)

    # --- X axis (verification length 3–25) ---
    ax_bot.set_xlabel("Verification Length")
    xticks = list(range(3, 26))
    ax_bot.set_xticks(xticks)
    ax_bot.set_xticklabels([str(x) for x in xticks])

    fig.tight_layout()
    plt.show()




# --- Run ---
if __name__ == "__main__":
    filename = "/data2/ruipan/diffspec/logs/2025_11_19_17_18_math.ansi"  # your log file
    data = parse_log(filename)
    avg_acc, avg_spd, avg_fwd = compute_avg(data)
    plot_thresholds(avg_acc, avg_spd, avg_fwd)

# %%
