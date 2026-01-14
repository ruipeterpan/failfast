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
ORDER = [f"dllm_{x:.2f}".rstrip("0").rstrip(".") for x in np.arange(0.9, 0.0, -0.05)]

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


# def compute_global_avg(data):
#     """
#     Compute the global average across all problems.
#     Returns: dicts avg_acc, avg_spd, avg_fwd indexed by drafter name.
#     """
#     sums = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # acc_sum, spd_sum, fwd_sum, count
#     for _, drafter_data in data.items():
#         for drafter, (acc, spd, fwd) in drafter_data.items():
#             if acc is not None and spd is not None and fwd is not None:
#                 sums[drafter][0] += acc
#                 sums[drafter][1] += spd
#                 sums[drafter][2] += fwd
#                 sums[drafter][3] += 1

#     avg_acc, avg_spd, avg_fwd = {}, {}, {}
#     for drafter, (a_sum, s_sum, f_sum, cnt) in sums.items():
#         if cnt > 0:
#             avg_acc[drafter] = a_sum / cnt
#             avg_spd[drafter] = s_sum / cnt
#             avg_fwd[drafter] = f_sum / cnt
#     return avg_acc, avg_spd, avg_fwd


def compute_global_avg(data):
    """Compute averages across all problems and print diagnostics."""
    drafter_sums = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    max_per_problem = []

    for pid, drafter_data in data.items():
        problem_max_speedup = 0.0
        for drafter, (acc, spd, fwd) in drafter_data.items():
            if spd is None or acc is None or fwd is None:
                continue
            drafter_sums[drafter][0] += acc
            drafter_sums[drafter][1] += spd
            drafter_sums[drafter][2] += fwd
            drafter_sums[drafter][3] += 1
            problem_max_speedup = max(problem_max_speedup, spd)
        if problem_max_speedup > 0:
            max_per_problem.append(problem_max_speedup)

    avg_acc, avg_speedup, avg_fwd = {}, {}, {}
    for drafter, (a_sum, s_sum, f_sum, cnt) in drafter_sums.items():
        if cnt > 0:
            avg_acc[drafter] = a_sum / cnt
            avg_speedup[drafter] = s_sum / cnt
            avg_fwd[drafter] = f_sum / cnt

    # Identify best global drafter (single config for all problems)
    best_drafter = max(avg_speedup.items(), key=lambda x: x[1])[0] if avg_speedup else None
    best_global_speedup = avg_speedup[best_drafter] if best_drafter else 0.0

    # Compute average of per-problem max speedups (oracle upper bound)
    avg_of_max_speedups = sum(max_per_problem) / len(max_per_problem) if max_per_problem else 0.0

    print(f"Best global drafter config: {best_drafter}")
    print(f"Average speedup of best global config: {best_global_speedup:.3f}x")
    print(f"Average of per-problem max speedups (oracle upper bound): {avg_of_max_speedups:.3f}x")

    return avg_acc, avg_speedup, avg_fwd



def extract_threshold(drafter_name):
    """Extract numeric threshold from drafter name, e.g., dllm_0.9 → 0.9"""
    try:
        return float(drafter_name.split("_")[1])
    except Exception:
        return None


def plot_thresholds(avg_acc, avg_spd, avg_fwd):
    """Plot global average stats."""
    thresholds, accs, spds, fwds = [], [], [], []
    for d in ORDER:
        thr = extract_threshold(d)
        if thr is not None and d in avg_acc:
            thresholds.append(thr)
            accs.append(avg_acc[d])
            spds.append(avg_spd[d])
            fwds.append(avg_fwd[d])

    if not thresholds:
        print("No valid data to plot.")
        return

    # Sort ascending (0.05 → 0.9)
    thresholds, accs, spds, fwds = zip(*sorted(zip(thresholds, accs, spds, fwds)))

    # Shared x-axis: top = speedup, bottom = acceptance rate + fwd passes
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]}
    )

    # --- Top: Speedup ---
    ax_top.plot(thresholds, spds, marker="o", color="tab:blue", label="Speedup (x)")
    ax_top.set_ylabel("Speedup (x)", color="tab:blue")
    ax_top.tick_params(axis="y", labelcolor="tab:blue")
    ax_top.grid(True, linestyle="--", alpha=0.4)
    ax_top.legend(loc="best")
    ax_top.set_title("Average Speedup and Acceptance vs Drafter Threshold (All Problems)")

    # --- Bottom: Acceptance rate (left) + fwd passes (right) ---
    ax_bot.plot(thresholds, accs, marker="o", color="tab:green", label="Acceptance rate (%)")
    ax2 = ax_bot.twinx()
    ax2.plot(thresholds, fwds, marker="s", color="tab:red", label="Avg # fwd passes")

    ax_bot.set_ylabel("Acceptance Rate (%)", color="tab:green")
    ax2.set_ylabel("Avg # Forward Passes", color="tab:red")
    ax_bot.tick_params(axis="y", labelcolor="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax_bot.grid(True, linestyle="--", alpha=0.4)

    # --- Shared x-axis formatting ---
    ax_bot.set_xlabel("Drafter Threshold")
    xticks = np.arange(0.05, 0.95, 0.05)
    ax_bot.set_xticks(xticks)
    ax_bot.set_xticklabels([f"{x:.2f}".rstrip("0").rstrip(".") for x in xticks])

    fig.tight_layout()
    plt.show()


# --- Run ---
if __name__ == "__main__":
    # filename = "/data2/USERNAME/failfast/logs/2025_11_13_01_19_math.ansi"  # updated multi-problem log
    # filename = "/data2/USERNAME/failfast/logs/2025_11_13_01_20_aime.ansi"  # updated multi-problem log
    
    # filename = "/data2/USERNAME/failfast/logs/2025_11_18_00_32_math.ansi"
    filename = "/data2/USERNAME/failfast/logs/2025_11_18_00_38_aime.ansi"
    data = parse_log(filename)
    avg_acc, avg_spd, avg_fwd = compute_global_avg(data)
    plot_thresholds(avg_acc, avg_spd, avg_fwd)

# %%