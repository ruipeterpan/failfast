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

# Threshold order (x-axis) — keep it but we'll rely on parsed drafter keys for plotting.
ORDER = [f"dllm_{v}" for v in range(3, 26)]


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
    """Extract verification length from drafter name, e.g., dllm_0.05_25 -> 25"""
    try:
        parts = drafter_name.split("_")
        return int(parts[-1])
    except Exception:
        return None


def plot_thresholds(avg_acc, avg_spd, avg_fwd, baseline=None):
    """Plot global average stats.
    baseline: dict with keys 'acc','spd','fwd' or None
    """
    vlengths, accs, spds, fwds = [], [], [], []
    # Collect points from actual avg_* keys (these are full drafter names: dllm_0.05_25)
    for drafter in avg_acc.keys():
        # skip AR baseline drafter if present (we removed it earlier, but keep defensive check)
        if drafter == "ar_None_5":
            continue
        vlen = extract_threshold(drafter)
        if vlen is None:
            continue
        if 3 <= vlen <= 25:
            vlengths.append(vlen)
            accs.append(avg_acc[drafter])
            spds.append(avg_spd[drafter])
            fwds.append(avg_fwd[drafter])

    if not vlengths:
        print("No valid data to plot.")
        return

    # Sort ascending by verification length
    vlengths, accs, spds, fwds = zip(*sorted(zip(vlengths, accs, spds, fwds)))

    # Shared x-axis: top = speedup, bottom = acceptance rate + fwd passes
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]}
    )

    # --- Top: Speedup (blue) ---
    ax_top.plot(vlengths, spds, marker="o", color="tab:blue", label="Speedup (x)")
    ax_top.set_ylabel("Speedup (x)", color="tab:blue")
    ax_top.tick_params(axis="y", labelcolor="tab:blue")
    ax_top.grid(True, linestyle="--", alpha=0.4)
    ax_top.legend(loc="best")
    ax_top.set_title("Average Speedup and Acceptance vs Verification Length (All Problems)")

    # baseline speedup line (blue dashed)
    if baseline and baseline.get("spd") is not None:
        ax_top.axhline(baseline["spd"], linestyle="--", color="tab:blue", label="ar_None_5 baseline")
        # put legend entry for baseline
        ax_top.legend(loc="best")

    # --- Bottom: Acceptance rate (green) + fwd passes (red) ---
    ax_bot.plot(vlengths, accs, marker="o", color="tab:green", label="Acceptance rate (%)")
    ax2 = ax_bot.twinx()
    ax2.plot(vlengths, fwds, marker="s", color="tab:red", label="Avg # fwd passes")

    ax_bot.set_ylabel("Acceptance Rate (%)", color="tab:green")
    ax2.set_ylabel("Avg # Forward Passes", color="tab:red")
    ax_bot.tick_params(axis="y", labelcolor="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax_bot.grid(True, linestyle="--", alpha=0.4)

    # baseline acceptance (green dashed)
    if baseline and baseline.get("acc") is not None:
        ax_bot.axhline(baseline["acc"], linestyle="--", color="tab:green", alpha=0.8, label="ar_None_5 baseline")

    # # baseline forward passes (red dashed) on twin axis
    # if baseline and baseline.get("fwd") is not None:
    #     ax2.axhline(baseline["fwd"], linestyle="--", color="tab:red", alpha=0.8)

    # X axis formatting: verification lengths 3..25
    ax_bot.set_xlabel("Verification Length")
    xticks = np.arange(3, 26, 1)
    ax_bot.set_xticks(xticks)
    ax_bot.set_xticklabels([str(int(x)) for x in xticks])

    # bottom plot should ONLY show its own legends
    handles_bot, labels_bot = ax_bot.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax_bot.legend(handles_bot + handles2, labels_bot + labels2, loc="center right")


    fig.tight_layout()
    plt.show()


# --- Run ---
if __name__ == "__main__":
    # filename = "/data2/ruipan/diffspec/logs/2025_11_13_01_19_math.ansi"  # updated multi-problem log
    # filename = "/data2/ruipan/diffspec/logs/2025_11_13_01_20_aime.ansi"  # updated multi-problem log

    # filename = "/data2/ruipan/diffspec/logs/2025_11_18_00_32_math.ansi"
    filename = "/data2/ruipan/diffspec/logs/2025_11_19_17_18_math.ansi"
    data = parse_log(filename)

    # -------------------------
    # Compute ar_None_5 baseline across problems, then remove those entries from data
    # -------------------------
    baseline_acc_vals = []
    baseline_spd_vals = []
    baseline_fwd_vals = []
    for pid, drafter_data in list(data.items()):
        if "ar_None_5" in drafter_data:
            vals = drafter_data["ar_None_5"]
            # Only include if all three values are present (not None)
            if vals[0] is not None and vals[1] is not None and vals[2] is not None:
                baseline_acc_vals.append(vals[0])
                baseline_spd_vals.append(vals[1])
                baseline_fwd_vals.append(vals[2])
            # remove the entry so it won't be part of the main averages/plots
            del data[pid]["ar_None_5"]

    baseline = {}
    if baseline_acc_vals:
        baseline["acc"] = sum(baseline_acc_vals) / len(baseline_acc_vals)
    else:
        baseline["acc"] = None

    if baseline_spd_vals:
        baseline["spd"] = sum(baseline_spd_vals) / len(baseline_spd_vals)
    else:
        baseline["spd"] = None

    if baseline_fwd_vals:
        baseline["fwd"] = sum(baseline_fwd_vals) / len(baseline_fwd_vals)
    else:
        baseline["fwd"] = None

    if baseline["acc"] is not None:
        print(f"ar_None_5 baseline (Acceptance): {baseline['acc']:.3f}%")
    if baseline["spd"] is not None:
        print(f"ar_None_5 baseline (Speedup): {baseline['spd']:.3f}x")
    if baseline["fwd"] is not None:
        print(f"ar_None_5 baseline (Avg fwd passes): {baseline['fwd']:.3f}")

    # Now compute global averages excluding ar_None_5 (since we've removed it from data)
    avg_acc, avg_spd, avg_fwd = compute_global_avg(data)

    # Plot and pass the baseline for horizontal reference lines
    plot_thresholds(avg_acc, avg_spd, avg_fwd, baseline=baseline)

# %%
