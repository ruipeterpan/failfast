# %%
import re
from collections import defaultdict

# --- Regex patterns (same as original) ---
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
RUNNING_RE = re.compile(r"Running drafter:\s*(\S+)")
PROB_DRAFTER_BRACKET_RE = re.compile(r"\[Problem\s+(\d+),\s*([^\]]+)\]")
PROB_ONLY_RE = re.compile(r"\[Problem\s+(\d+)\]")

ACCEPT_RE = re.compile(r"Acceptance rate:\s*([\d.]+)%")
FWD_RE = re.compile(r"Avg fwd passes/round:\s*([\d.]+)")

# New: capture WHICH engine’s speedup it is
HF_SPEED_RE = re.compile(r"\[HuggingFace\]\s*Speedup:\s*([\d.]+)x")
VLLM_SPEED_RE = re.compile(r"\[vLLM\]\s*Speedup:\s*([\d.]+)x")


def strip_ansi(s):
    return ANSI_RE.sub("", s)


def parse_log(filename):
    """
    Returns:
      data[problem_id][drafter] = {
          "hf": float or None,
          "vllm": float or None,
      }
    """
    data = defaultdict(lambda: defaultdict(lambda: {"hf": None, "vllm": None}))
    cur_prob = None
    cur_drafter = None

    with open(filename, "r", encoding="utf-8") as f:
        for raw in f:
            line = strip_ansi(raw.rstrip("\n"))

            # Detect: Running drafter: X
            m_run = RUNNING_RE.search(line)
            if m_run:
                m_prob = PROB_ONLY_RE.search(line)
                if m_prob:
                    cur_prob = int(m_prob.group(1))
                cur_drafter = m_run.group(1)
                continue

            # Detect: [Problem X, drafter]
            m_pd = PROB_DRAFTER_BRACKET_RE.search(line)
            if m_pd:
                cur_prob = int(m_pd.group(1))
                cur_drafter = m_pd.group(2)

            if cur_prob is None or cur_drafter is None:
                continue

            # Extract engine-specific speedups
            hf = HF_SPEED_RE.search(line)
            if hf:
                data[cur_prob][cur_drafter]["hf"] = float(hf.group(1))

            v = VLLM_SPEED_RE.search(line)
            if v:
                data[cur_prob][cur_drafter]["vllm"] = float(v.group(1))

    return data


def compute_averages_and_print(data):
    """
    Compute:
      - Average HF speedup per drafter
      - Average vLLM speedup per drafter
      - Best drafter for HF
      - Best drafter for vLLM
    """

    sums = defaultdict(lambda: {"hf_sum": 0.0, "hf_cnt": 0,
                                "v_sum": 0.0, "v_cnt": 0})

    for pid, drafters in data.items():
        for drafter, stats in drafters.items():
            if stats["hf"] is not None:
                sums[drafter]["hf_sum"] += stats["hf"]
                sums[drafter]["hf_cnt"] += 1
            if stats["vllm"] is not None:
                sums[drafter]["v_sum"] += stats["vllm"]
                sums[drafter]["v_cnt"] += 1

    # Compute averages
    avg_hf = {}
    avg_vllm = {}

    for drafter, s in sums.items():
        if s["hf_cnt"] > 0:
            avg_hf[drafter] = s["hf_sum"] / s["hf_cnt"]
        if s["v_cnt"] > 0:
            avg_vllm[drafter] = s["v_sum"] / s["v_cnt"]

    # Print results
    print("=== Average HuggingFace Speedups per Drafter ===")
    for d, v in sorted(avg_hf.items(), key=lambda x: -x[1]):
        print(f"{d}: {v:.3f}x")

    print("\n=== Average vLLM Speedups per Drafter ===")
    for d, v in sorted(avg_vllm.items(), key=lambda x: -x[1]):
        print(f"{d}: {v:.3f}x")

    # Best configs
    best_hf = max(avg_hf.items(), key=lambda x: x[1]) if avg_hf else None
    best_v = max(avg_vllm.items(), key=lambda x: x[1]) if avg_vllm else None

    print("\n=== Best Drafter Configs ===")
    if best_hf:
        print(f"Best HuggingFace Drafter: {best_hf[0]} ({best_hf[1]:.3f}x)")
    if best_v:
        print(f"Best vLLM Drafter: {best_v[0]} ({best_v[1]:.3f}x)")


if __name__ == "__main__":
    # log_file = "/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_25_21_07_math.ansi"  # AR drafter, MATH
    # log_file = "/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_25_21_11_aime.ansi"  # AR drafter, AIME
    
    # log_file = "/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_25_21_23_math.ansi"  # dLLM drafter, threshold 0.05
    # log_file = "/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_25_21_23_aime.ansi"  # dLLM drafter, threshold 0.05
    
    # log_file = "/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_25_21_32_math.ansi"  # dLLM drafter, threshold 0.9
    # log_file = "/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_25_21_31_aime.ansi"  # dLLM drafter, threshold 0.9

    # log_file = "/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_26_02_13_math.ansi"  # dLLM drafter, threshold 0.05, more sweeps
    # log_file = "/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_26_02_14_aime.ansi"  # dLLM drafter, threshold 0.05, more sweeps
    
    # log_file = "/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_26_00_49_math.ansi"  # dLLM drafter, threshold 0.9, more sweeps
    # log_file = "/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_26_01_55_aime.ansi"  # dLLM drafter, threshold 0.9, more sweeps
    
    # log_file = "/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_25_22_26_math.ansi"  # lookahead dynamic frequency sweep
    # log_file = "/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_25_22_27_aime.ansi"  # lookahead dynamic frequency sweep

    data = parse_log(log_file)
    compute_averages_and_print(data)

# %%
