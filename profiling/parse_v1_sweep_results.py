# %%
import re
from collections import defaultdict

# --- Regex patterns (unchanged) ---
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
RUNNING_RE = re.compile(r"Running drafter:\s*(\S+)")
PROB_DRAFTER_BRACKET_RE = re.compile(r"\[Problem\s+(\d+),\s*([^\]]+)\]")
PROB_ONLY_RE = re.compile(r"\[Problem\s+(\d+)\]")
ACCEPT_RE = re.compile(r"Acceptance rate:\s*([\d.]+)%")
FWD_RE = re.compile(r"Avg fwd passes/round:\s*([\d.]+)")
SPEED_RE = re.compile(r"Speedup:\s*([\d.]+)x")

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


# ----------------------------------------------------------------------
# NEW: extract per-problem stats and compute cross-problem averages
# ----------------------------------------------------------------------

AR_NAME = "ar_None_sf_None_None"
SF_NAME = "dllm_0.05_sf_None_None"

def analyze(data):
    """Print per-problem summaries and compute averages."""

    # Accumulate averages
    sums = defaultdict(float)
    counts = defaultdict(int)

    print("=== Per-Problem Results ===")
    for pid in sorted(data.keys()):
        drafter_data = data[pid]

        ar_spd = drafter_data.get(AR_NAME, [None, None, None])[1]
        sf_spd = drafter_data.get(SF_NAME, [None, None, None])[1]

        if ar_spd is None or sf_spd is None:
            print(f"[Problem {pid}] Missing AR or SF config — skipping.")
            continue

        # record sums
        sums[AR_NAME] += ar_spd
        counts[AR_NAME] += 1

        sums[SF_NAME] += sf_spd
        counts[SF_NAME] += 1

        # find best among other 30 configs (exclude AR and SF)
        best_name = None
        best_spd = -1

        for name, (_, spd, _) in drafter_data.items():
            if spd is None:
                continue
            if name in (AR_NAME, SF_NAME):
                continue
            if spd > best_spd:
                best_spd = spd
                best_name = name

            # accumulate averages
            sums[name] += spd
            counts[name] += 1

        diff = best_spd - sf_spd

        print(f"[Problem {pid}] AR speedup = {ar_spd:.3f}x")
        print(f"[Problem {pid}] SF (dllm_0.05_sf_None_None) speedup = {sf_spd:.3f}x")
        print(f"[Problem {pid}] Best other config = {best_name} ({best_spd:.3f}x), win over SF = {diff:.3f}x\n")

    # ------------------------------------------------------------------
    # Global averages
    # ------------------------------------------------------------------
    avg = {k: (sums[k] / counts[k]) for k in sums.keys() if counts[k] > 0}

    # Best config overall
    best_global_cfg = max(avg.items(), key=lambda x: x[1])

    print("=== Global Averages (Across All Problems) ===")
    print(f"AR average speedup: {avg[AR_NAME]:.3f}x")
    print(f"SF (dllm_0.05_sf_None_None) average speedup: {avg[SF_NAME]:.3f}x")
    print(f"Best average config overall: {best_global_cfg[0]} with {best_global_cfg[1]:.3f}x\n")

    return avg

# %%
# Example usage:
# data = parse_log("/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_24_22_22_math.ansi")
data = parse_log("/scratch/gpfs/RAVIAN/rp2773/data/diffspec/logs/2025_11_24_22_06_aime.ansi")
analyze(data)
