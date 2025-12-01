# %%
import re
from collections import defaultdict

# --- Regex patterns ---
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
RUNNING_RE = re.compile(r"Running drafter:\s*(\S+)")
PROB_DRAFTER_BRACKET_RE = re.compile(r"\[Problem\s+(\d+),\s*([^\]]+)\]")
PROB_ONLY_RE = re.compile(r"\[Problem\s+(\d+)\]")

ACCEPT_RE = re.compile(r"Acceptance rate:\s*([\d.]+)%")
# Capture avg forwards and optional (num_out/num_rounds)
FWD_RE = re.compile(r"Avg fwd passes/round:\s*([\d.]+)(?:\s*\((\d+)/(\d+)\))?")

# Generic engine speedup capture:
# captures engine name inside brackets and the speedup number
# e.g. "[HuggingFace_A6000] Speedup: 2.26x" -> engine="HuggingFace_A6000", speed="2.26"
ENGINE_SPEED_RE = re.compile(r"\[([^\]]+)\]\s*Speedup:\s*([\d.]+)x")

# Capture the latency pair like "2903.6ms/6480.4ms"
LATENCY_PAIR_RE = re.compile(r"([\d.]+)ms/([\d.]+)ms")

# Accepted/speculated: avg X/Y, max A/B
ACCEPTED_SPEC_RE = re.compile(
    r"Accepted/speculated:\s*avg\s*([\d.]+)/([\d.]+),\s*max\s*([\d.]+)/([\d.]+)"
)


def strip_ansi(s):
    return ANSI_RE.sub("", s)


def parse_log(filename):
    """
    Returns:
      data[problem_id][drafter] = {
          "engines": { engine_name: {"speed": float or None, "spec_ms": float or None, "total_ms": float or None}, ... },
          "accept_rate": float or None,
          "avg_acc": float or None,
          "avg_spec": float or None,
          "max_acc": float or None,
          "max_spec": float or None,
          "num_rounds": int or None,
      }
    """
    data = defaultdict(
        lambda: defaultdict(
            lambda: {
                "engines": {},
                "accept_rate": None,
                "avg_acc": None,
                "avg_spec": None,
                "max_acc": None,
                "max_spec": None,
                "num_rounds": None,
            }
        )
    )
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

            # Generic engine speedup lines (may appear multiple engines)
            m_eng = ENGINE_SPEED_RE.search(line)
            if m_eng:
                engine = m_eng.group(1)
                try:
                    speed = float(m_eng.group(2))
                except ValueError:
                    speed = None

                # default engine entry for this problem/drafter
                ed = {"speed": speed, "spec_ms": None, "total_ms": None}

                # try to capture latency pair on the same line
                m_lat = LATENCY_PAIR_RE.search(line)
                if m_lat:
                    try:
                        ed["spec_ms"] = float(m_lat.group(1))
                        ed["total_ms"] = float(m_lat.group(2))
                    except ValueError:
                        pass

                data[cur_prob][cur_drafter]["engines"][engine] = ed

            # Acceptance rate
            m_acc = ACCEPT_RE.search(line)
            if m_acc:
                try:
                    data[cur_prob][cur_drafter]["accept_rate"] = float(
                        m_acc.group(1)
                    )
                except ValueError:
                    pass

            # Avg fwd passes/round: capture rounds if present (321/107 -> rounds=107)
            m_fwd = FWD_RE.search(line)
            if m_fwd:
                if m_fwd.group(3):
                    try:
                        data[cur_prob][cur_drafter]["num_rounds"] = int(
                            m_fwd.group(3)
                        )
                    except ValueError:
                        pass

            # Accepted/speculated: avg X/Y, max A/B
            m_as = ACCEPTED_SPEC_RE.search(line)
            if m_as:
                try:
                    data[cur_prob][cur_drafter]["avg_acc"] = float(m_as.group(1))
                    data[cur_prob][cur_drafter]["avg_spec"] = float(m_as.group(2))
                    data[cur_prob][cur_drafter]["max_acc"] = float(m_as.group(3))
                    data[cur_prob][cur_drafter]["max_spec"] = float(m_as.group(4))
                except ValueError:
                    pass

    return data


def _format_stats_for_drafter(sums, drafter):
    """
    Returns: acc_rate_str, num_rounds_str, avg_acc_spec_str, max_acc_spec_str
    """
    s = sums[drafter]

    # Acceptance rate average
    if s["acc_rate_cnt"] > 0:
        acc_rate_avg = s["acc_rate_sum"] / s["acc_rate_cnt"]
        acc_rate_str = f"{acc_rate_avg:.1f}%"
    else:
        acc_rate_str = "N/A"

    # average number of rounds (float with one decimal)
    if s["rounds_cnt"] > 0:
        rounds_avg = s["rounds_sum"] / s["rounds_cnt"]
        num_rounds_str = f"{rounds_avg:.1f}"
    else:
        num_rounds_str = "N/A"

    # average of avg accepted/speculated lengths across problems
    if s["avg_acc_cnt"] > 0 and s["avg_spec_cnt"] > 0:
        avg_acc_avg = s["avg_acc_sum"] / s["avg_acc_cnt"]
        avg_spec_avg = s["avg_spec_sum"] / s["avg_spec_cnt"]
        avg_acc_spec_str = f"{avg_acc_avg:.1f}/{avg_spec_avg:.1f}"
    elif s["avg_acc_cnt"] > 0:
        avg_acc_avg = s["avg_acc_sum"] / s["avg_acc_cnt"]
        avg_acc_spec_str = f"{avg_acc_avg:.1f}/N/A"
    elif s["avg_spec_cnt"] > 0:
        avg_spec_avg = s["avg_spec_sum"] / s["avg_spec_cnt"]
        avg_acc_spec_str = f"N/A/{avg_spec_avg:.1f}"
    else:
        avg_acc_spec_str = "N/A"

    # max of max accepted/speculated across problems
    max_acc = s["max_acc_max"]
    max_spec = s["max_spec_max"]
    if max_acc is not None and max_spec is not None:
        max_acc_spec_str = f"{max_acc:.1f}/{max_spec:.1f}"
    elif max_acc is not None:
        max_acc_spec_str = f"{max_acc:.1f}/N/A"
    elif max_spec is not None:
        max_acc_spec_str = f"N/A/{max_spec:.1f}"
    else:
        max_acc_spec_str = "N/A"

    return acc_rate_str, num_rounds_str, avg_acc_spec_str, max_acc_spec_str


def compute_averages_and_print(data):
    """
    Aggregate per-drafter sums, compute per-engine averages (including latencies),
    and print annotated tables for each engine plus best-drafter-per-engine summary.
    """

    # sums per drafter; store per-engine sums under "engines"
    sums = defaultdict(
        lambda: {
            "engines": {},  # engine -> dict with speed/spec/total sums and counts
            # acceptance & accepted/spec stats
            "acc_rate_sum": 0.0,
            "acc_rate_cnt": 0,
            "avg_acc_sum": 0.0,
            "avg_acc_cnt": 0,
            "avg_spec_sum": 0.0,
            "avg_spec_cnt": 0,
            "max_acc_max": None,
            "max_spec_max": None,
            # rounds
            "rounds_sum": 0.0,
            "rounds_cnt": 0,
        }
    )

    # Fill sums
    for pid, drafters in data.items():
        for drafter, stats in drafters.items():
            # engines: dynamic
            for engine, eng_stats in stats.get("engines", {}).items():
                ed = sums[drafter]["engines"].setdefault(
                    engine,
                    {
                        "speed_sum": 0.0,
                        "speed_cnt": 0,
                        "spec_sum": 0.0,
                        "spec_cnt": 0,
                        "total_sum": 0.0,
                        "total_cnt": 0,
                    },
                )
                # speed
                sp = eng_stats.get("speed")
                if sp is not None:
                    ed["speed_sum"] += sp
                    ed["speed_cnt"] += 1
                # spec latency
                spec = eng_stats.get("spec_ms")
                if spec is not None:
                    ed["spec_sum"] += spec
                    ed["spec_cnt"] += 1
                # total latency
                tot = eng_stats.get("total_ms")
                if tot is not None:
                    ed["total_sum"] += tot
                    ed["total_cnt"] += 1

            if stats.get("accept_rate") is not None:
                sums[drafter]["acc_rate_sum"] += stats["accept_rate"]
                sums[drafter]["acc_rate_cnt"] += 1

            if stats.get("avg_acc") is not None:
                sums[drafter]["avg_acc_sum"] += stats["avg_acc"]
                sums[drafter]["avg_acc_cnt"] += 1

            if stats.get("avg_spec") is not None:
                sums[drafter]["avg_spec_sum"] += stats["avg_spec"]
                sums[drafter]["avg_spec_cnt"] += 1

            if stats.get("max_acc") is not None:
                cur = sums[drafter]["max_acc_max"]
                if cur is None or stats["max_acc"] > cur:
                    sums[drafter]["max_acc_max"] = stats["max_acc"]

            if stats.get("max_spec") is not None:
                cur = sums[drafter]["max_spec_max"]
                if cur is None or stats["max_spec"] > cur:
                    sums[drafter]["max_spec_max"] = stats["max_spec"]

            if stats.get("num_rounds") is not None:
                sums[drafter]["rounds_sum"] += stats["num_rounds"]
                sums[drafter]["rounds_cnt"] += 1

    # Discover all engines
    engine_set = set()
    for drafter, s in sums.items():
        engine_set.update(s["engines"].keys())

    # Build avg maps per engine: engine -> { drafter: {avg_speed, avg_spec, avg_total} }
    avg_per_engine = {}
    for engine in sorted(engine_set):
        mapper = {}
        for drafter, s in sums.items():
            ed = s["engines"].get(engine)
            if not ed:
                continue
            avg_speed = ed["speed_sum"] / ed["speed_cnt"] if ed["speed_cnt"] > 0 else None
            avg_spec = ed["spec_sum"] / ed["spec_cnt"] if ed["spec_cnt"] > 0 else None
            avg_total = ed["total_sum"] / ed["total_cnt"] if ed["total_cnt"] > 0 else None
            # Only include drafter if it has at least a speed measurement for ordering; include None stats too.
            if avg_speed is not None:
                mapper[drafter] = {
                    "avg_speed": avg_speed,
                    "avg_spec": avg_spec,
                    "avg_total": avg_total,
                }
        if mapper:
            avg_per_engine[engine] = mapper

    # Print per-engine tables
    for engine, mapper in avg_per_engine.items():
        print(f"=== Average {engine} Speedups per Drafter ===")
        for drafter, info in sorted(mapper.items(), key=lambda x: -x[1]["avg_speed"]):
            avg_speed = info["avg_speed"]
            avg_spec = info.get("avg_spec")
            avg_total = info.get("avg_total")
            # compute verification latency if possible
            if (avg_spec is not None) and (avg_total is not None):
                avg_verify = avg_total - avg_spec
            else:
                avg_verify = None

            acc_rate_str, num_rounds_str, avg_acc_spec_str, max_acc_spec_str = _format_stats_for_drafter(
                sums, drafter
            )

            # latency strings
            spec_str = f"{avg_spec:.1f}ms" if avg_spec is not None else "N/A"
            total_str = f"{avg_total:.1f}ms" if avg_total is not None else "N/A"
            verify_str = f"{avg_verify:.1f}ms" if avg_verify is not None else "N/A"

            print(
                f"{drafter}: {avg_speed:.3f}x, acc rate {acc_rate_str}, num rounds {num_rounds_str}, "
                f"spec latency {spec_str}, verify latency {verify_str}, total latency {total_str}, "
                f"avg accepted/speculated: {avg_acc_spec_str}, "
                f"max accepted/speculated: {max_acc_spec_str}"
            )
        print()  # blank line between engine tables

    # Best configs per engine
    print("=== Best Drafter Configs ===")
    for engine, mapper in avg_per_engine.items():
        best = max(mapper.items(), key=lambda x: x[1]["avg_speed"])
        print(f"Best {engine} Drafter: {best[0]} ({best[1]['avg_speed']:.3f}x)")


if __name__ == "__main__":
    # AR drafters
    # log_file = "/data2/ruipan/diffspec/logs/2025_11_28_11_44_math.ansi"
    # log_file = "/data2/ruipan/diffspec/logs/2025_11_28_11_48_aime.ansi"
    # log_file = "/data2/ruipan/diffspec/logs/2025_11_28_11_14_gpqa.ansi"
    
    # dLLM 0.9
    # log_file = "/data2/ruipan/diffspec/logs/2025_11_28_12_47_math.ansi"
    # log_file = "/data2/ruipan/diffspec/logs/2025_11_28_12_50_aime.ansi"
    # log_file = "/data2/ruipan/diffspec/logs/2025_11_28_11_19_gpqa.ansi"
    
    # dLLM 0.05
    # log_file = "/data2/ruipan/diffspec/logs/2025_11_28_15_00_math.ansi"
    # log_file = "/data2/ruipan/diffspec/logs/2025_11_28_15_04_aime.ansi"
    
    # FailFast (vLLM: dllm_0.05_df_0.4_60_10)
    # log_file = "/data2/ruipan/diffspec/logs/2025_11_28_13_31_math.ansi"
    # log_file = "/data2/ruipan/diffspec/logs/2025_11_28_13_31_aime.ansi"
    log_file = "/data2/ruipan/diffspec/logs/2025_11_28_13_16_gpqa.ansi"
    

    data = parse_log(log_file)
    compute_averages_and_print(data)

# %%
