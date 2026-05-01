# %%
import math
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar


def calculate_spec_decoding_speedup(alpha, gamma, c, verbose=False):
    """Calculate the speculative decoding speedup.

    Reference: Theorem 3.8 in https://arxiv.org/pdf/2211.17192
    
    Args:
        alpha (float): Avg per-token acceptance rate, between 0 and 1.
        gamma (int): The number of drafted tokens.
        c (float): The drafter-to-target per-token latency ratio.
    """
    numerator = 1 - alpha ** (gamma + 1)
    denominator = (1 - alpha) * (c * gamma + 1)
    speedup = numerator / denominator
    if verbose:
        print(f"Spec len {gamma}, numerator {numerator}, denominator {denominator}")
    return speedup

def get_actual_acceptance_ratio(overall_ratio, spec_len):
    """This function strips away the impact of the actual draft len on acceptance rate.
    (Longer draft -> exponentially lower per-token rate -> lower acc rate on average)
    """
    # Let's say that overall_ratio is 58.2, and spec_len is 8.
    # The actual ratio we want to return is r.
    # Then, avg([r, r ** 2, ..., r ** 8]) = overall_ratio.
    
    # We need to solve: avg([r, r^2, ..., r^spec_len]) = overall_ratio
    # Which is: (r + r^2 + ... + r^spec_len) / spec_len = overall_ratio
    # The sum of geometric series: r + r^2 + ... + r^n = r * (1 - r^n) / (1 - r) for r != 1
    # So: r * (1 - r^spec_len) / (1 - r) = overall_ratio * spec_len
    
    def equation(r):
        if abs(r - 1.0) < 1e-10:
            # Special case: if r == 1, the sum is just spec_len
            return spec_len - overall_ratio * spec_len
        # Geometric series sum
        geometric_sum = r * (1 - r ** spec_len) / (1 - r)
        return geometric_sum - overall_ratio * spec_len
    
    # Find root using Brent's method (works well for bounded intervals)
    # r should be between 0 and 1 (acceptance rates are probabilities)
    try:
        result = root_scalar(equation, bracket=[0.001, 0.999], method='brentq')
        return result.root
    except ValueError:
        # If bracket doesn't work, try a wider range or different method
        result = root_scalar(equation, bracket=[0.0, 1.0], method='brentq')
        return result.root 


drafter_fwd = 6.1
target_fwd = 52.6
# remove impact of draft len on acceptance ratio
ar_alpha = get_actual_acceptance_ratio(0.582, 8)
dllm_alpha = get_actual_acceptance_ratio(0.406, 14.6)
# small_block_size = 16  # each forward pass produces 1 small block
small_block_size = 8  # each forward pass produces 1 small block
# small_block_size = 4  # each forward pass produces 1 small block

# draft_len_list = list(range(2, 30))
draft_len_list = list(range(2, 50))
ar_speedup_list = []
dllm_speedup_list = []

for draft_len in draft_len_list:
    ar_speedup = calculate_spec_decoding_speedup(ar_alpha, draft_len,
        # both models needs one fwd pass for each token
        (drafter_fwd * draft_len) / (target_fwd * draft_len))

    num_blocks_needed = (draft_len - 1) // small_block_size
    dllm_speedup = calculate_spec_decoding_speedup(dllm_alpha, draft_len, 
        # target model needs one fwd pass per token
        # dLLM drafter is sublinear (1 fwd pass generates up to 8 tokens)
        # but also semi-autoregressive, leading to the sawtooth shape
        # +2: 1 is from the one-step generation. the other is the other misc passes, e.g.,
        # tidying up the KVs. The number 2 comes from Fig. 2, where 2.2 fwd passes is required
        # in the absolute minimal compute setting.
        (drafter_fwd * (num_blocks_needed + 2)) / (target_fwd * draft_len))  # drafter does one-pass generation
    # print(f"draft len {draft_len}, dllm_speedup {dllm_speedup}")
    print(f"draft_len {draft_len}, num_blocks_needed {num_blocks_needed}, dllm_speedup {dllm_speedup}")
    ar_speedup_list.append(ar_speedup)
    dllm_speedup_list.append(dllm_speedup)

fig, ax = plt.subplots(figsize=(4.5, 2.5))
# fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(draft_len_list, ar_speedup_list, color='#74C69D', marker='D', markersize=3.5, label='AR Drafter')
ax.plot(draft_len_list, dllm_speedup_list, color='#2D6A4F', marker='o', markersize=3.5, label='dLLM Drafter')
ax.legend(fontsize=12, markerscale=1.75)
ax.set_xticks([0, 10, 20, 30, 40, 50])
ax.set_yticks([1, 2, 3, 4, 5])
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_ylabel("Theoretical Speedup ($\\times$)", fontsize=12)
ax.set_xlabel("Speculation Length per Round", fontsize=12)
ax.yaxis.set_label_coords(-0.08, 0.4)  # x, y

# Styling
ax.grid(True, linestyle='--', alpha=0.7)
fig.tight_layout()
plt.show()
# fig.savefig(f"../figures/theoretical_analysis.pdf", dpi=500, bbox_inches='tight')
# %%
