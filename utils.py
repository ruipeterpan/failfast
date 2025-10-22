


# %%
def calculate_spec_decoding_speedup(alpha, gamma, c):
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
    return speedup
# %%
