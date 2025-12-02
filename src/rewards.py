import numpy as np

# Reward based on improvement in normalized vitals.
# Positive reward -> vitals move closer to 0 (healthier).
# Negative reward -> vitals drift away from 0. 3 sd away = most unhealthy


def reward_function(Z_t, Z_next, w, noise_scale=0.0005, seed=None):
    rng = np.random.default_rng(seed)
    delta = np.abs(Z_t) - np.abs(Z_next)   #improvement toward zero
    R = delta @ w + rng.normal(0, noise_scale, len(Z_t))
    return R