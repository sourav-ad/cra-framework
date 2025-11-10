import numpy as np
import pandas as pd
from dataclasses import dataclass

seed = 123
rng = np.random.default_rng(seed=seed)

#Vitals with a badness score to make rewards make sense

# Define mean and SD for each vital
stats = {
    "weight": (80, 12),
    "hr": (72, 8),
    "bp": (124, 13),
    "hb1ac": (6.2, 1.0)
}

def initialize_states(n, seed=seed):
    rng = np.random.default_rng(seed)
    X = np.column_stack([
        rng.normal(mu, sd, n) for mu, sd in stats.values()
    ])
    return X  # shape (n, p)

# Standardize each vital metric to the [-1, 1] 'badness' scale
# using 3*sd normalization: z = (x - mu)/(3*sd)

def standardize_states(S):
    mus = np.array([v[0] for v in stats.values()])
    sds = np.array([v[1] for v in stats.values()])
    Z = (S - mus) / (3 * sds)
    return Z

# Revert normalized scores back to real-world vitals, as and when needed
# But, DO NOT transition actual values, ALWAYS transit standardized metrics

def destandardize_states(Z):
    mus = np.array([v[0] for v in stats.values()])
    sds = np.array([v[1] for v in stats.values()])
    S = 3 * sds * Z + mus
    return S


# Transition on normalized scale:
# Z_{t+1} = Z_t * phi + A_t * psi^T + eta
# where:
#     Z_t: (n, p)
#     A_t: (n, k) one-hot encoded
#     phi: (p, p) transition matrix, time invariant
#     psi: (p, k) action-effect matrix, patient invariant (for ease of use)
    

def transition_function(Z_t, A_t, phi, psi, noise_scale=0.02, seed=None):
    rng = np.random.default_rng(seed)
    n, p = Z_t.shape
    _, k = A_t.shape

    eta = rng.normal(0, noise_scale, size=(n, p))
    Z_next = Z_t @ phi + A_t @ psi.T + eta
    return Z_next

