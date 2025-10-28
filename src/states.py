import numpy as np
import pandas as pd

seed = 123

#initialize state S0 using standard normal
def initialize_states(n, p, seed = seed):
    np.random.seed(seed)
    S0 = np.random.normal(0, 1, size=(n, p))
    return S0

#transition function
def transition_function(S_t, A_t, noise_scale=0.1):
    """
    first order Gaussian state space model, simple controllable markov chain 
    next state = 0.7*S_t + action effect + noise, for mean reverting, stable trajectories
    """
    n, p = S_t.shape
    A_effect = np.expand_dims(A_t - np.mean(A_t), 1) * np.random.uniform(0.1, 0.3, size=(1, p)) #ensures net action effect = 0
    noise = np.random.normal(0, noise_scale, size=(n, p))
    S_next = 0.7 * S_t + A_effect + noise
    return S_next