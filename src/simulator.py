import numpy as np
import pandas as pd
from src.states import initialize_states, transition_function, standardize_states
from src.policy import generate_policy_parameters, sample_action
from src.rewards import reward_function

seed = 123

def generate_data(n=500, T=3, p=4, k=3, seed=None):
    rng = np.random.default_rng(seed)

    # policy parameters
    alphas, betas = generate_policy_parameters(p, k, seed)
    
    # transition matrices
    phi = np.eye(p) * rng.uniform(0.9, 1.1, size=p) + rng.uniform(-0.05, 0.05, size=(p, p))
    psi = rng.uniform(-0.1, 0.1, size=(p, k))
    
    # Vital importance weights for reward
    w = rng.uniform(0.1, 1.0, size=p)
    w = w / w.sum()  # normalize to sum=1

    # initialize and standardize states
    S_0 = initialize_states(n, seed)
    Z_t = standardize_states(S_0)

    records = []

    for t in range(T):
        #sample actions based on policy
        actions, probs, A_onehot = sample_action(Z_t, alphas, betas, seed)
        
        # transition to next state 
        Z_next = transition_function(Z_t, A_onehot, phi, psi, noise_scale=0.02, seed=seed)
        
        # reward
        R_t = reward_function(Z_t, Z_next, w, noise_scale=0.01, seed=seed)

        #Record the data
        for i in range(n):
            row = [i, t, *Z_t[i], actions[i], R_t[i]] #each row in data
            records.append(row)
        
        #update state for next time step
        Z_t = Z_next

    cols = ["id", "t"] + [f"Z{j+1}" for j in range(p)] + ["A", "R"]
    df = pd.DataFrame(records, columns=cols)
    return df