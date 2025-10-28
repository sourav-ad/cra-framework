import numpy as np
import pandas as pd
from src.states import initialize_states, transition_function
from src.policy import generate_policy_parameters, sample_action
from src.rewards import reward_function

seed = 123

def generate_data(n=500, T=3, p=4, k=3, seed = seed):
    np.random.seed(seed)
    alphas, betas = generate_policy_parameters(p, k, seed)
    beta_s = np.random.uniform(-0.5, 0.5, size=p)
    beta_a = np.random.uniform(-1, 1, size=k)

    S_t = initialize_states(n, p, seed)
    records = []

    for t in range(T):
        A_t, _ = sample_action(S_t, alphas, betas)
        R_t = reward_function(S_t, A_t, beta_s, beta_a)
        for i in range(n):
            row = [i, t, *S_t[i], A_t[i], R_t[i]]
            records.append(row)
        S_t = transition_function(S_t, A_t)

    cols = ["id", "t"] + [f"S{j+1}" for j in range(p)] + ["A", "R"]
    df = pd.DataFrame(records, columns=cols)
    return df