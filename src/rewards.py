import numpy as np

def reward_function(S_t, A_t, beta_s, beta_a, noise_scale = 0.1):
    #S_t @ beta_s is state effect, beta_a[A_t] is effect of action, indexed by A_t
    linear_part = S_t @ beta_s + beta_a[A_t]
    noise = np.random.normal(0, noise_scale, len(S_t)) #stochastic noise
    R_t = linear_part + noise
    return R_t