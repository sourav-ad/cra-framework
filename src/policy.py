import numpy as np
from sklearn.linear_model import LogisticRegression

seed = 123

#Baseline multinomial logistic policy \pi_0(A | S_t)
# \frac{\exp(alpha + beta^T S_t)}{\sum {\exp(alpha + beta^T S_t)}}

def generate_policy_parameters(p, k, seed = None):
    rng = np.random.default_rng(seed)
    alphas = rng.uniform(-0.5, 0.5, size = k)
    betas = rng.uniform(-1, 1, size = (k, p))
    return alphas, betas

#discrete actions based on a multinomial logit policy:
# P(A_t = a | Z_t) ∝ exp(alpha_a + Z_t beta_a^T)

# Parameters
# Z_t : (n, p) normalized state matrix
# alphas : (k,) action intercepts
# betas : (k, p) action coefficients
# Output:
# actions : (n,) chosen action indices [0..k-1]
# probs   : (n, k) action probability distribution
# A_onehot: (n, k) one-hot encoded actions

#create the hypothetical policy \pi_0
def sample_action(Z_t, alphas, betas, seed = None):
    #multinomial logit expression
    rng = np.random.default_rng(seed)
    logits = alphas + Z_t @ betas.T
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    actions = np.array([rng.choice(len(probs[i]), p=probs[i]) for i in range(len(probs))])
    A_onehot = np.eye(len(alphas))[actions] #one hot encoding the action matrix
    return actions, probs, A_onehot