import numpy as np
from sklearn.linear_model import LogisticRegression

seed = 123

#Baseline multinomial logistic policy \pi_0(A | S_t)
# \frac{\exp(alpha + beta^T S_t)}{\sum {\exp(alpha + beta^T S_t)}}

def generate_policy_parameters(p, k, seed = seed):
    np.random.seed(seed)
    alphas = np.random.uniform(-0.5, 0.5, size = k)
    betas = np.random.uniform(-1, 1, size = (k, p))
    return alphas, betas


def sample_action(S_t, alphas, betas):
    #multinomial logit expression
    logits = alphas + S_t @ betas.T
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    actions = np.array([np.random.choice(len(probs[i]), p=probs[i]) for i in range(len(probs))])
    return actions, probs