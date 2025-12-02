import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

#propensity score model
def fit_propensity(df, features, action_col="A"):
    X = df[features].values
    y = df[action_col].values
    model = LogisticRegression(multi_class="multinomial", max_iter=1000)
    model.fit(X, y)
    e_hat = model.predict_proba(X)
    return model, e_hat

#outcome regression model
def fit_outcome(df, features, action_col="A", reward_col="R", k=3):
    X = df[features].values
    y = df[reward_col].values
    A = df[action_col].values
    m_hat = np.zeros((len(df), k))
    for a in range(k):
        mask = (A == a)
        if mask.sum() > 1:  # ensure at least 2 samples
            reg = LinearRegression()
            reg.fit(X[mask], y[mask])
            m_hat[:, a] = reg.predict(X)
        else:
            # fallback: use mean reward if insufficient samples for action
            m_hat[:, a] = y.mean()
    return m_hat

#AIPW estimator
def cra_estimator(df, features_outcome, features_propensity, k=3):

    # Propensity
    X_e = df[features_propensity].values
    y   = df["A"].values
    model_e = LogisticRegression(multi_class="multinomial", max_iter=1000)
    model_e.fit(X_e, y)
    e_hat = model_e.predict_proba(X_e)

    # Outcome
    X_m = df[features_outcome].values
    R   = df["R"].values
    A   = df["A"].values

    m_hat = np.zeros((len(df), k))
    for a in range(k):
        reg = LinearRegression()
        reg.fit(X_m[A == a], R[A == a])
        m_hat[:, a] = reg.predict(X_m)

    # AIPW
    psi = np.zeros((len(df), k))
    for a in range(k):
        psi[:, a] = m_hat[:, a] + ((A == a) / e_hat[:, a]) * (R - m_hat[:, a])

    theta_hat = psi.mean(axis=0)
    return theta_hat, psi


#Time indexed, AIPW estimator independently at each time step
# def cra_timewise(df, features, T=3, k=3):
#     results = {}
#     for t in range(T):
#         df_t = df[df["t"] == t]
#         theta_t, _ = cra_estimator(df_t, features, k=k)
#         results[t] = theta_t
#     return results