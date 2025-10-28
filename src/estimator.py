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
        reg = LinearRegression()
        reg.fit(X[mask], y[mask])
        m_hat[:, a] = reg.predict(X)
    return m_hat

#AIPW estimator
def cra_estimator(df, features, k=3):
    _, e_hat = fit_propensity(df, features)
    m_hat = fit_outcome(df, features, k=k)
    A = df["A"].values
    R = df["R"].values
    psi = np.zeros((len(df), k))
    for a in range(k):
        psi[:, a] = m_hat[:, a] + ( (A == a) / e_hat[:, a] ) * (R - m_hat[:, a])
    theta_hat = psi.mean(axis=0)
    return theta_hat, psi

#Time indexed, AIPW estimator independently at each time step
def cra_timewise(df, features, T=3, k=3):
    results = {}
    for t in range(T):
        df_t = df[df["t"] == t]
        theta_t, _ = cra_estimator(df_t, features, k=k)
        results[t] = theta_t
    return results