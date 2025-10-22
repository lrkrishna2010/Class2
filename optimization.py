"""
optimization.py — Portfolio construction and optimization utilities.
"""
import numpy as np
from scipy.optimize import minimize

def mv_weights_gamma(mu: np.ndarray, Sigma: np.ndarray, gamma=5.0, long_only=True):
    """Mean–Variance optimization: maximize mu^T w - gamma * w^T Sigma w."""
    n = len(mu)
    def obj(w): return -(w@mu - gamma*(w@Sigma@w))
    cons = ({'type':'eq','fun':lambda w: np.sum(w)-1.0},)
    bnds = [(0,1)]*n if long_only else None
    res = minimize(obj, x0=np.ones(n)/n, bounds=bnds, constraints=cons, method='SLSQP')
    return res.x
