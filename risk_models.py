"""
risk_models.py — Basic statistical and risk metrics for portfolio returns.
""" 
import numpy as np, pandas as pd

def var_historical(series: pd.Series, alpha=0.05):
    """Compute historical Value-at-Risk (VaR) at confidence level alpha."""
    return -np.quantile(series.dropna(), alpha)

def cvar_empirical(series: pd.Series, alpha=0.05):
    """Conditional VaR (Expected Shortfall): mean loss beyond the VaR threshold."""
    q = np.quantile(series.dropna(), alpha)
    tail = series[series <= q]
    return -tail.mean()

def rolling_volatility(returns: pd.Series, window=252):
    return returns.rolling(window).std()*np.sqrt(252)
