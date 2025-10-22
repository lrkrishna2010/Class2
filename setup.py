from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (9,4)
plt.rcParams['axes.grid'] = True

# Optional libs (Plotly & ipywidgets): detected at runtime
try:
    import plotly.express as px
    _HAVE_PLOTLY = True
except Exception:
    _HAVE_PLOTLY = False

try:
    import ipywidgets as widgets
    from IPython.display import display
    _HAVE_WIDGETS = True
except Exception:
    _HAVE_WIDGETS = False

# === Figure export helper ===
def export_fig(fig=None, name='chart', fmt='png', out_dir='reports'):
    """Save the current or provided matplotlib figure to reports/ as PNG/PDF."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    path = Path(out_dir) / f"{name}.{fmt}"
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f" Exported chart to {path}")

# === Summary stats under charts ===
def show_summary_stats(returns_series: pd.Series):
    r = returns_series.dropna()
    ann_ret = r.mean()*252
    ann_vol = r.std()*np.sqrt(252)
    sharpe  = ann_ret/(ann_vol+1e-12)
    cum = (1+r).cumprod()
    mdd = (cum/cum.cummax()-1.0).min()
    print(f" Annualized Return: {ann_ret:.2%}")
    print(f" Volatility: {ann_vol:.2%}")
    print(f" Sharpe Ratio: {sharpe:.2f}")
    print(f" Max Drawdown: {mdd:.2%}")

# === Rolling Sharpe ===
def rolling_sharpe(r: pd.Series, window=252):
    mu = r.rolling(window).mean()
    sd = r.rolling(window).std()
    return (np.sqrt(252)*mu)/(sd+1e-12)

# === Simple backtest: monthly rebalancing, transaction costs (bps) ===
def rebalance_backtest(returns: pd.DataFrame, weight_fn, rebalance_freq='M', tx_bps: float = 5.0):
    idx = returns.resample(rebalance_freq).last().index
    weights_rec, port_rets = [], []
    prev_w = None
    for i, end in enumerate(idx):
        window = returns.loc[:end]
        if window.empty: continue
        w = weight_fn(window)
        w = w/(w.sum()+1e-12)
        weights_rec.append((end, w))
        nxt = idx[i+1] if i+1 < len(idx) else returns.index[-1]
        seg = returns.loc[(returns.index> end) & (returns.index<=nxt)]
        if seg.empty: continue
        pr = seg @ w
        if prev_w is not None and tx_bps>0:
            turnover = np.abs(w - prev_w).sum()
            if len(pr)>0:
                pr.iloc[0] -= (tx_bps/10000.0)*turnover
        port_rets.append(pr)
        prev_w = w
    if not port_rets:
        raise ValueError('No backtest segments computed')
    r = pd.concat(port_rets).sort_index()
    curve = (1+r).cumprod()
    return {'returns': r, 'curve': curve, 'weights': weights_rec}

# === Mean–Variance weights via risk aversion γ (SLSQP fallback) ===
from scipy.optimize import minimize
def mv_weights_gamma(mu: np.ndarray, Sigma: np.ndarray, gamma: float = 5.0, long_only=True):
    n = len(mu)
    def obj(w): return -(w@mu - gamma*(w@Sigma@w))
    cons = ({'type':'eq','fun':lambda w: np.sum(w)-1.0},)
    bnds = [(0,1)]*n if long_only else None
    res = minimize(obj, x0=np.ones(n)/n, bounds=bnds, constraints=cons, method='SLSQP')
    return res.x

# === Risk measures (historical VaR / CVaR) ===
def var_historical(series: pd.Series, alpha=0.05):
    return -np.quantile(series.dropna(), alpha)
def cvar_empirical(series: pd.Series, alpha=0.05):
    q = np.quantile(series.dropna(), alpha)
    tail = series[series <= q]
    return -tail.mean()
