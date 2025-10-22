"""
backtest_utils.py — Functions for simple rolling-window backtesting.
"""
import pandas as pd, numpy as np

def rebalance_backtest(returns: pd.DataFrame, weight_fn, rebalance_freq='M', tx_bps=5.0):
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
            if len(pr)>0: pr.iloc[0] -= (tx_bps/10000.0)*turnover
        port_rets.append(pr)
        prev_w = w
    if not port_rets: raise ValueError('No backtest segments computed')
    r = pd.concat(port_rets).sort_index()
    return {'returns': r, 'curve': (1+r).cumprod(), 'weights': weights_rec}
