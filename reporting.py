"""
reporting.py — Plotting, exporting, and summary tools.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def export_fig(fig=None, name='chart', fmt='png', out_dir='reports'):
    """Save matplotlib figure to /reports."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if fig is None: fig = plt.gcf()
    path = Path(out_dir)/f"{name}.{fmt}"
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Exported chart to {path}")

def show_summary_stats(returns_series: pd.Series):
    """Display quick performance summary under plots."""
    r = returns_series.dropna()
    ann_ret = r.mean()*252
    ann_vol = r.std()*np.sqrt(252)
    sharpe = ann_ret/(ann_vol+1e-12)
    cum = (1+r).cumprod()
    mdd = (cum/cum.cummax()-1.0).min()
    print(f" Ann. Return: {ann_ret:.2%}\n📉 Vol: {ann_vol:.2%}\n Sharpe: {sharpe:.2f}\n MaxDD: {mdd:.2%}")
