
# Quant Risk Framework  — Educational Edition

This package contains a ready-to-run educational notebook and supporting modules for **quantitative risk and portfolio optimization**.

## Structure
```
quant_risk_framework/
├── notebooks/
│   └── Week1_Interactive.ipynb
├── src/
│   ├── risk_models.py
│   ├── optimization.py
│   ├── backtest_utils.py
│   └── reporting.py
├── config.yaml
├── reports/
├── data/
└── logs/
```

## Features
- Live data via Yahoo Finance (with synthetic fallback)
- VaR, CVaR, Sharpe, Drawdown, Rolling metrics
- Mean–Variance optimization with risk aversion γ
- Interactive controls (VaR α slider, γ slider)
- Exportable charts and summary statistics


Run the notebook in JupyterLab or VSCode with interactive widgets enabled.
