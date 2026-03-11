![visitors](https://visitor-badge.laobi.icu/badge?page_id=username.repo)

# Time Series Engine: Modeling + Simulation

A small library for time-series diagnostics, AR modeling, and Monte Carlo simulation of stochastic processes.
Designed to demonstrate statistical rigor and software engineering structure.

## Features
- Stationarity diagnostics (ADF test via statsmodels if available; fallback heuristics)
- AR(p) model estimation (Yule-Walker + OLS baseline)
- Forecasting + evaluation (MAE, RMSE)
- Monte Carlo simulation:
  - Random Walk
  - Geometric Brownian Motion (GBM)

## Installation
Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
