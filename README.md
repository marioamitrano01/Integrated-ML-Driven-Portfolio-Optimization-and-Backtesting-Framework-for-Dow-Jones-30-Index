# Dow Jones Machine Learning Portfolio

This is a machine learning-driven framework for portfolio optimization and backtesting using Dow Jones 30 Index. It combines Monte Carlo simulation and deep learning techniques to generate, optimize, and evaluate portfolio allocations.

## Techniques Utilized
- **Monte Carlo Simulation:** Generate and assess portfolio weights to optimize a risk-adjusted objective.
- **Deep Learning:** Utilize Bidirectional LSTM layers with a custom Attention mechanism to predict portfolio weights.
- **Backtesting:** Compare the ML-optimized portfolio against the Dow Jones 30 Index using historical data, calculating key performance metrics like cumulative returns, volatility, Sharpe Ratio, and drawdowns.

## Required Libraries
- **Data Processing:** `numpy`, `pandas`
- **Financial Data:** `yfinance`
- **Visualization:** `plotly`
- **Machine Learning:** `tensorflow` (with Keras API)
- **Utilities:** `os`, `datetime`

