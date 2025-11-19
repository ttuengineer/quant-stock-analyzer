"""
Backtesting module for strategy validation.

Includes:
- Walk-forward validation
- Performance metrics (Sharpe, Sortino, max drawdown)
- Transaction cost modeling
- Multiple rebalancing frequencies
- Portfolio-level backtesting
"""

from .engine import BacktestEngine, BacktestResult

__all__ = ["BacktestEngine", "BacktestResult"]
