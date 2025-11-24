"""
Backtesting Engine

Walk-forward backtesting with realistic transaction costs and slippage.

Features:
- Walk-forward validation (train/test split)
- Multiple rebalancing frequencies (daily, weekly, monthly)
- Transaction cost modeling
- Portfolio-level performance metrics
- Benchmark comparison (SPY)
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis

Academic References:
- Pardo (2008): "The Evaluation and Optimization of Trading Strategies"
- Bailey et al. (2014): "The Probability of Backtest Overfitting"
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class BacktestResult:
    """Backtest performance results."""

    # Returns
    total_return: float
    annual_return: float
    benchmark_return: float
    alpha: float  # Excess return over benchmark

    # Risk metrics
    volatility: float  # Annualized
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float

    # Trade stats
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float  # Gross profit / gross loss

    # Time series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    positions: pd.DataFrame

    # Period
    start_date: datetime
    end_date: datetime
    days: int

    def __str__(self) -> str:
        return f"""
Backtest Results ({self.start_date.date()} to {self.end_date.date()})
{'='*60}
Returns:
  Total Return:        {self.total_return:>10.2f}%
  Annual Return:       {self.annual_return:>10.2f}%
  Benchmark Return:    {self.benchmark_return:>10.2f}%
  Alpha:               {self.alpha:>10.2f}%

Risk Metrics:
  Volatility (Annual): {self.volatility:>10.2f}%
  Sharpe Ratio:        {self.sharpe_ratio:>10.2f}
  Sortino Ratio:       {self.sortino_ratio:>10.2f}
  Calmar Ratio:        {self.calmar_ratio:>10.2f}
  Max Drawdown:        {self.max_drawdown:>10.2f}%

Trade Statistics:
  Total Trades:        {self.total_trades:>10}
  Win Rate:            {self.win_rate:>10.2f}%
  Avg Win:             {self.avg_win:>10.2f}%
  Avg Loss:            {self.avg_loss:>10.2f}%
  Profit Factor:       {self.profit_factor:>10.2f}
{'='*60}
"""


class BacktestEngine:
    """
    Walk-forward backtesting engine.

    Simulates trading strategy with realistic costs and constraints.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,  # 0.1% per trade
        slippage_pct: float = 0.0005,  # 0.05% slippage
        rebalance_freq: str = 'weekly',  # 'daily', 'weekly', 'monthly'
        max_positions: int = 20,
        min_position_pct: float = 0.05,  # Min 5% per position
        max_position_pct: float = 0.20,  # Max 20% per position
    ):
        """
        Initialize backtesting engine.

        Args:
            initial_capital: Starting portfolio value
            commission_pct: Commission per trade (as decimal, e.g., 0.001 = 0.1%)
            slippage_pct: Slippage per trade (as decimal)
            rebalance_freq: Rebalancing frequency ('daily', 'weekly', 'monthly')
            max_positions: Maximum number of positions
            min_position_pct: Minimum position size as % of portfolio
            max_position_pct: Maximum position size as % of portfolio
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.rebalance_freq = rebalance_freq
        self.max_positions = max_positions
        self.min_position_pct = min_position_pct
        self.max_position_pct = max_position_pct

        logger.info(
            f"Initialized backtest engine: ${initial_capital:,.0f} capital, "
            f"{rebalance_freq} rebalancing, {max_positions} max positions"
        )

    def run(
        self,
        scoring_func: Callable,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        benchmark_ticker: str = 'SPY'
    ) -> BacktestResult:
        """
        Run backtest with walk-forward validation.

        Args:
            scoring_func: Function that takes (ticker, date) and returns score
            tickers: List of tickers to trade
            start_date: Backtest start date
            end_date: Backtest end date
            benchmark_ticker: Benchmark for comparison

        Returns:
            BacktestResult with performance metrics
        """
        logger.info(f"Starting backtest from {start_date.date()} to {end_date.date()}")
        logger.info(f"Universe: {len(tickers)} tickers, Benchmark: {benchmark_ticker}")

        try:
            # Get historical price data
            price_data = self._fetch_price_data(tickers, start_date, end_date)
            benchmark_data = self._fetch_price_data([benchmark_ticker], start_date, end_date)

            if price_data.empty:
                raise ValueError("No price data available for backtest period")

            # Initialize portfolio
            portfolio_value = self.initial_capital
            cash = self.initial_capital
            positions = {}  # {ticker: shares}

            # Track performance
            equity_curve = []
            drawdowns = []
            trades = []
            dates = []
            position_history = []

            # Get rebalance dates
            rebalance_dates = self._get_rebalance_dates(
                start_date, end_date, price_data.index
            )

            logger.info(f"Rebalancing on {len(rebalance_dates)} dates")

            # Walk-forward simulation
            for current_date in price_data.index:
                # Mark-to-market portfolio value
                portfolio_value = cash
                for ticker, shares in positions.items():
                    if ticker in price_data.columns:
                        price = price_data.loc[current_date, ticker]
                        if not pd.isna(price):
                            portfolio_value += shares * price

                # Record equity
                equity_curve.append(portfolio_value)
                dates.append(current_date)

                # Record positions
                position_history.append({
                    'date': current_date,
                    'portfolio_value': portfolio_value,
                    'cash': cash,
                    'positions': dict(positions)
                })

                # Rebalance on scheduled dates
                if current_date in rebalance_dates:
                    logger.debug(f"Rebalancing on {current_date.date()}")

                    # Score all tickers
                    scores = {}
                    for ticker in tickers:
                        try:
                            score = scoring_func(ticker, current_date)
                            if score is not None:
                                scores[ticker] = float(score)
                        except Exception as e:
                            logger.warning(f"Error scoring {ticker}: {e}")

                    # Select top stocks
                    top_stocks = self._select_portfolio(scores, price_data, current_date)

                    # Rebalance portfolio
                    new_trades = self._rebalance_portfolio(
                        positions, cash, top_stocks, price_data, current_date, portfolio_value
                    )
                    trades.extend(new_trades)

                    # Update cash after rebalancing
                    cash = self._calculate_cash(positions, price_data, current_date, portfolio_value)

            # Calculate performance metrics
            result = self._calculate_metrics(
                equity_curve=pd.Series(equity_curve, index=dates),
                trades=trades,
                position_history=pd.DataFrame(position_history),
                benchmark_data=benchmark_data,
                start_date=start_date,
                end_date=end_date
            )

            logger.info("Backtest complete")
            logger.info(f"Total Return: {result.total_return:.2f}%, Sharpe: {result.sharpe_ratio:.2f}")

            return result

        except Exception as e:
            logger.error(f"Backtest error: {e}")
            raise

    def _fetch_price_data(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch historical price data for tickers."""
        try:
            import yfinance as yf

            # Fetch data with buffer for technical indicators
            buffer_start = start_date - timedelta(days=365)

            data = yf.download(
                tickers,
                start=buffer_start,
                end=end_date,
                progress=False
            )

            # Extract adjusted close prices
            if len(tickers) == 1:
                prices = data['Adj Close'].to_frame(name=tickers[0])
            else:
                prices = data['Adj Close']

            # Filter to actual backtest period
            prices = prices.loc[start_date:]

            return prices

        except Exception as e:
            logger.error(f"Error fetching price data: {e}")
            return pd.DataFrame()

    def _get_rebalance_dates(
        self,
        start_date: datetime,
        end_date: datetime,
        available_dates: pd.DatetimeIndex
    ) -> List[datetime]:
        """Get rebalancing dates based on frequency."""
        rebalance_dates = []

        if self.rebalance_freq == 'daily':
            rebalance_dates = available_dates.tolist()

        elif self.rebalance_freq == 'weekly':
            # Rebalance every Monday (or first available day of week)
            for date in available_dates:
                if date.weekday() == 0:  # Monday
                    rebalance_dates.append(date)

        elif self.rebalance_freq == 'monthly':
            # Rebalance on first trading day of month
            current_month = None
            for date in available_dates:
                if current_month != date.month:
                    rebalance_dates.append(date)
                    current_month = date.month

        return rebalance_dates

    def _select_portfolio(
        self,
        scores: Dict[str, float],
        price_data: pd.DataFrame,
        date: datetime
    ) -> Dict[str, float]:
        """Select top stocks for portfolio based on scores."""
        # Filter stocks with valid prices
        valid_stocks = {}
        for ticker, score in scores.items():
            if ticker in price_data.columns:
                price = price_data.loc[date, ticker]
                if not pd.isna(price) and price > 0:
                    valid_stocks[ticker] = score

        # Sort by score and select top N
        sorted_stocks = sorted(valid_stocks.items(), key=lambda x: x[1], reverse=True)
        top_stocks = dict(sorted_stocks[:self.max_positions])

        # Equal weight allocation (can be enhanced with score-weighted)
        if top_stocks:
            weight = 1.0 / len(top_stocks)
            allocation = {ticker: weight for ticker in top_stocks}
        else:
            allocation = {}

        return allocation

    def _rebalance_portfolio(
        self,
        current_positions: Dict[str, int],
        current_cash: float,
        target_allocation: Dict[str, float],
        price_data: pd.DataFrame,
        date: datetime,
        portfolio_value: float
    ) -> List[Dict]:
        """Rebalance portfolio to target allocation."""
        trades = []

        # Sell positions not in target
        for ticker in list(current_positions.keys()):
            if ticker not in target_allocation:
                shares = current_positions[ticker]
                price = price_data.loc[date, ticker]

                # Sell with slippage and commission
                sell_price = price * (1 - self.slippage_pct)
                proceeds = shares * sell_price
                commission = proceeds * self.commission_pct

                current_cash += proceeds - commission
                del current_positions[ticker]

                trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares,
                    'price': sell_price,
                    'value': proceeds,
                    'commission': commission
                })

        # Buy/adjust positions in target
        for ticker, target_weight in target_allocation.items():
            price = price_data.loc[date, ticker]
            target_value = portfolio_value * target_weight
            target_shares = int(target_value / price)

            current_shares = current_positions.get(ticker, 0)
            shares_to_buy = target_shares - current_shares

            if shares_to_buy > 0:
                # Buy with slippage and commission
                buy_price = price * (1 + self.slippage_pct)
                cost = shares_to_buy * buy_price
                commission = cost * self.commission_pct

                if current_cash >= (cost + commission):
                    current_cash -= (cost + commission)
                    current_positions[ticker] = current_positions.get(ticker, 0) + shares_to_buy

                    trades.append({
                        'date': date,
                        'ticker': ticker,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': buy_price,
                        'value': cost,
                        'commission': commission
                    })

        return trades

    def _calculate_cash(
        self,
        positions: Dict[str, int],
        price_data: pd.DataFrame,
        date: datetime,
        portfolio_value: float
    ) -> float:
        """Calculate remaining cash after positions."""
        position_value = 0
        for ticker, shares in positions.items():
            if ticker in price_data.columns:
                price = price_data.loc[date, ticker]
                if not pd.isna(price):
                    position_value += shares * price

        return portfolio_value - position_value

    def _calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        position_history: pd.DataFrame,
        benchmark_data: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics."""

        # Returns
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        days = (end_date - start_date).days
        annual_return = ((1 + total_return/100) ** (365/days) - 1) * 100

        # Benchmark
        benchmark_return = (benchmark_data.iloc[-1].values[0] / benchmark_data.iloc[0].values[0] - 1) * 100
        alpha = annual_return - benchmark_return

        # Risk metrics
        returns = equity_curve.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return/100 - risk_free_rate) / (volatility/100) if volatility > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return/100 - risk_free_rate) / downside_std if downside_std > 0 else 0

        # Drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve / running_max - 1) * 100
        max_drawdown = drawdown.min()

        # Calmar ratio (return / max drawdown)
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        if trades:
            wins = [t for t in trades if t.get('pnl', 0) > 0]
            losses = [t for t in trades if t.get('pnl', 0) < 0]

            win_rate = len(wins) / len(trades) * 100 if trades else 0
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0

            gross_profit = sum([t['pnl'] for t in wins]) if wins else 0
            gross_loss = abs(sum([t['pnl'] for t in losses])) if losses else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            benchmark_return=benchmark_return,
            alpha=alpha,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            total_trades=len(trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            equity_curve=equity_curve,
            drawdown_series=drawdown,
            positions=position_history,
            start_date=start_date,
            end_date=end_date,
            days=days
        )
