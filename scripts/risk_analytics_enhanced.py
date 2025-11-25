"""
Advanced Risk Analytics Module.

Implements professional-grade risk metrics:
- Value at Risk (VaR)
- Conditional VaR (CVaR / Expected Shortfall)
- Omega Ratio
- Sortino Ratio
- Calmar Ratio
- Maximum Drawdown
- Downside Deviation
- Skewness & Kurtosis

These provide deeper insight into portfolio risk than standard deviation alone.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats


class RiskAnalytics:
    """
    Advanced risk metrics for portfolio analysis.
    """

    @staticmethod
    def value_at_risk(
        returns: pd.Series,
        confidence_level: float = 0.95,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        VaR answers: "What is the maximum loss we can expect with X% confidence?"

        Args:
            returns: Series of returns
            confidence_level: Confidence level (default: 95%)
            method: 'historical' or 'parametric'

        Returns:
            VaR as a positive number (e.g., 0.05 = 5% loss)
        """
        if method == 'historical':
            # Historical VaR: percentile of empirical distribution
            var = -np.percentile(returns, (1 - confidence_level) * 100)
        else:
            # Parametric VaR: assume normal distribution
            mean = returns.mean()
            std = returns.std()
            var = -(mean + std * stats.norm.ppf(1 - confidence_level))

        return float(max(var, 0))

    @staticmethod
    def conditional_var(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Conditional VaR (CVaR), also called Expected Shortfall.

        CVaR answers: "Given that we exceed VaR, what's the expected loss?"

        This is better than VaR because it accounts for tail risk.

        Args:
            returns: Series of returns
            confidence_level: Confidence level

        Returns:
            CVaR as a positive number
        """
        var = RiskAnalytics.value_at_risk(returns, confidence_level, 'historical')
        # CVaR is the mean of all returns worse than VaR
        losses = returns[returns < -var]
        if len(losses) > 0:
            cvar = -losses.mean()
        else:
            cvar = var

        return float(max(cvar, 0))

    @staticmethod
    def omega_ratio(
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate Omega Ratio.

        Omega is the probability-weighted ratio of gains vs losses.
        Superior to Sharpe because it uses the full return distribution.

        Values > 1.0 are good, > 1.5 is excellent.

        Args:
            returns: Series of returns
            threshold: Minimum acceptable return (default: 0)

        Returns:
            Omega ratio
        """
        excess_returns = returns - threshold
        gains = excess_returns[excess_returns > 0].sum()
        losses = -excess_returns[excess_returns < 0].sum()

        if losses == 0:
            return np.inf if gains > 0 else 1.0

        return gains / losses

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino Ratio.

        Like Sharpe ratio, but only penalizes downside volatility.
        Better for asymmetric return distributions.

        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annual)
            periods_per_year: Trading days per year

        Returns:
            Sortino ratio
        """
        excess_return = returns.mean() - risk_free_rate / periods_per_year
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return np.inf

        downside_std = downside_returns.std()

        if downside_std == 0:
            return np.inf

        # Annualize
        sortino = (excess_return * np.sqrt(periods_per_year)) / (downside_std * np.sqrt(periods_per_year))

        return sortino

    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar Ratio.

        Ratio of annualized return to maximum drawdown.
        Good for comparing strategies with different risk profiles.

        Values > 0.5 are good, > 1.0 is excellent.

        Args:
            returns: Series of returns
            periods_per_year: Trading days per year

        Returns:
            Calmar ratio
        """
        annual_return = returns.mean() * periods_per_year
        max_dd = RiskAnalytics.maximum_drawdown(returns)

        if max_dd == 0:
            return np.inf if annual_return > 0 else 0.0

        return annual_return / abs(max_dd)

    @staticmethod
    def maximum_drawdown(returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            returns: Series of returns

        Returns:
            Maximum drawdown as negative number (e.g., -0.35 = -35%)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        return drawdown.min()

    @staticmethod
    def downside_deviation(
        returns: pd.Series,
        threshold: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate downside deviation.

        Only considers returns below threshold.
        Better than standard deviation for asymmetric distributions.

        Args:
            returns: Series of returns
            threshold: Minimum acceptable return
            periods_per_year: Trading days per year

        Returns:
            Annualized downside deviation
        """
        downside_returns = returns[returns < threshold]

        if len(downside_returns) == 0:
            return 0.0

        downside_std = downside_returns.std() * np.sqrt(periods_per_year)

        return downside_std

    @staticmethod
    def calculate_all_metrics(
        returns: pd.Series,
        risk_free_rate: float = 0.04,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Calculate all risk metrics at once.

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            confidence_level: For VaR/CVaR

        Returns:
            Dictionary of all metrics
        """
        metrics = {
            # Basic stats
            'mean_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),

            # Risk metrics
            'var_95': RiskAnalytics.value_at_risk(returns, confidence_level),
            'cvar_95': RiskAnalytics.conditional_var(returns, confidence_level),
            'max_drawdown': RiskAnalytics.maximum_drawdown(returns),
            'downside_deviation': RiskAnalytics.downside_deviation(returns, 0.0),

            # Risk-adjusted returns
            'sharpe_ratio': (returns.mean() * 252 - risk_free_rate) / (returns.std() * np.sqrt(252)),
            'sortino_ratio': RiskAnalytics.sortino_ratio(returns, risk_free_rate),
            'calmar_ratio': RiskAnalytics.calmar_ratio(returns),
            'omega_ratio': RiskAnalytics.omega_ratio(returns, risk_free_rate/252)
        }

        return metrics

    @staticmethod
    def stress_test(
        returns: pd.Series,
        scenarios: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Stress test portfolio under various scenarios.

        Args:
            returns: Historical returns
            scenarios: Dict of scenario names to market shocks (e.g., {'2008 Crisis': -0.50})

        Returns:
            Dict of {scenario: expected_loss}
        """
        results = {}

        # Estimate beta to market (rough approximation)
        beta = 1.0  # Assume market beta for simplicity

        for name, market_shock in scenarios.items():
            # Simple linear model: portfolio shock ~= beta * market shock
            portfolio_shock = beta * market_shock
            results[name] = portfolio_shock

        return results
