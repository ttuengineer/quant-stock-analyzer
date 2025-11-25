"""
Statistical Validation Module.

Provides rigorous statistical tests to validate strategy performance:
- t-tests for significance
- Monte Carlo simulation
- Permutation tests
- Sharpe ratio significance
- Regime stability analysis
- Out-of-sample decay

Answers: "Is this real alpha or just luck?"
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class StatisticalValidator:
    """
    Rigorous statistical validation of trading strategies.
    """

    @staticmethod
    def t_test_excess_returns(
        excess_returns: pd.Series,
        null_hypothesis_mean: float = 0.0
    ) -> Dict:
        """
        T-test: Are excess returns significantly different from zero?

        Args:
            excess_returns: Series of excess returns (strategy - benchmark)
            null_hypothesis_mean: H0 mean (default: 0)

        Returns:
            Dict with t_statistic, p_value, significant (bool)
        """
        t_stat, p_value = stats.ttest_1samp(excess_returns, null_hypothesis_mean)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_5pct': p_value < 0.05,
            'significant_1pct': p_value < 0.01,
            'interpretation': f"{'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at 5% level"
        }

    @staticmethod
    def monte_carlo_simulation(
        returns: pd.Series,
        n_simulations: int = 10000,
        n_periods: int = None
    ) -> Dict:
        """
        Monte Carlo simulation of returns.

        Simulates many possible future paths to understand distribution
        of outcomes.

        Args:
            returns: Historical returns
            n_simulations: Number of simulations
            n_periods: Number of periods to simulate (default: len(returns))

        Returns:
            Dict with simulation results
        """
        if n_periods is None:
            n_periods = len(returns)

        mean_return = returns.mean()
        std_return = returns.std()

        # Simulate returns
        simulated_returns = np.random.normal(
            mean_return,
            std_return,
            size=(n_simulations, n_periods)
        )

        # Calculate terminal wealth for each simulation
        terminal_values = (1 + simulated_returns).prod(axis=1)

        return {
            'mean_terminal_value': terminal_values.mean(),
            'median_terminal_value': np.median(terminal_values),
            'percentile_5': np.percentile(terminal_values, 5),
            'percentile_95': np.percentile(terminal_values, 95),
            'prob_positive': (terminal_values > 1.0).mean(),
            'prob_double': (terminal_values > 2.0).mean(),
            'worst_case_5pct': np.percentile(terminal_values, 5)
        }

    @staticmethod
    def sharpe_ratio_significance(
        returns: pd.Series,
        benchmark_sharpe: float = 0.40,
        risk_free_rate: float = 0.04
    ) -> Dict:
        """
        Test if Sharpe ratio is significantly better than benchmark.

        Uses Andrew Lo's test for Sharpe ratio significance.

        Args:
            returns: Strategy returns
            benchmark_sharpe: Benchmark Sharpe ratio
            risk_free_rate: Annual risk-free rate

        Returns:
            Dict with test results
        """
        n = len(returns)
        mean_return = returns.mean() * 252 - risk_free_rate
        std_return = returns.std() * np.sqrt(252)

        if std_return == 0:
            return {'error': 'Zero volatility'}

        sharpe = mean_return / std_return

        # Standard error of Sharpe ratio
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n)

        # Z-score
        z_score = (sharpe - benchmark_sharpe) / se_sharpe

        # P-value (one-tailed test: is our Sharpe > benchmark?)
        p_value = 1 - stats.norm.cdf(z_score)

        return {
            'sharpe_ratio': sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'z_score': z_score,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': f"Sharpe {sharpe:.2f} is {'SIGNIFICANTLY BETTER' if p_value < 0.05 else 'NOT significantly better'} than benchmark {benchmark_sharpe:.2f}"
        }

    @staticmethod
    def permutation_test(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        n_permutations: int = 10000
    ) -> Dict:
        """
        Permutation test for strategy vs benchmark.

        Non-parametric test: randomly shuffle labels and see if observed
        difference could occur by chance.

        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            n_permutations: Number of random permutations

        Returns:
            Dict with test results
        """
        observed_diff = strategy_returns.mean() - benchmark_returns.mean()

        # Combine returns
        combined = np.concatenate([strategy_returns.values, benchmark_returns.values])
        n_strategy = len(strategy_returns)

        # Permutation test
        perm_diffs = []
        for _ in range(n_permutations):
            # Randomly assign to strategy vs benchmark
            np.random.shuffle(combined)
            perm_strategy = combined[:n_strategy]
            perm_benchmark = combined[n_strategy:]

            perm_diff = perm_strategy.mean() - perm_benchmark.mean()
            perm_diffs.append(perm_diff)

        perm_diffs = np.array(perm_diffs)

        # P-value: fraction of permutations with difference >= observed
        p_value = (perm_diffs >= observed_diff).mean()

        return {
            'observed_difference': observed_diff,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': f"Observed difference {'IS' if p_value < 0.05 else 'is NOT'} statistically significant"
        }

    @staticmethod
    def regime_stability_analysis(
        returns: pd.Series,
        regime_labels: pd.Series = None
    ) -> Dict:
        """
        Analyze performance stability across market regimes.

        Tests if strategy performs consistently in different market conditions.

        Args:
            returns: Strategy returns
            regime_labels: Optional regime labels (bull/bear/sideways)

        Returns:
            Dict with regime analysis
        """
        if regime_labels is None:
            # Simple regime detection: rolling 60-day return
            rolling_return = returns.rolling(60).sum()
            regime_labels = pd.cut(
                rolling_return,
                bins=3,
                labels=['Bear', 'Neutral', 'Bull']
            )

        results = {}

        for regime in regime_labels.unique():
            if pd.isna(regime):
                continue

            regime_returns = returns[regime_labels == regime]

            if len(regime_returns) > 0:
                results[str(regime)] = {
                    'mean_return': regime_returns.mean() * 252,
                    'volatility': regime_returns.std() * np.sqrt(252),
                    'sharpe': (regime_returns.mean() / regime_returns.std()) * np.sqrt(252) if regime_returns.std() > 0 else 0,
                    'n_periods': len(regime_returns)
                }

        return results

    @staticmethod
    def calculate_confidence_intervals(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Calculate confidence intervals for returns.

        Args:
            returns: Returns series
            confidence_level: Confidence level (default: 95%)

        Returns:
            Dict with confidence intervals
        """
        mean = returns.mean() * 252
        std_err = returns.std() / np.sqrt(len(returns))

        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        ci_lower = (mean - z_score * std_err * np.sqrt(252))
        ci_upper = (mean + z_score * std_err * np.sqrt(252))

        return {
            'mean_annual_return': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence_level
        }

    @staticmethod
    def comprehensive_validation(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.04
    ) -> Dict:
        """
        Run comprehensive statistical validation.

        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Dict with all validation results
        """
        excess_returns = strategy_returns - benchmark_returns

        results = {
            't_test': StatisticalValidator.t_test_excess_returns(excess_returns),
            'monte_carlo': StatisticalValidator.monte_carlo_simulation(strategy_returns),
            'sharpe_significance': StatisticalValidator.sharpe_ratio_significance(
                strategy_returns,
                (benchmark_returns.mean() * 252 - risk_free_rate) / (benchmark_returns.std() * np.sqrt(252)),
                risk_free_rate
            ),
            'permutation': StatisticalValidator.permutation_test(strategy_returns, benchmark_returns),
            'confidence_intervals': StatisticalValidator.calculate_confidence_intervals(strategy_returns)
        }

        return results
