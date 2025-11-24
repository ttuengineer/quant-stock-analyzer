"""
Portfolio Optimizer using CVXPY.
Implements institutional-grade portfolio optimization with:
- Expected return maximization from ML predictions
- Risk control via covariance matrix
- Beta neutralization
- Sector exposure limits
- Position size constraints
- Turnover penalties

This is what real quant funds use to translate ML signals into portfolios.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("WARNING: cvxpy not installed. Run: pip install cvxpy")


class PortfolioOptimizer:
    """
    Convex portfolio optimizer using CVXPY.

    Solves:
        maximize: alpha' @ w - (gamma/2) * w' @ Sigma @ w - tau * ||w - w_prev||_1
        subject to:
            sum(w) = 0  (dollar neutral) or sum(w) = 1 (long only)
            |w_i| <= max_weight
            beta' @ w = target_beta
            sector_exposure constraints
            sum(|w|) <= max_gross_exposure
    """

    def __init__(
        self,
        max_weight: float = 0.05,           # Max 5% per position
        min_weight: float = 0.001,          # Min 0.1% per position (if non-zero)
        max_gross_exposure: float = 2.0,    # 200% gross (100% long + 100% short)
        target_beta: float = 0.0,           # Beta neutral
        risk_aversion: float = 1.0,         # Risk aversion parameter (gamma)
        turnover_penalty: float = 0.01,     # Turnover cost penalty (tau)
        sector_max_deviation: float = 0.10, # Max 10% deviation from benchmark per sector
        long_only: bool = False,            # Long-only or long-short
        verbose: bool = True
    ):
        if not CVXPY_AVAILABLE:
            raise ImportError("cvxpy required for portfolio optimization")

        self.max_weight = max_weight
        self.min_weight = min_weight
        self.max_gross_exposure = max_gross_exposure
        self.target_beta = target_beta
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty
        self.sector_max_deviation = sector_max_deviation
        self.long_only = long_only
        self.verbose = verbose

    def estimate_covariance(
        self,
        returns_df: pd.DataFrame,
        method: str = 'shrinkage',
        lookback_days: int = 252
    ) -> np.ndarray:
        """
        Estimate covariance matrix from historical returns.

        Args:
            returns_df: DataFrame with daily returns (columns = tickers)
            method: 'sample', 'shrinkage', or 'factor'
            lookback_days: Number of days for estimation

        Returns:
            Covariance matrix (N x N)
        """
        # Use most recent data
        returns = returns_df.iloc[-lookback_days:].copy()

        # Drop tickers with too many missing values
        valid_cols = returns.columns[returns.notna().sum() > lookback_days * 0.5]
        returns = returns[valid_cols].fillna(0)

        if method == 'sample':
            # Simple sample covariance
            cov = returns.cov().values

        elif method == 'shrinkage':
            # Ledoit-Wolf shrinkage (more stable)
            sample_cov = returns.cov().values
            n = sample_cov.shape[0]

            # Shrink towards diagonal (single factor model)
            avg_var = np.trace(sample_cov) / n
            shrinkage_target = np.eye(n) * avg_var

            # Optimal shrinkage intensity (simplified)
            shrinkage_intensity = 0.3  # Can be estimated more precisely

            cov = (1 - shrinkage_intensity) * sample_cov + shrinkage_intensity * shrinkage_target

        elif method == 'factor':
            # Simple factor model (market factor only)
            # More sophisticated: use PCA or Barra-style factors
            market_returns = returns.mean(axis=1)  # Equal-weight market proxy

            # Regress each stock on market
            betas = []
            residual_vars = []
            for col in returns.columns:
                stock_ret = returns[col].values
                market_ret = market_returns.values

                # Simple OLS
                cov_sm = np.cov(stock_ret, market_ret)[0, 1]
                var_m = np.var(market_ret)
                beta = cov_sm / var_m if var_m > 0 else 1.0

                residual = stock_ret - beta * market_ret
                resid_var = np.var(residual)

                betas.append(beta)
                residual_vars.append(resid_var)

            betas = np.array(betas)
            residual_vars = np.array(residual_vars)
            market_var = np.var(market_returns)

            # Factor covariance: beta @ beta.T * market_var + diag(residual_vars)
            cov = np.outer(betas, betas) * market_var + np.diag(residual_vars)

        else:
            raise ValueError(f"Unknown covariance method: {method}")

        # Ensure positive semi-definite
        cov = self._make_psd(cov)

        return cov, list(valid_cols)

    def _make_psd(self, matrix: np.ndarray, min_eigenvalue: float = 1e-6) -> np.ndarray:
        """Ensure matrix is positive semi-definite."""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def optimize(
        self,
        alpha: np.ndarray,
        tickers: List[str],
        cov_matrix: np.ndarray,
        betas: Optional[np.ndarray] = None,
        sectors: Optional[Dict[str, str]] = None,
        previous_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights.

        Args:
            alpha: Expected returns/alpha scores for each ticker (N,)
            tickers: List of ticker symbols
            cov_matrix: Covariance matrix (N x N)
            betas: Stock betas to market (N,) - for beta neutralization
            sectors: Dict mapping ticker -> sector
            previous_weights: Previous period weights for turnover penalty

        Returns:
            Dict of {ticker: weight}
        """
        n = len(tickers)

        if n == 0:
            return {}

        # Normalize alpha to reasonable scale
        alpha = np.array(alpha)
        alpha = (alpha - alpha.mean()) / (alpha.std() + 1e-8)

        # Decision variable: weights
        w = cp.Variable(n)

        # Build objective: maximize alpha - risk - turnover cost
        # Expected return term
        expected_return = alpha @ w

        # Risk term (quadratic)
        risk = cp.quad_form(w, cov_matrix)

        # Turnover penalty
        if previous_weights is not None:
            w_prev = np.array([previous_weights.get(t, 0.0) for t in tickers])
            turnover_cost = cp.norm(w - w_prev, 1)
        else:
            turnover_cost = 0

        # Objective
        objective = cp.Maximize(
            expected_return
            - (self.risk_aversion / 2) * risk
            - self.turnover_penalty * turnover_cost
        )

        # Constraints
        constraints = []

        # Long-only or dollar-neutral
        if self.long_only:
            constraints.append(cp.sum(w) == 1.0)  # Fully invested
            constraints.append(w >= 0)  # No shorts
        else:
            constraints.append(cp.sum(w) == 0)  # Dollar neutral

        # Position limits
        constraints.append(w <= self.max_weight)
        constraints.append(w >= -self.max_weight)

        # Gross exposure limit
        constraints.append(cp.norm(w, 1) <= self.max_gross_exposure)

        # Beta neutrality
        if betas is not None and not self.long_only:
            betas = np.array(betas)
            constraints.append(betas @ w == self.target_beta)

        # Sector constraints
        if sectors is not None and self.sector_max_deviation > 0:
            unique_sectors = list(set(sectors.values()))
            for sector in unique_sectors:
                sector_mask = np.array([1.0 if sectors.get(t) == sector else 0.0 for t in tickers])
                sector_weight = sector_mask @ w
                # Allow deviation from zero (neutral) up to max
                constraints.append(sector_weight <= self.sector_max_deviation)
                constraints.append(sector_weight >= -self.sector_max_deviation)

        # Solve
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.OSQP, verbose=False)

            if problem.status not in ['optimal', 'optimal_inaccurate']:
                if self.verbose:
                    print(f"Warning: Optimization status = {problem.status}")
                # Try with relaxed constraints
                problem.solve(solver=cp.SCS, verbose=False)
        except Exception as e:
            if self.verbose:
                print(f"Optimization error: {e}")
            # Fallback to simple z-score weights
            weights = alpha / np.abs(alpha).sum() * self.max_gross_exposure / 2
            return {t: float(w) for t, w in zip(tickers, weights)}

        # Extract weights
        if w.value is None:
            if self.verbose:
                print("Warning: No solution found, using fallback")
            weights = alpha / np.abs(alpha).sum() * self.max_gross_exposure / 2
        else:
            weights = w.value

        # Filter tiny positions
        result = {}
        for ticker, weight in zip(tickers, weights):
            if abs(weight) >= self.min_weight:
                result[ticker] = float(weight)

        if self.verbose:
            long_exposure = sum(w for w in result.values() if w > 0)
            short_exposure = sum(w for w in result.values() if w < 0)
            print(f"Optimized portfolio: {len(result)} positions")
            print(f"  Long exposure:  {long_exposure:.1%}")
            print(f"  Short exposure: {short_exposure:.1%}")
            print(f"  Net exposure:   {long_exposure + short_exposure:.1%}")
            print(f"  Gross exposure: {long_exposure - short_exposure:.1%}")

        return result

    def optimize_from_predictions(
        self,
        predictions_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        betas_df: Optional[pd.DataFrame] = None,
        sectors_dict: Optional[Dict[str, str]] = None,
        previous_weights: Optional[Dict[str, float]] = None,
        date: Optional[str] = None
    ) -> Dict[str, float]:
        """
        High-level interface: optimize from prediction DataFrame.

        Args:
            predictions_df: DataFrame with 'ticker' and 'pred_proba' columns
            returns_df: Historical returns DataFrame (dates x tickers)
            betas_df: DataFrame with 'ticker' and 'beta' columns
            sectors_dict: Dict mapping ticker -> sector
            previous_weights: Previous period weights
            date: Date string for logging

        Returns:
            Dict of {ticker: weight}
        """
        if self.verbose and date:
            print(f"\n=== Portfolio Optimization: {date} ===")

        # Get tickers and predictions
        tickers = predictions_df['ticker'].tolist()
        alpha = predictions_df['pred_proba'].values

        # Estimate covariance
        available_tickers = [t for t in tickers if t in returns_df.columns]
        if len(available_tickers) < 10:
            if self.verbose:
                print(f"Warning: Only {len(available_tickers)} tickers with return data")
            # Fallback to simple weights
            weights = (alpha - alpha.mean()) / np.abs(alpha - alpha.mean()).sum()
            return {t: float(w) for t, w in zip(tickers, weights)}

        # Filter to available tickers
        mask = predictions_df['ticker'].isin(available_tickers)
        predictions_df = predictions_df[mask].copy()
        tickers = predictions_df['ticker'].tolist()
        alpha = predictions_df['pred_proba'].values

        # Get covariance for these tickers
        returns_subset = returns_df[tickers].copy()
        cov_matrix, valid_tickers = self.estimate_covariance(returns_subset)

        # Align everything to valid_tickers
        valid_mask = predictions_df['ticker'].isin(valid_tickers)
        predictions_df = predictions_df[valid_mask]
        tickers = predictions_df['ticker'].tolist()
        alpha = predictions_df['pred_proba'].values

        # Get betas
        betas = None
        if betas_df is not None:
            betas = []
            for t in tickers:
                beta_row = betas_df[betas_df['ticker'] == t]
                if len(beta_row) > 0:
                    betas.append(beta_row['beta'].values[0])
                else:
                    betas.append(1.0)  # Default beta
            betas = np.array(betas)

        # Optimize
        return self.optimize(
            alpha=alpha,
            tickers=tickers,
            cov_matrix=cov_matrix,
            betas=betas,
            sectors=sectors_dict,
            previous_weights=previous_weights
        )


class LongOnlyOptimizer:
    """
    Long-only portfolio optimizer with institutional constraints.

    Based on ChatGPT's recommendations:
    - Beta target: 0.9-1.1 (not neutral)
    - Vol target: 15-18% annual
    - Position limits: max 5% per stock
    - Sector limits: max 20% per sector
    - Turnover constraint: max 25% per month
    """

    def __init__(
        self,
        max_weight: float = 0.05,           # Max 5% per position
        min_weight: float = 0.01,           # Min 1% per position (if included)
        target_beta: float = 1.0,           # Target beta (0.9-1.1 range)
        beta_tolerance: float = 0.1,        # Beta can be within target ± tolerance
        target_vol: float = 0.16,           # 16% annual volatility target
        vol_tolerance: float = 0.02,        # Vol can be within target ± tolerance
        max_sector_weight: float = 0.20,    # Max 20% per sector
        max_turnover: float = 0.25,         # Max 25% turnover per month
        n_stocks: int = 30,                 # Number of stocks to hold
        verbose: bool = True
    ):
        if not CVXPY_AVAILABLE:
            raise ImportError("cvxpy required. Run: pip install cvxpy")

        self.max_weight = max_weight
        self.min_weight = min_weight
        self.target_beta = target_beta
        self.beta_tolerance = beta_tolerance
        self.target_vol = target_vol
        self.vol_tolerance = vol_tolerance
        self.max_sector_weight = max_sector_weight
        self.max_turnover = max_turnover
        self.n_stocks = n_stocks
        self.verbose = verbose

    def estimate_volatility(self, returns_df: pd.DataFrame, lookback: int = 60) -> pd.Series:
        """Estimate annualized volatility for each stock."""
        recent = returns_df.iloc[-lookback:]
        daily_vol = recent.std()
        annual_vol = daily_vol * np.sqrt(252)
        return annual_vol

    def estimate_betas(self, returns_df: pd.DataFrame, spy_returns: pd.Series, lookback: int = 252) -> pd.Series:
        """Estimate rolling beta to SPY for each stock."""
        recent_stocks = returns_df.iloc[-lookback:]
        recent_spy = spy_returns.iloc[-lookback:]

        betas = {}
        spy_var = recent_spy.var()

        for col in recent_stocks.columns:
            stock_ret = recent_stocks[col].dropna()
            aligned_spy = recent_spy.loc[stock_ret.index]

            if len(stock_ret) > 60 and spy_var > 0:
                cov = np.cov(stock_ret, aligned_spy)[0, 1]
                beta = cov / spy_var
                beta = np.clip(beta, 0.2, 3.0)  # Reasonable bounds
            else:
                beta = 1.0
            betas[col] = beta

        return pd.Series(betas)

    def optimize(
        self,
        scores: pd.Series,                  # ticker -> ML score
        volatilities: pd.Series,            # ticker -> annualized vol
        betas: pd.Series,                   # ticker -> beta
        sectors: Dict[str, str],            # ticker -> sector
        cov_matrix: np.ndarray,             # covariance matrix
        tickers: List[str],                 # ordered tickers matching cov_matrix
        previous_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio with institutional constraints.

        Returns Dict of {ticker: weight} where weights sum to 1.0
        """
        n = len(tickers)

        if n < self.n_stocks:
            if self.verbose:
                print(f"Warning: Only {n} stocks available, need {self.n_stocks}")

        # Align all data to tickers
        scores_arr = np.array([scores.get(t, 0) for t in tickers])
        vol_arr = np.array([volatilities.get(t, 0.3) for t in tickers])
        beta_arr = np.array([betas.get(t, 1.0) for t in tickers])

        # Normalize scores to alpha
        alpha = (scores_arr - scores_arr.mean()) / (scores_arr.std() + 1e-8)

        # Decision variable
        w = cp.Variable(n)

        # === OBJECTIVE: Maximize signal-weighted return - risk ===
        expected_return = alpha @ w
        risk = cp.quad_form(w, cov_matrix)

        # Turnover penalty if previous weights exist
        if previous_weights is not None:
            w_prev = np.array([previous_weights.get(t, 0.0) for t in tickers])
            turnover = cp.norm(w - w_prev, 1)
            objective = cp.Maximize(expected_return - 0.5 * risk - 0.01 * turnover)
        else:
            objective = cp.Maximize(expected_return - 0.5 * risk)

        # === CONSTRAINTS ===
        constraints = []

        # 1. Fully invested (long-only)
        constraints.append(cp.sum(w) == 1.0)

        # 2. No short selling
        constraints.append(w >= 0)

        # 3. Max weight per position
        constraints.append(w <= self.max_weight)

        # 4. Beta constraint: target_beta ± tolerance
        portfolio_beta = beta_arr @ w
        constraints.append(portfolio_beta >= self.target_beta - self.beta_tolerance)
        constraints.append(portfolio_beta <= self.target_beta + self.beta_tolerance)

        # 5. Volatility constraint (approximation using diagonal)
        # Full: sqrt(w' @ cov @ w) <= target_vol
        # Approximation: w' @ var <= target_vol^2
        portfolio_var = cp.quad_form(w, cov_matrix)
        # Annualize: daily var * 252
        annual_var_limit = (self.target_vol + self.vol_tolerance) ** 2
        daily_var_limit = annual_var_limit / 252
        constraints.append(portfolio_var <= daily_var_limit)

        # 6. Sector constraints
        if sectors:
            unique_sectors = list(set(sectors.values()))
            for sector in unique_sectors:
                sector_mask = np.array([1.0 if sectors.get(t) == sector else 0.0 for t in tickers])
                sector_weight = sector_mask @ w
                constraints.append(sector_weight <= self.max_sector_weight)

        # 7. Turnover constraint
        if previous_weights is not None:
            w_prev = np.array([previous_weights.get(t, 0.0) for t in tickers])
            constraints.append(cp.norm(w - w_prev, 1) <= self.max_turnover * 2)

        # === SOLVE ===
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.OSQP, verbose=False)

            if problem.status not in ['optimal', 'optimal_inaccurate']:
                if self.verbose:
                    print(f"OSQP status: {problem.status}, trying SCS...")
                problem.solve(solver=cp.SCS, verbose=False)

        except Exception as e:
            if self.verbose:
                print(f"Optimization error: {e}, using fallback")
            return self._fallback_weights(scores, tickers)

        # Extract solution
        if w.value is None:
            if self.verbose:
                print("No solution found, using fallback")
            return self._fallback_weights(scores, tickers)

        weights = w.value

        # Build result dict
        result = {}
        for ticker, weight in zip(tickers, weights):
            if weight >= self.min_weight:
                result[ticker] = float(weight)

        # Normalize to ensure sum = 1
        total = sum(result.values())
        if total > 0:
            result = {t: w/total for t, w in result.items()}

        if self.verbose:
            actual_beta = sum(betas.get(t, 1.0) * w for t, w in result.items())
            actual_vol = np.sqrt(252 * problem.constraints[5].dual_value) if problem.constraints else 0
            print(f"Optimized: {len(result)} positions")
            print(f"  Portfolio Beta: {actual_beta:.2f} (target: {self.target_beta:.1f})")
            print(f"  Top weight: {max(result.values()):.1%}")

        return result

    def _fallback_weights(self, scores: pd.Series, tickers: List[str]) -> Dict[str, float]:
        """Fallback to score-weighted top-N when optimization fails."""
        # Get top N by score
        top_tickers = scores.nlargest(self.n_stocks).index.tolist()
        top_scores = scores.loc[top_tickers]

        # Score-weighted (normalized)
        total_score = top_scores.sum()
        if total_score > 0:
            weights = {t: s/total_score for t, s in top_scores.items()}
        else:
            weights = {t: 1.0/len(top_tickers) for t in top_tickers}

        # Apply max weight constraint
        weights = {t: min(w, self.max_weight) for t, w in weights.items()}

        # Renormalize
        total = sum(weights.values())
        return {t: w/total for t, w in weights.items()}

    def optimize_from_dataframe(
        self,
        predictions_df: pd.DataFrame,       # Must have 'ticker', 'pred_proba'
        returns_df: pd.DataFrame,           # Historical returns (dates x tickers)
        spy_returns: pd.Series,             # SPY daily returns
        sectors_dict: Dict[str, str],       # ticker -> sector
        previous_weights: Optional[Dict[str, float]] = None,
        date: Optional[str] = None
    ) -> Dict[str, float]:
        """
        High-level interface to optimize from prediction DataFrame.
        """
        if self.verbose and date:
            print(f"\n=== Long-Only Optimization: {date} ===")

        # Get scores
        scores = predictions_df.set_index('ticker')['pred_proba']

        # Get tickers with both predictions and returns
        available = [t for t in scores.index if t in returns_df.columns]
        if len(available) < 20:
            if self.verbose:
                print(f"Warning: Only {len(available)} tickers, using fallback")
            return self._fallback_weights(scores, available)

        scores = scores.loc[available]
        returns_subset = returns_df[available]

        # Estimate volatilities and betas
        volatilities = self.estimate_volatility(returns_subset)
        betas = self.estimate_betas(returns_subset, spy_returns)

        # Estimate covariance (shrinkage)
        returns_clean = returns_subset.fillna(0).iloc[-252:]
        sample_cov = returns_clean.cov().values
        n = sample_cov.shape[0]

        # Ledoit-Wolf shrinkage
        avg_var = np.trace(sample_cov) / n
        shrinkage_target = np.eye(n) * avg_var
        cov_matrix = 0.7 * sample_cov + 0.3 * shrinkage_target

        # Make PSD
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        return self.optimize(
            scores=scores,
            volatilities=volatilities,
            betas=betas,
            sectors=sectors_dict,
            cov_matrix=cov_matrix,
            tickers=available,
            previous_weights=previous_weights
        )


def demo_optimizer():
    """Demo the optimizer with synthetic data."""
    print("=" * 60)
    print("PORTFOLIO OPTIMIZER DEMO")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    n_stocks = 50
    tickers = [f"STOCK_{i}" for i in range(n_stocks)]

    # Synthetic alpha (predictions)
    alpha = np.random.randn(n_stocks) * 0.02  # Small alpha signals

    # Synthetic covariance (factor model)
    market_beta = np.random.uniform(0.5, 1.5, n_stocks)
    market_var = 0.04  # 20% annual vol -> 0.04 variance
    idio_var = np.random.uniform(0.02, 0.08, n_stocks)
    cov_matrix = np.outer(market_beta, market_beta) * market_var + np.diag(idio_var)

    # Sectors
    sectors = {t: f"Sector_{i % 5}" for i, t in enumerate(tickers)}

    # Create optimizer
    optimizer = PortfolioOptimizer(
        max_weight=0.05,
        max_gross_exposure=2.0,
        target_beta=0.0,
        risk_aversion=1.0,
        turnover_penalty=0.005,
        long_only=False,
        verbose=True
    )

    # Optimize
    print("\n--- Long-Short Optimization ---")
    weights_ls = optimizer.optimize(
        alpha=alpha,
        tickers=tickers,
        cov_matrix=cov_matrix,
        betas=market_beta,
        sectors=sectors
    )

    # Show top positions
    sorted_weights = sorted(weights_ls.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 Long Positions:")
    for t, w in sorted_weights[:5]:
        print(f"  {t}: {w:+.2%}")
    print("\nTop 5 Short Positions:")
    for t, w in sorted_weights[-5:]:
        print(f"  {t}: {w:+.2%}")

    # Long-only version
    print("\n--- Long-Only Optimization ---")
    optimizer_lo = PortfolioOptimizer(
        max_weight=0.10,
        max_gross_exposure=1.0,
        long_only=True,
        verbose=True
    )

    weights_lo = optimizer_lo.optimize(
        alpha=alpha,
        tickers=tickers,
        cov_matrix=cov_matrix,
        betas=market_beta,
        sectors=sectors
    )

    sorted_weights_lo = sorted(weights_lo.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Long-Only Positions:")
    for t, w in sorted_weights_lo[:10]:
        print(f"  {t}: {w:+.2%}")


if __name__ == "__main__":
    demo_optimizer()
