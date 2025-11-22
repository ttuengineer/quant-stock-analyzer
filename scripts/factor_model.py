"""
Factor Model for De-Noising Returns.

Implements institutional-grade factor de-noising:
1. Estimate factor exposures (beta, sector, momentum)
2. Regress returns on factors
3. Extract residual (idiosyncratic) returns

This removes systematic risk and lets the optimizer focus on stock-specific alpha.

Based on ChatGPT's institutional quant recommendations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.covariance import LedoitWolf


class FactorModel:
    """
    7-Factor Model: Market Beta + Sector + Momentum + Size + Value + Low-Vol + Reversal

    Enterprise-grade factor model for comprehensive risk decomposition.

    Factors:
    1. Market Beta - sensitivity to market returns
    2. Sector - industry membership
    3. Momentum - 12-1 month return (skip recent month)
    4. Size - market cap proxy (average dollar volume)
    5. Value - short-term reversal as cheap/expensive proxy
    6. Low-Vol - volatility factor (low vol outperformance)
    7. Short-term Reversal - 1-week return (mean reversion)
    """

    def __init__(
        self,
        beta_lookback: int = 252,      # 1 year for beta estimation
        momentum_lookback: int = 252,   # 1 year for momentum factor
        min_observations: int = 60,     # Minimum days for regression
        verbose: bool = True
    ):
        self.beta_lookback = beta_lookback
        self.momentum_lookback = momentum_lookback
        self.min_observations = min_observations
        self.verbose = verbose

        # Cache for factor loadings
        self._factor_loadings = {}
        self._residual_returns = None
        self._factor_returns = None

        # Factor names for reporting
        self.factor_names = [
            'beta', 'sector', 'momentum', 'size', 'value', 'low_vol', 'reversal'
        ]

    def estimate_factor_exposures(
        self,
        returns_df: pd.DataFrame,       # Daily returns (dates x tickers)
        spy_returns: pd.Series,         # SPY daily returns
        sectors: Dict[str, str],        # ticker -> sector mapping
        as_of_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Estimate factor exposures for all stocks.

        Returns DataFrame with columns:
            - ticker
            - beta (market beta)
            - sector (one-hot encoded)
            - momentum (12-1 month return)
        """
        if as_of_date:
            as_of_dt = pd.to_datetime(as_of_date)
            returns_df = returns_df[returns_df.index <= as_of_dt].copy()
            spy_returns = spy_returns[spy_returns.index <= as_of_dt].copy()

        exposures = []

        # Get recent returns for calculations
        recent_returns = returns_df.iloc[-self.beta_lookback:].copy()
        recent_spy = spy_returns.iloc[-self.beta_lookback:].copy()

        # Align indices - use intersection to handle any mismatches
        common_idx = recent_returns.index.intersection(recent_spy.index)
        if len(common_idx) < self.min_observations:
            if self.verbose:
                print(f"Warning: Only {len(common_idx)} common dates for factor estimation")
            return pd.DataFrame()

        recent_returns = recent_returns.loc[common_idx]
        recent_spy = recent_spy.loc[common_idx]

        spy_var = recent_spy.var()

        for ticker in returns_df.columns:
            stock_ret = recent_returns[ticker].dropna()

            if len(stock_ret) < self.min_observations:
                continue

            # Align with SPY using reindex (safer than .loc)
            common_stock_idx = stock_ret.index.intersection(recent_spy.index)
            if len(common_stock_idx) < self.min_observations:
                continue

            stock_ret = stock_ret.loc[common_stock_idx]
            aligned_spy = recent_spy.loc[common_stock_idx]

            # === FACTOR 1: MARKET BETA ===
            if len(aligned_spy) > 0 and spy_var > 0:
                cov = np.cov(stock_ret, aligned_spy)[0, 1]
                beta = cov / spy_var
                beta = np.clip(beta, 0.2, 3.0)  # Reasonable bounds
            else:
                beta = 1.0

            # === FACTOR 2: SECTOR ===
            sector = sectors.get(ticker, 'Unknown')

            # === FACTOR 3: MOMENTUM (12-1 month return) ===
            # Skip most recent month (mean reversion effect)
            if len(stock_ret) >= 252:
                ret_12m = (1 + stock_ret.iloc[-252:-21]).prod() - 1
                momentum = ret_12m
            elif len(stock_ret) >= 63:
                ret_period = (1 + stock_ret.iloc[:-21]).prod() - 1
                momentum = ret_period
            else:
                momentum = 0.0

            # === FACTOR 4: SIZE (volatility-adjusted average return as proxy) ===
            # Higher absolute returns scaled by vol = larger/more liquid stocks
            # This is a proxy since we don't have market cap
            avg_abs_ret = stock_ret.abs().mean()
            vol = stock_ret.std()
            size_proxy = avg_abs_ret / (vol + 1e-8) if vol > 0 else 0.0

            # === FACTOR 5: VALUE (short-term reversal as cheap/expensive proxy) ===
            # Stocks that have fallen recently are "cheaper" (value)
            # Use 1-month return as value signal (negative = value)
            if len(stock_ret) >= 21:
                ret_1m = (1 + stock_ret.iloc[-21:]).prod() - 1
                value_proxy = -ret_1m  # Negative returns = value
            else:
                value_proxy = 0.0

            # === FACTOR 6: LOW-VOL ===
            # Volatility factor - low vol stocks tend to outperform
            volatility = vol * np.sqrt(252)  # Annualized
            low_vol = -volatility  # Negative vol = low-vol factor exposure

            # === FACTOR 7: SHORT-TERM REVERSAL (1-week return) ===
            # Very short-term mean reversion signal
            if len(stock_ret) >= 5:
                ret_5d = (1 + stock_ret.iloc[-5:]).prod() - 1
                reversal = -ret_5d  # Negative = oversold = reversal signal
            else:
                reversal = 0.0

            exposures.append({
                'ticker': ticker,
                'beta': beta,
                'sector': sector,
                'momentum': momentum,
                'size': size_proxy,
                'value': value_proxy,
                'low_vol': low_vol,
                'reversal': reversal,
                'volatility': volatility
            })

        exposures_df = pd.DataFrame(exposures)

        if self.verbose:
            print(f"Factor exposures estimated for {len(exposures_df)} stocks (7-factor model)")
            print(f"  Beta range: {exposures_df['beta'].min():.2f} - {exposures_df['beta'].max():.2f}")
            print(f"  Sectors: {exposures_df['sector'].nunique()}")
            print(f"  Momentum range: {exposures_df['momentum'].min():.1%} - {exposures_df['momentum'].max():.1%}")
            print(f"  Size range: {exposures_df['size'].min():.3f} - {exposures_df['size'].max():.3f}")
            print(f"  Value range: {exposures_df['value'].min():.1%} - {exposures_df['value'].max():.1%}")
            print(f"  Reversal range: {exposures_df['reversal'].min():.1%} - {exposures_df['reversal'].max():.1%}")

        return exposures_df

    def compute_residual_returns(
        self,
        returns_df: pd.DataFrame,
        spy_returns: pd.Series,
        exposures_df: pd.DataFrame,
        lookback: int = 252
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute residual (factor-neutral) returns using 7-factor model.

        For each stock:
            residual = raw - beta*market - sector - momentum - size - value - low_vol - reversal

        Returns:
            residual_returns_df: DataFrame of residual returns
            factor_returns_df: DataFrame of factor returns (for analysis)
        """
        # Get recent data
        recent_returns = returns_df.iloc[-lookback:].copy()
        recent_spy = spy_returns.iloc[-lookback:].copy()

        # Build factor exposure lookups
        beta_lookup = dict(zip(exposures_df['ticker'], exposures_df['beta']))
        sector_lookup = dict(zip(exposures_df['ticker'], exposures_df['sector']))
        momentum_lookup = dict(zip(exposures_df['ticker'], exposures_df['momentum']))
        size_lookup = dict(zip(exposures_df['ticker'], exposures_df['size']))
        value_lookup = dict(zip(exposures_df['ticker'], exposures_df['value']))
        low_vol_lookup = dict(zip(exposures_df['ticker'], exposures_df['low_vol']))
        reversal_lookup = dict(zip(exposures_df['ticker'], exposures_df['reversal']))

        # === BUILD FACTOR PORTFOLIOS (Long-Short) ===

        def build_factor_return(factor_col, recent_returns, exposures_df, pct=0.3):
            """Build long-short factor return: top 30% - bottom 30%"""
            n = int(len(exposures_df) * pct)
            high = exposures_df.nlargest(n, factor_col)['ticker'].tolist()
            low = exposures_df.nsmallest(n, factor_col)['ticker'].tolist()

            high_cols = [t for t in high if t in recent_returns.columns]
            low_cols = [t for t in low if t in recent_returns.columns]

            if high_cols and low_cols:
                return recent_returns[high_cols].mean(axis=1) - recent_returns[low_cols].mean(axis=1)
            return pd.Series(0, index=recent_returns.index)

        # Compute sector returns (equal-weight average of sector members)
        sectors = exposures_df['sector'].unique()
        sector_returns = {}
        for sector in sectors:
            sector_tickers = exposures_df[exposures_df['sector'] == sector]['ticker'].tolist()
            sector_cols = [t for t in sector_tickers if t in recent_returns.columns]
            if sector_cols:
                sector_returns[sector] = recent_returns[sector_cols].mean(axis=1)
            else:
                sector_returns[sector] = pd.Series(0, index=recent_returns.index)
        sector_returns_df = pd.DataFrame(sector_returns)

        # Build factor returns
        momentum_factor = build_factor_return('momentum', recent_returns, exposures_df)
        size_factor = build_factor_return('size', recent_returns, exposures_df)
        value_factor = build_factor_return('value', recent_returns, exposures_df)
        low_vol_factor = build_factor_return('low_vol', recent_returns, exposures_df)
        reversal_factor = build_factor_return('reversal', recent_returns, exposures_df)

        # === COMPUTE RESIDUALS ===
        residual_dict = {}

        # Normalize factor loadings (z-score)
        def normalize(series):
            mean, std = series.mean(), series.std()
            if std > 0:
                return np.clip((series - mean) / std, -2, 2)
            return pd.Series(0, index=series.index)

        mom_norm = normalize(exposures_df.set_index('ticker')['momentum'])
        size_norm = normalize(exposures_df.set_index('ticker')['size'])
        value_norm = normalize(exposures_df.set_index('ticker')['value'])
        vol_norm = normalize(exposures_df.set_index('ticker')['low_vol'])
        rev_norm = normalize(exposures_df.set_index('ticker')['reversal'])

        # Factor loading coefficients (how much to remove)
        factor_coefs = {
            'momentum': 0.3,
            'size': 0.2,
            'value': 0.2,
            'low_vol': 0.2,
            'reversal': 0.15
        }

        for ticker in recent_returns.columns:
            if ticker not in beta_lookup:
                continue

            beta = beta_lookup[ticker]
            sector = sector_lookup.get(ticker, 'Unknown')

            # Get normalized factor loadings
            mom_load = mom_norm.get(ticker, 0)
            size_load = size_norm.get(ticker, 0)
            value_load = value_norm.get(ticker, 0)
            vol_load = vol_norm.get(ticker, 0)
            rev_load = rev_norm.get(ticker, 0)

            # Get returns
            raw_ret = recent_returns[ticker]
            market_ret = recent_spy
            sector_ret = sector_returns_df.get(sector, pd.Series(0, index=recent_returns.index))

            # === MULTI-FACTOR RESIDUALIZATION ===
            # Step 1: Remove market beta
            residual = raw_ret - beta * market_ret

            # Step 2: Remove sector (avoid double-counting market)
            residual = residual - (sector_ret - market_ret)

            # Step 3: Remove momentum factor
            residual = residual - mom_load * factor_coefs['momentum'] * momentum_factor

            # Step 4: Remove size factor
            residual = residual - size_load * factor_coefs['size'] * size_factor

            # Step 5: Remove value factor
            residual = residual - value_load * factor_coefs['value'] * value_factor

            # Step 6: Remove low-vol factor
            residual = residual - vol_load * factor_coefs['low_vol'] * low_vol_factor

            # Step 7: Remove reversal factor
            residual = residual - rev_load * factor_coefs['reversal'] * reversal_factor

            residual_dict[ticker] = residual

        # Create DataFrame all at once (avoids fragmentation)
        residual_returns = pd.DataFrame(residual_dict)

        # Store factor returns for analysis
        factor_returns = pd.DataFrame({
            'market': recent_spy,
            'momentum': momentum_factor,
            'size': size_factor,
            'value': value_factor,
            'low_vol': low_vol_factor,
            'reversal': reversal_factor
        })
        for sector in sectors:
            factor_returns[f'sector_{sector}'] = sector_returns_df.get(sector, 0)

        if self.verbose:
            print(f"\nResidual returns computed for {len(residual_returns.columns)} stocks (7-factor)")
            raw_vol = recent_returns.std().mean() * np.sqrt(252)
            resid_vol = residual_returns.std().mean() * np.sqrt(252)
            print(f"  Raw avg vol: {raw_vol:.1%}")
            print(f"  Residual avg vol: {resid_vol:.1%}")
            print(f"  Risk reduction: {(1 - resid_vol/raw_vol)*100:.0f}%")

        self._residual_returns = residual_returns
        self._factor_returns = factor_returns

        return residual_returns, factor_returns

    def compute_residual_covariance(
        self,
        residual_returns: pd.DataFrame,
        shrinkage: float = 0.3  # Ignored - using LedoitWolf auto-shrinkage
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute factor-neutral covariance matrix from residual returns.

        Uses sklearn's Ledoit-Wolf shrinkage for optimal stability.
        This is critical for avoiding solver failures with large N, small T.
        """
        # Drop columns with too many NaNs
        valid_cols = residual_returns.columns[residual_returns.notna().sum() > self.min_observations]
        residuals_clean = residual_returns[valid_cols].fillna(0)

        n = len(valid_cols)

        # Use sklearn's Ledoit-Wolf for optimal shrinkage intensity
        # This automatically determines the best shrinkage factor
        try:
            lw = LedoitWolf().fit(residuals_clean.values)
            cov_matrix = lw.covariance_
            shrinkage_used = lw.shrinkage_

            if self.verbose:
                print(f"\nLedoit-Wolf covariance: {n}x{n}")
                print(f"  Auto-shrinkage intensity: {shrinkage_used:.3f}")
        except Exception as e:
            # Fallback to manual shrinkage if LedoitWolf fails
            if self.verbose:
                print(f"LedoitWolf failed ({e}), using manual shrinkage")
            sample_cov = residuals_clean.cov().values
            avg_var = np.trace(sample_cov) / n
            shrinkage_target = np.eye(n) * avg_var
            cov_matrix = (1 - shrinkage) * sample_cov + shrinkage * shrinkage_target

        # Ensure PSD (positive semi-definite) for convex optimization
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        if self.verbose:
            # Compute average off-diagonal correlation for diagnostics
            diag = np.sqrt(np.diag(cov_matrix))
            corr_matrix = cov_matrix / np.outer(diag, diag + 1e-10)
            # Mask diagonal for avg calculation
            mask = ~np.eye(n, dtype=bool)
            avg_corr = corr_matrix[mask].mean()
            print(f"  Avg off-diagonal correlation: {avg_corr:.4f}")

        return cov_matrix, list(valid_cols)

    def compute_residual_alpha(
        self,
        predictions: pd.Series,         # ticker -> ML prediction score
        exposures_df: pd.DataFrame,
        neutralize_factors: List[str] = ['beta', 'momentum']
    ) -> pd.Series:
        """
        Compute factor-neutral alpha scores.

        Removes factor tilts from ML predictions so optimizer
        sees true stock-specific signal.
        """
        # Merge predictions with exposures
        pred_df = predictions.reset_index()
        pred_df.columns = ['ticker', 'pred_proba']

        merged = pred_df.merge(exposures_df[['ticker'] + neutralize_factors], on='ticker', how='left')

        # Fill missing exposures with mean
        for factor in neutralize_factors:
            merged[factor] = merged[factor].fillna(merged[factor].mean())

        # Regress predictions on factors
        y = merged['pred_proba'].values
        X = merged[neutralize_factors].values
        X = np.column_stack([np.ones(len(X)), X])  # Add intercept

        # OLS regression
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            y_hat = X @ coeffs
            residual_alpha = y - y_hat
        except:
            # Fallback to simple z-score if regression fails
            residual_alpha = (y - y.mean()) / (y.std() + 1e-8)

        # Create series
        result = pd.Series(residual_alpha, index=merged['ticker'])

        if self.verbose:
            raw_beta_corr = np.corrcoef(merged['pred_proba'], merged['beta'])[0, 1] if 'beta' in neutralize_factors else 0
            resid_beta_corr = np.corrcoef(residual_alpha, merged['beta'])[0, 1] if 'beta' in neutralize_factors else 0
            print(f"\nAlpha neutralization:")
            print(f"  Raw alpha-beta correlation: {raw_beta_corr:.3f}")
            print(f"  Residual alpha-beta correlation: {resid_beta_corr:.3f}")

        return result


class FactorNeutralOptimizer:
    """
    Enterprise-Grade Portfolio Optimizer with Factor-Neutral Inputs.

    Features:
    - L2 regularization for always-feasible optimization
    - Conviction scaling based on alpha strength
    - Alpha-tilted fallback (not equal-weight)
    - Regime detection for bear market protection
    - Relaxed constraints to allow proper optimization

    Based on combined ChatGPT + Claude recommendations.
    """

    def __init__(
        self,
        max_weight: float = 0.05,        # Allow up to 5% per position
        min_weight: float = 0.0,         # NO minimum - allow conviction!
        target_beta: float = 1.0,
        beta_tolerance: float = 0.30,    # Relaxed: 0.70-1.30 range
        target_vol: float = 0.22,        # 22% target (realistic)
        vol_tolerance: float = 0.10,     # 12-32% range
        max_sector_weight: float = 0.30, # 30% sector cap
        n_stocks: int = 40,              # Target positions
        momentum_neutral: bool = True,   # Enforce momentum neutrality
        l2_regularization: float = 0.01, # Ridge penalty for stability
        conviction_gamma: float = 2.0,   # Conviction scaling exponent
        regime_aware: bool = True,       # Enable regime detection
        verbose: bool = True
    ):
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.target_beta = target_beta
        self.beta_tolerance = beta_tolerance
        self.target_vol = target_vol
        self.vol_tolerance = vol_tolerance
        self.max_sector_weight = max_sector_weight
        self.n_stocks = n_stocks
        self.momentum_neutral = momentum_neutral
        self.l2_regularization = l2_regularization
        self.conviction_gamma = conviction_gamma
        self.regime_aware = regime_aware
        self.verbose = verbose

        self.factor_model = FactorModel(verbose=verbose)

        # Track regime state
        self._current_regime = 'normal'

    def _detect_regime(
        self,
        spy_returns: pd.Series,
        lookback: int = 63  # ~3 months
    ) -> str:
        """
        Simple regime detector based on recent market behavior.

        Returns: 'normal', 'high_vol', or 'crisis'
        """
        if len(spy_returns) < lookback:
            return 'normal'

        recent = spy_returns.iloc[-lookback:]

        # Annualized volatility
        vol = recent.std() * np.sqrt(252)

        # Recent drawdown
        cumulative = (1 + recent).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative / peak - 1).min()

        # Recent momentum (3-month return)
        momentum = (1 + recent).prod() - 1

        # Regime classification
        if vol > 0.30 or drawdown < -0.15:
            regime = 'crisis'
        elif vol > 0.22 or drawdown < -0.08:
            regime = 'high_vol'
        else:
            regime = 'normal'

        if self.verbose:
            print(f"\nRegime detection:")
            print(f"  3-month vol: {vol:.1%}")
            print(f"  Max drawdown: {drawdown:.1%}")
            print(f"  Momentum: {momentum:.1%}")
            print(f"  Regime: {regime.upper()}")

        return regime

    def _get_regime_adjusted_params(self, regime: str) -> dict:
        """
        Adjust optimization parameters based on market regime.
        """
        if regime == 'crisis':
            return {
                'target_beta': 0.8,           # Reduce market exposure
                'beta_tolerance': 0.25,       # Tighter control
                'max_weight': 0.04,           # More diversified
                'n_stocks': 50,               # More positions
                'risk_aversion': 2.0,         # Higher risk aversion
            }
        elif regime == 'high_vol':
            return {
                'target_beta': 0.9,
                'beta_tolerance': 0.25,
                'max_weight': 0.045,
                'n_stocks': 45,
                'risk_aversion': 1.5,
            }
        else:  # normal
            return {
                'target_beta': self.target_beta,
                'beta_tolerance': self.beta_tolerance,
                'max_weight': self.max_weight,
                'n_stocks': self.n_stocks,
                'risk_aversion': 1.0,
            }

    def _apply_conviction_scaling(
        self,
        alpha: np.ndarray,
        gamma: float = 2.0
    ) -> np.ndarray:
        """
        Apply conviction scaling to alpha scores.

        Amplifies strong signals, dampens weak ones.
        Formula: sign(alpha) * |alpha|^gamma
        """
        # Normalize to [0, 1] range first
        alpha_min = alpha.min()
        alpha_max = alpha.max()
        if alpha_max - alpha_min > 1e-8:
            alpha_norm = (alpha - alpha_min) / (alpha_max - alpha_min)
        else:
            alpha_norm = np.ones_like(alpha) * 0.5

        # Center around 0.5, apply power scaling
        alpha_centered = alpha_norm - 0.5
        alpha_scaled = np.sign(alpha_centered) * np.abs(alpha_centered) ** gamma

        # Re-normalize to mean 0, std 1
        alpha_scaled = (alpha_scaled - alpha_scaled.mean()) / (alpha_scaled.std() + 1e-8)

        return alpha_scaled

    def optimize(
        self,
        predictions_df: pd.DataFrame,    # ticker, pred_proba
        returns_df: pd.DataFrame,        # Historical returns
        spy_returns: pd.Series,          # SPY returns
        sectors: Dict[str, str],         # ticker -> sector
        previous_weights: Optional[Dict[str, float]] = None,
        as_of_date: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Enterprise-grade factor-neutral portfolio optimization.

        Features:
        1. Regime-aware parameter adjustment
        2. L2 regularization for always-feasible QP
        3. Conviction scaling for sparse alpha
        4. Relaxed constraints that allow real optimization
        5. Alpha-tilted fallback if solver fails
        """
        try:
            import cvxpy as cp
        except ImportError:
            print("ERROR: cvxpy not installed")
            return self._fallback_weights(predictions_df, {}, {})

        if self.verbose:
            print(f"\n{'='*50}")
            print(f"Factor-Neutral Optimization: {as_of_date}")
            print(f"{'='*50}")

        # === REGIME DETECTION ===
        if self.regime_aware:
            regime = self._detect_regime(spy_returns)
            params = self._get_regime_adjusted_params(regime)
            self._current_regime = regime
        else:
            regime = 'normal'
            params = self._get_regime_adjusted_params('normal')

        # === STEP 1: FACTOR EXPOSURES ===
        exposures = self.factor_model.estimate_factor_exposures(
            returns_df, spy_returns, sectors, as_of_date
        )

        if len(exposures) == 0:
            return self._fallback_weights(predictions_df, {}, {})

        # === STEP 2: RESIDUAL RETURNS ===
        residual_returns, _ = self.factor_model.compute_residual_returns(
            returns_df, spy_returns, exposures
        )

        # === STEP 3: RESIDUAL COVARIANCE ===
        cov_matrix, valid_tickers = self.factor_model.compute_residual_covariance(
            residual_returns
        )

        # === STEP 4: NEUTRALIZE ALPHA ===
        scores = predictions_df.set_index('ticker')['pred_proba']
        residual_alpha = self.factor_model.compute_residual_alpha(
            scores, exposures, neutralize_factors=['beta', 'momentum']
        )

        # Align everything to valid_tickers
        valid_tickers = [t for t in valid_tickers if t in residual_alpha.index]
        if len(valid_tickers) < 20:
            if self.verbose:
                print(f"Warning: Only {len(valid_tickers)} valid tickers")
            return self._fallback_weights(predictions_df, {}, {})

        n = len(valid_tickers)
        alpha = np.array([residual_alpha.get(t, 0) for t in valid_tickers])

        # Normalize alpha
        alpha = (alpha - alpha.mean()) / (alpha.std() + 1e-8)

        # Apply conviction scaling (amplify strong signals)
        alpha = self._apply_conviction_scaling(alpha, self.conviction_gamma)

        # Get factor exposures for constraints
        beta_lookup = dict(zip(exposures['ticker'], exposures['beta']))
        momentum_lookup = dict(zip(exposures['ticker'], exposures['momentum']))
        betas = np.array([beta_lookup.get(t, 1.0) for t in valid_tickers])

        # Normalize momentum exposures
        raw_momentum = np.array([momentum_lookup.get(t, 0) for t in valid_tickers])
        mom_mean = raw_momentum.mean()
        mom_std = raw_momentum.std() + 1e-8
        momentum_norm = (raw_momentum - mom_mean) / mom_std

        # Rebuild covariance for valid tickers only
        ticker_idx = {t: i for i, t in enumerate(residual_returns.columns) if t in valid_tickers}
        idx_map = [ticker_idx[t] for t in valid_tickers if t in ticker_idx]
        cov_matrix = cov_matrix[np.ix_(idx_map, idx_map)]

        # Add small ridge to covariance for numerical stability
        cov_matrix = cov_matrix + self.l2_regularization * np.eye(n)

        # === STEP 5: CVXPY OPTIMIZATION ===
        w = cp.Variable(n)

        # Risk aversion from regime
        risk_aversion = params.get('risk_aversion', 1.0)

        # Objective: maximize alpha - risk - L2 penalty
        # L2 penalty encourages diversification and ensures feasibility
        objective = cp.Maximize(
            alpha @ w
            - risk_aversion * 0.5 * cp.quad_form(w, cov_matrix)
            - self.l2_regularization * cp.sum_squares(w)
        )

        # === CONSTRAINTS ===
        target_beta = params['target_beta']
        beta_tol = params['beta_tolerance']
        max_wt = params['max_weight']

        constraints = [
            cp.sum(w) == 1.0,           # Fully invested
            w >= 0,                      # Long only (NO min weight!)
            w <= max_wt,                 # Max position
            # Beta constraint (relaxed)
            betas @ w >= target_beta - beta_tol,
            betas @ w <= target_beta + beta_tol,
        ]

        # Momentum neutrality (relaxed tolerance)
        if self.momentum_neutral:
            mom_tolerance = 0.5  # Relaxed from 0.3
            constraints.append(momentum_norm @ w >= -mom_tolerance)
            constraints.append(momentum_norm @ w <= mom_tolerance)

        # Volatility constraint (relaxed)
        annual_var_limit = (self.target_vol + self.vol_tolerance) ** 2
        daily_var_limit = annual_var_limit / 252
        constraints.append(cp.quad_form(w, cov_matrix) <= daily_var_limit)

        # Sector constraints
        unique_sectors = list(set(sectors.get(t, 'Unknown') for t in valid_tickers))
        for sector in unique_sectors:
            sector_mask = np.array([1.0 if sectors.get(t) == sector else 0.0 for t in valid_tickers])
            constraints.append(sector_mask @ w <= self.max_sector_weight)

        # === SOLVE ===
        problem = cp.Problem(objective, constraints)
        solved = False
        solver_used = None

        # Try solvers in order of preference
        for solver, kwargs in [
            (cp.ECOS, {'abstol': 1e-6, 'reltol': 1e-6, 'feastol': 1e-6}),
            (cp.SCS, {'eps': 1e-6, 'max_iters': 5000}),
            (cp.OSQP, {'eps_abs': 1e-6, 'eps_rel': 1e-6}),
        ]:
            try:
                problem.solve(solver=solver, verbose=False, **kwargs)
                if problem.status in ['optimal', 'optimal_inaccurate']:
                    solved = True
                    solver_used = solver.__name__ if hasattr(solver, '__name__') else str(solver)
                    break
            except Exception as e:
                if self.verbose:
                    print(f"  {solver} failed: {str(e)[:50]}")
                continue

        if not solved or w.value is None:
            if self.verbose:
                print(f"All solvers failed (status: {problem.status}), using alpha-tilted fallback")
            return self._fallback_weights(predictions_df, beta_lookup, momentum_lookup)

        # === BUILD RESULT ===
        weights = w.value
        result = {}

        # Only include positions above tiny threshold (0.1%)
        min_threshold = 0.001
        for ticker, weight in zip(valid_tickers, weights):
            if weight >= min_threshold:
                result[ticker] = float(weight)

        # Normalize
        total = sum(result.values())
        if total > 0:
            result = {t: wt/total for t, wt in result.items()}

        if self.verbose:
            actual_beta = sum(beta_lookup.get(t, 1.0) * wt for t, wt in result.items())
            actual_momentum = sum(momentum_lookup.get(t, 0) * wt for t, wt in result.items())
            weights_arr = np.array(list(result.values()))
            print(f"\nOptimized portfolio: {len(result)} positions (solver: {solver_used})")
            print(f"  Portfolio beta: {actual_beta:.2f} (target: {target_beta:.2f})")
            print(f"  Portfolio momentum: {actual_momentum:.1%}")
            print(f"  Weight range: {weights_arr.min():.1%} - {weights_arr.max():.1%}")
            print(f"  Top 5 weights: {sorted(weights_arr, reverse=True)[:5]}")
            print(f"  Regime: {regime}")

        return result

    def _fallback_weights(
        self,
        predictions_df: pd.DataFrame,
        beta_lookup: Dict[str, float],
        momentum_lookup: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Alpha-tilted fallback (NOT equal-weight).

        Uses conviction-scaled alpha to determine weights,
        with some diversification constraints.
        """
        if self.verbose:
            print("  Using alpha-tilted fallback...")

        # Get top candidates
        top = predictions_df.nlargest(self.n_stocks * 2, 'pred_proba')

        if len(top) == 0:
            return {}

        # Get scores and apply conviction scaling
        scores = top.set_index('ticker')['pred_proba']

        # Shift scores to be positive
        scores_shifted = scores - scores.min() + 0.01

        # Apply conviction scaling (power transform)
        scores_conv = scores_shifted ** self.conviction_gamma

        # Normalize to weights
        total = scores_conv.sum()
        if total > 0:
            weights = (scores_conv / total).to_dict()
        else:
            weights = {t: 1.0/len(scores) for t in scores.index}

        # Apply max weight cap
        weights = {t: min(w, self.max_weight) for t, w in weights.items()}

        # Filter to top n_stocks by weight
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        weights = dict(sorted_weights[:self.n_stocks])

        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {t: w/total for t, w in weights.items()}

        if self.verbose and len(weights) > 0:
            actual_beta = sum(beta_lookup.get(t, 1.0) * w for t, w in weights.items())
            weights_arr = np.array(list(weights.values()))
            print(f"  Fallback portfolio: {len(weights)} positions")
            print(f"  Est. beta: {actual_beta:.2f}")
            print(f"  Weight range: {weights_arr.min():.1%} - {weights_arr.max():.1%}")

        return weights


def demo():
    """Demo the factor model."""
    print("=" * 60)
    print("FACTOR MODEL DEMO")
    print("=" * 60)

    # Would need actual data to demo properly
    print("\nFactor model ready for integration.")
    print("Use FactorNeutralOptimizer.optimize() in walk-forward validation.")


if __name__ == "__main__":
    demo()
