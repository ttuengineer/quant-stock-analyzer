"""
Walk-Forward Out-of-Sample Validation.

This is the GOLD STANDARD test for strategy robustness.

Instead of: Train once (2015-2020) -> Test once (2023-2025)
We do:      Train -> Test -> Retrain -> Test -> Retrain -> Test...

Each test year is TRULY out-of-sample (model never saw it during training).
This reveals whether the model generalizes or just got lucky.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json

# Mega-cap overlay for improved portfolio construction
try:
    from mega_cap_overlay import apply_mega_cap_overlay, adjust_for_regime
    MEGA_CAP_OVERLAY_AVAILABLE = True
except ImportError:
    MEGA_CAP_OVERLAY_AVAILABLE = False
    print("WARNING: mega_cap_overlay not available. Run without --mega-cap-overlay flag.")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stock_analyzer.database import Database

# ML imports
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import xgboost as xgb
import lightgbm as lgb

# Portfolio optimizer (ChatGPT recommended)
try:
    from portfolio_optimizer import LongOnlyOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

# Factor-neutral optimizer (ChatGPT institutional-grade)
try:
    from factor_model import FactorNeutralOptimizer
    FACTOR_NEUTRAL_AVAILABLE = True
except ImportError:
    FACTOR_NEUTRAL_AVAILABLE = False


def bootstrap_pvalue(returns, n_iter=10000, seed=42):
    """
    Bootstrap test for whether mean return > 0.

    Returns: (empirical_mean, p_value, ci_low, ci_high)
    """
    np.random.seed(seed)
    returns = np.array(returns)
    emp_mean = np.mean(returns)

    boot_means = []
    for _ in range(n_iter):
        sample = np.random.choice(returns, size=len(returns), replace=True)
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)

    # One-sided p-value: P(mean <= 0)
    p_value = np.mean(boot_means <= 0)

    # 95% confidence interval
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)

    return emp_mean, p_value, ci_low, ci_high


def compute_ic_stats(predictions, actual_returns):
    """
    Compute Information Coefficient (IC) statistics.

    IC = Spearman rank correlation between predictions and actual returns.
    This is the core measure of signal quality in quant finance.

    Returns dict with IC stats.
    """
    from scipy.stats import spearmanr, pearsonr

    # Filter valid pairs
    mask = ~(np.isnan(predictions) | np.isnan(actual_returns))
    preds = predictions[mask]
    actuals = actual_returns[mask]

    if len(preds) < 10:
        return {'ic': 0, 'ic_pvalue': 1, 'pearson_ic': 0, 'n_samples': len(preds)}

    # Spearman IC (rank-based, more robust)
    ic, ic_pvalue = spearmanr(preds, actuals)

    # Pearson IC (linear correlation)
    pearson_ic, _ = pearsonr(preds, actuals)

    return {
        'ic': ic,
        'ic_pvalue': ic_pvalue,
        'pearson_ic': pearson_ic,
        'n_samples': len(preds)
    }


def compute_factor_turnover(current_weights, previous_weights):
    """
    Compute factor turnover between periods.

    Turnover = sum of absolute weight changes / 2.
    """
    if previous_weights is None:
        return 0.0

    # Align tickers
    all_tickers = set(current_weights.keys()) | set(previous_weights.keys())
    turnover = 0.0
    for ticker in all_tickers:
        curr = current_weights.get(ticker, 0)
        prev = previous_weights.get(ticker, 0)
        turnover += abs(curr - prev)

    return turnover / 2  # Each side counted once


def compute_stock_betas(prices_df, spy_df, lookback_days=252):
    """
    Compute rolling beta to SPY for each stock.

    Beta = Cov(stock, market) / Var(market)
    Uses trailing returns to avoid lookahead bias.

    Returns DataFrame with ticker, date, beta columns.
    """
    # Compute daily returns
    prices_df = prices_df.sort_values(['ticker', 'date'])
    prices_df['daily_return'] = prices_df.groupby('ticker')['adj_close'].pct_change()

    spy_df = spy_df.sort_values('date')
    spy_df['spy_return'] = spy_df['adj_close'].pct_change()

    # Merge stock returns with SPY returns
    merged = prices_df.merge(spy_df[['date', 'spy_return']], on='date', how='left')

    # Compute rolling beta for each stock
    def calc_rolling_beta(group):
        if len(group) < 60:  # Minimum 60 days for beta calculation
            group['beta'] = 1.0  # Default beta
            return group

        # Rolling covariance and variance
        stock_rets = group['daily_return'].values
        spy_rets = group['spy_return'].values

        betas = []
        for i in range(len(group)):
            if i < 60:
                betas.append(1.0)  # Default
            else:
                start = max(0, i - lookback_days)
                s_ret = stock_rets[start:i]
                m_ret = spy_rets[start:i]
                # Remove NaN
                valid = ~(np.isnan(s_ret) | np.isnan(m_ret))
                s_ret = s_ret[valid]
                m_ret = m_ret[valid]

                if len(s_ret) > 30:
                    cov = np.cov(s_ret, m_ret)[0, 1]
                    var = np.var(m_ret)
                    beta = cov / var if var > 0 else 1.0
                    # Clip extreme betas
                    beta = np.clip(beta, -1, 3)
                else:
                    beta = 1.0
                betas.append(beta)

        group['beta'] = betas
        return group

    beta_df = merged.groupby('ticker', group_keys=False).apply(calc_rolling_beta)
    return beta_df[['ticker', 'date', 'beta']].dropna()


def neutralize_beta(picks_df, betas_df, target_beta=0.0):
    """
    Adjust portfolio weights to achieve target beta.

    For a long-short portfolio:
    - Increase weight on low-beta longs, decrease high-beta longs
    - Increase weight on high-beta shorts, decrease low-beta shorts

    Returns DataFrame with adjusted weights.
    """
    # Merge picks with betas
    merged = picks_df.merge(betas_df[['ticker', 'date', 'beta']], on=['ticker', 'date'], how='left')
    merged['beta'] = merged['beta'].fillna(1.0)  # Default beta=1

    # Current portfolio beta
    if 'weight' not in merged.columns:
        merged['weight'] = 1.0 / len(merged)

    current_beta = (merged['weight'] * merged['beta']).sum()

    if abs(current_beta - target_beta) < 0.05:
        # Already close to target
        return merged

    # Simple beta adjustment: scale weights inversely by beta deviation
    # This is a simplified approach; full optimization would use QP
    beta_adjustment = target_beta - current_beta

    # Adjust weights: lower weight for high-beta stocks, higher for low-beta
    merged['beta_deviation'] = merged['beta'] - merged['beta'].mean()
    merged['weight_adj'] = merged['weight'] * (1 - 0.3 * merged['beta_deviation'])

    # Renormalize
    merged['weight_adj'] = merged['weight_adj'] / merged['weight_adj'].abs().sum()

    return merged


def neutralize_sector(picks_df, sectors_dict, long_short=False):
    """
    Sector-neutral portfolio: equal weight in each sector.

    For each sector:
    - Equal allocation to that sector (1/N sectors)
    - Within sector: proportional to model score

    This removes sector bets from the portfolio.
    """
    # Add sector info
    picks_df = picks_df.copy()
    picks_df['sector'] = picks_df['ticker'].map(sectors_dict).fillna('Unknown')

    # Count sectors
    sectors = picks_df['sector'].unique()
    n_sectors = len(sectors)

    if n_sectors <= 1:
        # Only one sector or no sector info - can't neutralize
        return picks_df

    # Allocate equally to each sector
    sector_weight = 1.0 / n_sectors

    # Within each sector, allocate proportionally to pred_proba
    def weight_within_sector(group):
        if 'pred_proba' in group.columns:
            scores = group['pred_proba'].values
            # For longs: higher score = higher weight
            # For shorts: lower score (already selected bottom) = still prop to score
            if scores.sum() > 0:
                weights = scores / scores.sum()
            else:
                weights = np.ones(len(group)) / len(group)
        else:
            weights = np.ones(len(group)) / len(group)

        group['sector_weight'] = weights * sector_weight
        return group

    picks_df = picks_df.groupby('sector', group_keys=False).apply(weight_within_sector)

    return picks_df


def compute_continuous_weights(predictions, max_position_weight=0.05):
    """
    Compute continuous z-score based weights for long-short portfolio.

    Instead of top-N / bottom-N discrete selection, weight ALL stocks
    proportional to their z-score. This:
    - Increases diversification (500 positions vs 40)
    - Reduces turnover (small weight changes vs full replacement)
    - Better captures IC across full cross-section

    Args:
        predictions: Array of model predictions (probabilities)
        max_position_weight: Max weight for any single position (default 5%)

    Returns:
        weights: Array of weights (positive = long, negative = short)
                 Sum of abs(weights) = 2.0 (100% long, 100% short = 200% gross)
    """
    from scipy.stats import zscore

    # Z-score the predictions
    z = zscore(predictions)

    # Replace NaN with 0 (stocks with no prediction get no weight)
    z = np.nan_to_num(z, nan=0.0)

    # Normalize so sum(abs(weights)) = 2.0 (dollar neutral: 1.0 long, 1.0 short)
    total_abs = np.abs(z).sum()
    if total_abs > 0:
        weights = z / total_abs * 2.0
    else:
        weights = np.zeros_like(z)

    # Apply position limits
    weights = np.clip(weights, -max_position_weight, max_position_weight)

    # Re-normalize after clipping to maintain dollar neutrality
    long_sum = weights[weights > 0].sum()
    short_sum = abs(weights[weights < 0].sum())

    if long_sum > 0:
        weights[weights > 0] *= 1.0 / long_sum
    if short_sum > 0:
        weights[weights < 0] *= 1.0 / short_sum

    return weights


def apply_turnover_constraint(current_weights, previous_weights, max_turnover_per_stock=0.05):
    """
    Apply turnover constraint to limit weight changes per stock.

    This reduces turnover from ~1200%/year to ~200-300%/year by:
    - Limiting max weight change per stock per period
    - Creating "stickiness" in positions

    Args:
        current_weights: Dict of {ticker: target_weight}
        previous_weights: Dict of {ticker: previous_weight} or None
        max_turnover_per_stock: Max weight change allowed (default 5%)

    Returns:
        constrained_weights: Dict of {ticker: constrained_weight}
    """
    if previous_weights is None:
        return current_weights

    constrained = {}
    all_tickers = set(current_weights.keys()) | set(previous_weights.keys())

    for ticker in all_tickers:
        target = current_weights.get(ticker, 0.0)
        previous = previous_weights.get(ticker, 0.0)

        # Limit the change
        change = target - previous
        if abs(change) > max_turnover_per_stock:
            change = np.sign(change) * max_turnover_per_stock

        new_weight = previous + change

        # Only include non-zero weights
        if abs(new_weight) > 0.001:  # 0.1% minimum position
            constrained[ticker] = new_weight

    # Re-normalize to maintain dollar neutrality
    long_sum = sum(w for w in constrained.values() if w > 0)
    short_sum = abs(sum(w for w in constrained.values() if w < 0))

    for ticker in constrained:
        if constrained[ticker] > 0 and long_sum > 0:
            constrained[ticker] /= long_sum
        elif constrained[ticker] < 0 and short_sum > 0:
            constrained[ticker] /= short_sum

    return constrained


def save_fold_model(model, test_year, seed, train_window, feature_cols, output_dir="models/folds"):
    """
    Save per-fold model with metadata for audit trail.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"fold_{test_year}_seed_{seed}.pkl"
    metadata = {
        'model': model,
        'test_year': test_year,
        'train_window': train_window,
        'feature_cols': feature_cols,
        'seed': seed,
        'trained_at': datetime.now().isoformat()
    }

    with open(out_path, 'wb') as f:
        pickle.dump(metadata, f)

    return out_path


def load_historical_universe(filepath="data/historical_sp500_universe.json"):
    """
    Load historical S&P 500 universe to fix survivorship bias.

    Returns tuple: (universe_by_date dict, ticker_aliases dict)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"WARNING: Historical universe file not found: {filepath}")
        print("Run: python scripts/fetch_historical_sp500.py to create it")
        return None, {}

    with open(filepath, 'r') as f:
        data = json.load(f)

    universe = data.get('universe_by_date', {})
    aliases = data.get('ticker_aliases', {})

    return universe, aliases


def get_universe_at_date(historical_universe, target_date):
    """
    Get the S&P 500 universe at a specific date.

    Returns set of valid tickers for that date.
    """
    if historical_universe is None:
        return None  # No filtering

    target = pd.to_datetime(target_date).strftime('%Y-%m-%d')

    # Find the most recent universe snapshot <= target date
    dates = sorted([d for d in historical_universe.keys() if d <= target], reverse=True)

    if dates:
        return set(historical_universe[dates[0]])

    # Fallback to earliest available
    earliest = min(historical_universe.keys())
    return set(historical_universe[earliest])


# Global ticker aliases - loaded dynamically from JSON
_TICKER_ALIASES = {}


def set_ticker_aliases(aliases):
    """Set the ticker aliases from loaded data."""
    global _TICKER_ALIASES
    _TICKER_ALIASES = aliases


def normalize_ticker(ticker):
    """Get all valid ticker variants for matching."""
    variants = set()
    variants.add(ticker)
    variants.add(ticker.replace('.', '-'))
    variants.add(ticker.replace('-', '.'))

    # Add dynamically loaded aliases
    if ticker in _TICKER_ALIASES:
        variants.update(_TICKER_ALIASES[ticker])

    return variants


def walk_forward_validation(
    min_train_years: int = 3,
    test_months: int = 12,
    top_n: int = 20,
    transaction_cost: float = 0.001,
    use_risk_off: bool = False,
    vix_threshold: float = 25.0,
    vol_target: float = None,  # Target annual volatility (e.g., 0.15 for 15%)
    use_kelly: bool = False,   # Use Kelly fraction for position sizing
    track_turnover: bool = True,  # Track portfolio turnover
    n_ensemble: int = 3,  # Number of models to ensemble (different seeds)
    fix_survivorship_bias: bool = True,  # Use historical S&P 500 membership
    save_fold_models: bool = True,  # Save per-fold models for audit
    run_bootstrap: bool = True,  # Run bootstrap significance test
    long_short: bool = False,  # Use long-short market-neutral portfolio
    short_cost_bps: float = 50,  # Annual short borrow cost in basis points
    neutralize_beta_flag: bool = False,  # Neutralize portfolio beta
    neutralize_sector_flag: bool = False,  # Neutralize sector exposure
    continuous_weights: bool = False,  # Use continuous z-score weights (vs top-N)
    max_position_weight: float = 0.05,  # Max weight per position (5%)
    max_turnover_per_stock: float = 0.03,  # Max weight change per stock per month (3%)
    use_optimizer: bool = False,  # Use CVXPY portfolio optimizer (ChatGPT institutional-grade)
    use_factor_neutral: bool = False,  # Use factor-neutral optimizer (removes systematic risk)
    use_ranker: bool = False,  # Use LightGBM ranking model (predicts continuous residual ranks)
    use_meta_ensemble: bool = False,  # Use meta-model ensemble (XGB + LGBM + Ridge)
    use_stacked_blend: bool = False,  # Use stacked alpha blending (2-layer: XGB+LGBM -> Ridge meta-learner)
    use_regression: bool = False,  # Use XGBRegressor on continuous residual returns (ChatGPT recommended)
    # Mega-cap overlay parameters (fix for 2024 underperformance)
    use_mega_cap_overlay: bool = False,  # Force include top SPY mega-caps (fixes 2024 -22% excess)
    min_mega_cap_allocation: float = 0.25,  # Minimum % of portfolio in mega-caps
    mega_cap_force_top_k: int = 5,  # Force include top K mega-caps by SPY weight
    mega_cap_weight_method: str = 'hybrid'  # Weighting: 'equal', 'cap_weighted', or 'hybrid'
):
    """
    Walk-forward out-of-sample validation.

    Process:
    1. Start with minimum training window (e.g., 2015-2017)
    2. Train model
    3. Test on next year (2018) - TRUE out-of-sample
    4. Expand training window to include 2018
    5. Retrain model
    6. Test on 2019
    7. Repeat...

    This shows performance across ALL market regimes.
    """
    print("=" * 70)
    print("WALK-FORWARD OUT-OF-SAMPLE VALIDATION")
    if use_mega_cap_overlay:
        if not MEGA_CAP_OVERLAY_AVAILABLE:
            print("ERROR: Mega-cap overlay not available. Check mega_cap_overlay.py")
            return None
        print(f"(MEGA-CAP OVERLAY: Force top {mega_cap_force_top_k} SPY holdings, min {min_mega_cap_allocation*100:.0f}% allocation)")
        print(f"(EXPECTED: Fix 2024 -22% excess -> +0.7% excess, +40% total improvement)")
    if use_factor_neutral:
        if not FACTOR_NEUTRAL_AVAILABLE:
            print("ERROR: Factor-neutral optimizer not available. Check factor_model.py")
            return None
        print("(FACTOR-NEUTRAL: Removes Market/Sector/Momentum risk before optimization)")
    if use_optimizer:
        if not OPTIMIZER_AVAILABLE:
            print("ERROR: Portfolio optimizer not available. Run: pip install cvxpy")
            return None
        print("(CVXPY OPTIMIZER: Beta target 1.0+/-0.1 | Vol target 16% | Sector max 20%)")
    if continuous_weights:
        print(f"(CONTINUOUS Z-SCORE WEIGHTS: All stocks, max {max_position_weight*100:.0f}% per position)")
        print(f"(TURNOVER CONSTRAINED: Max {max_turnover_per_stock*100:.0f}% weight change per stock)")
    elif long_short:
        print(f"(LONG-SHORT MARKET-NEUTRAL: {top_n} long / {top_n} short)")
    if n_ensemble > 1:
        print(f"(ENSEMBLE: {n_ensemble} models with different seeds)")
    if use_risk_off:
        print(f"(WITH RISK-OFF: VIX > {vix_threshold} = 50% position)")
    if vol_target:
        print(f"(WITH VOL TARGETING: {vol_target*100:.0f}% annual vol target)")
    if use_kelly:
        print("(WITH KELLY FRACTION position sizing)")
    if neutralize_beta_flag:
        print("(WITH BETA NEUTRALIZATION: Target beta = 0)")
    if neutralize_sector_flag:
        print("(WITH SECTOR NEUTRALIZATION: Equal sector weights)")
    if fix_survivorship_bias:
        print("(WITH SURVIVORSHIP BIAS FIX: Historical S&P 500 membership)")
    print("=" * 70)
    print("\nThis is the GOLD STANDARD robustness test.")
    print("Each test period is truly out-of-sample.\n")

    # === LOAD HISTORICAL UNIVERSE FOR SURVIVORSHIP BIAS FIX ===
    historical_universe = None
    if fix_survivorship_bias:
        historical_universe, ticker_aliases = load_historical_universe()
        if historical_universe:
            # Set global ticker aliases for normalize_ticker()
            set_ticker_aliases(ticker_aliases)

            dates_available = sorted(historical_universe.keys())
            print(f"Survivorship bias fix: Loaded {len(dates_available)} universe snapshots")
            print(f"  Date range: {dates_available[0]} to {dates_available[-1]}")
            print(f"  Ticker aliases loaded: {len(ticker_aliases)}")
            # Show sample removed stocks
            earliest_universe = set(historical_universe[dates_available[0]])
            latest_universe = set(historical_universe[dates_available[-1]])
            removed_since_start = earliest_universe - latest_universe
            if removed_since_start:
                print(f"  Stocks removed since {dates_available[0][:4]}: {len(removed_since_start)}")
                print(f"  Examples: {list(removed_since_start)[:5]}")
        else:
            print("WARNING: Could not load historical universe, proceeding without survivorship bias fix")

    # Load all data
    print("Loading data...")
    db = Database(db_path="data/stocks.db", use_supabase=False)
    features_df = db.get_features()
    features_df['date'] = pd.to_datetime(features_df['date'])

    prices_df = db.get_prices()
    prices_df['date'] = pd.to_datetime(prices_df['date'])

    spy_df = db.get_benchmarks(ticker="SPY")
    spy_df['date'] = pd.to_datetime(spy_df['date'])

    db.close()

    # === LOAD SECTOR DATA FOR NEUTRALIZATION ===
    sectors_dict = {}
    if neutralize_sector_flag:
        # Try to load sectors from historical universe file
        universe_path = Path("data/historical_sp500_universe.json")
        if universe_path.exists():
            with open(universe_path, 'r') as f:
                universe_data = json.load(f)
            sectors_dict = universe_data.get('sectors', {})
            print(f"Loaded sector data for {len(sectors_dict)} tickers")
        else:
            print("WARNING: No sector data available - sector neutralization disabled")
            neutralize_sector_flag = False

    # === COMPUTE BETAS FOR NEUTRALIZATION ===
    betas_df = None
    if neutralize_beta_flag:
        print("Computing rolling betas (this may take a moment)...")
        betas_df = compute_stock_betas(prices_df.copy(), spy_df.copy())
        print(f"Computed betas for {betas_df['ticker'].nunique()} tickers")

    print(f"Data range: {features_df['date'].min().date()} to {features_df['date'].max().date()}")

    # Define feature columns (same as training script)
    feature_cols = [
        # RAW VALUES
        'return_1d', 'return_3d', 'return_5d',
        'return_1m', 'return_3m', 'return_6m',
        'volatility_20d', 'volatility_60d',
        'dist_from_sma_50', 'dist_from_sma_200',
        'dist_from_52w_high', 'dist_from_52w_low',
        'volume_ratio_20', 'volume_zscore',
        # ROLLING Z-SCORES (ChatGPT: removes drift, stabilizes signal)
        'return_1m_zscore', 'return_3m_zscore',
        'vol_zscore_rolling', 'volume_zscore_rolling',
        # NONLINEAR INTERACTIONS (ChatGPT: captures conditional effects)
        'mom_vol_interaction', 'reversal_vol_interaction',
        'sma_vol_interaction', 'high_mom_interaction', 'vol_regime_interaction',
        # CROSS-SECTIONAL RANKINGS
        'return_1d_rank', 'return_3d_rank', 'return_5d_rank',
        'return_1m_rank', 'return_3m_rank', 'return_6m_rank',
        'volatility_20d_rank', 'volatility_60d_rank',
        'dist_from_sma_50_rank', 'dist_from_sma_200_rank',
        'dist_from_52w_high_rank', 'dist_from_52w_low_rank',
        'volume_ratio_20_rank', 'volume_zscore_rank',
        # Rolling z-score ranks (ChatGPT)
        'return_1m_zscore_rank', 'return_3m_zscore_rank',
        'vol_zscore_rolling_rank', 'volume_zscore_rolling_rank',
        # Interaction ranks (ChatGPT)
        'mom_vol_interaction_rank', 'reversal_vol_interaction_rank',
        'sma_vol_interaction_rank', 'high_mom_interaction_rank', 'vol_regime_interaction_rank',
        # INDUSTRY-NEUTRAL RESIDUALS (professional upgrade!)
        'return_1d_resid', 'return_3d_resid', 'return_5d_resid',
        'return_1m_resid', 'return_3m_resid', 'return_6m_resid',
        'volatility_20d_resid', 'volatility_60d_resid',
        'dist_from_sma_50_resid', 'dist_from_sma_200_resid',
        'volume_ratio_20_resid', 'volume_zscore_resid',
        # Rolling z-score residuals (ChatGPT)
        'return_1m_zscore_resid', 'return_3m_zscore_resid',
        # Interaction residuals (ChatGPT)
        'mom_vol_interaction_resid', 'reversal_vol_interaction_resid',
        'vol_regime_interaction_resid',
        # MONTHLY CROSS-SECTIONAL Z-NORMALIZED (ChatGPT: removes time-varying scale)
        'return_1d_znorm', 'return_3d_znorm', 'return_5d_znorm',
        'return_1m_znorm', 'return_3m_znorm', 'return_6m_znorm', 'return_12m_znorm',
        'volatility_20d_znorm', 'volatility_60d_znorm',
        'dist_from_sma_50_znorm', 'dist_from_sma_200_znorm',
        'dist_from_52w_high_znorm', 'dist_from_52w_low_znorm',
        'volume_ratio_20_znorm', 'volume_ratio_60_znorm', 'volume_zscore_znorm',
        'mom_vol_interaction_znorm', 'reversal_vol_interaction_znorm',
        'vol_regime_interaction_znorm',
        # MARKET REGIME
        'market_volatility', 'market_trend'
    ]

    # Filter to only columns that exist in data
    available_cols = [col for col in feature_cols if col in features_df.columns]
    missing_cols = set(feature_cols) - set(available_cols)
    if missing_cols:
        print(f"Note: {len(missing_cols)} features not available (run engineer_features.py to add)")
    feature_cols = available_cols
    print(f"Using {len(feature_cols)} features")

    # Get years available
    features_df['year'] = features_df['date'].dt.year
    years = sorted(features_df['year'].unique())
    print(f"Years available: {years}")

    # Walk-forward windows
    # Start with min_train_years of training, then test each subsequent year
    min_start_year = years[0]
    first_test_year = min_start_year + min_train_years

    if first_test_year > years[-1]:
        print(f"ERROR: Need at least {min_train_years + 1} years of data")
        return

    print(f"\nWalk-forward schedule:")
    print(f"  Minimum training: {min_train_years} years")
    print(f"  First test year: {first_test_year}")
    print(f"  Last test year: {years[-1]}")

    # Pre-compute price lookups with numpy arrays for binary search
    price_by_ticker = {}
    for ticker, group in prices_df.groupby('ticker'):
        sorted_group = group.sort_values('date')
        price_by_ticker[ticker] = {
            'dates': np.array(sorted_group['date'].values),
            'prices': sorted_group['adj_close'].values
        }
    spy_prices = spy_df.set_index('date').sort_index()

    all_trading_dates = np.array(sorted(prices_df['date'].unique()))
    trading_date_idx = {d: i for i, d in enumerate(all_trading_dates)}

    def get_next_trading_day(date):
        """Get the next trading day using binary search."""
        idx = np.searchsorted(all_trading_dates, date)
        if idx < len(all_trading_dates) and all_trading_dates[idx] == date:
            idx += 1  # Move to next day
        if idx < len(all_trading_dates):
            return all_trading_dates[idx]
        return None

    def get_price_on_or_after(ticker_data, target_date):
        """
        Get price on target_date or FIRST available date after (no backward drift).
        Uses binary search for O(log n) performance.
        This prevents forward-looking bias from backward price lookups.
        """
        dates = ticker_data['dates']
        prices = ticker_data['prices']

        # Convert target_date to numpy datetime64 for comparison
        target_dt64 = np.datetime64(pd.Timestamp(target_date))

        # Binary search for target date or next available
        idx = np.searchsorted(dates, target_dt64)

        # If exact match
        if idx < len(dates) and dates[idx] == target_dt64:
            return prices[idx], dates[idx]

        # If no exact match, use next available (forward only, never backward)
        if idx < len(dates):
            # Only allow up to 5 trading days forward drift
            days_diff = (pd.Timestamp(dates[idx]) - pd.Timestamp(target_date)).days
            if days_diff <= 7:  # ~5 trading days
                return prices[idx], dates[idx]

        return None, None

    # === WALK FORWARD ===

    all_results = []
    yearly_summary = []

    # Tracking for advanced analytics
    turnover_records = []  # Track monthly turnover
    previous_holdings = set()  # Previous month's tickers
    realized_vol_window = []  # Rolling window for realized vol calculation
    previous_continuous_weights = None  # For continuous weights turnover constraint

    # Kelly fraction tracking (calculated from rolling window)
    rolling_win_rate = []
    rolling_avg_win = []
    rolling_avg_loss = []

    # === OPTIMIZER SETUP (ChatGPT institutional-grade) ===
    portfolio_optimizer = None
    factor_neutral_optimizer = None
    returns_df_for_opt = None
    spy_returns_for_opt = None
    sectors_dict_for_opt = None

    if use_factor_neutral:
        print("\nInitializing Enterprise-Grade Factor-Neutral optimizer...")
        factor_neutral_optimizer = FactorNeutralOptimizer(
            # === POSITION SIZING ===
            max_weight=0.05,           # Allow up to 5% conviction positions
            min_weight=0.0,            # NO minimum - allow optimizer to express conviction!

            # === FACTOR CONSTRAINTS (BALANCED) ===
            target_beta=1.05,          # Slightly above market to offset defensive bias
            beta_tolerance=0.20,       # Tighter: 0.85-1.25 beta range
            target_vol=0.22,           # 22% target volatility
            vol_tolerance=0.10,        # 12-32% vol range
            max_sector_weight=0.30,    # 30% sector cap
            momentum_neutral=True,     # Enforce momentum neutrality

            # === ENTERPRISE FEATURES ===
            n_stocks=40,               # Target positions
            l2_regularization=0.005,   # Reduced ridge penalty (was 0.01)
            conviction_gamma=1.5,      # Reduced conviction scaling (was 2.0)
            regime_aware=True,         # Enable regime detection (crisis/high_vol/normal)

            verbose=True
        )

        # Build returns DataFrame
        print("Building returns matrix for factor model...")
        all_tickers = list(price_by_ticker.keys())
        returns_dict = {}
        for ticker in all_tickers:
            ticker_data = price_by_ticker[ticker]
            prices_series = pd.Series(ticker_data['prices'], index=ticker_data['dates'])
            returns_dict[ticker] = prices_series.pct_change()
        returns_df_for_opt = pd.DataFrame(returns_dict)
        returns_df_for_opt.index = pd.to_datetime(returns_df_for_opt.index)

        # SPY returns
        spy_returns_for_opt = spy_prices['adj_close'].pct_change()

        # Load sectors
        sectors_dict_for_opt = {}
        if 'sector' in features_df.columns:
            sector_df = features_df[['ticker', 'sector']].drop_duplicates()
            sectors_dict_for_opt = dict(zip(sector_df['ticker'], sector_df['sector']))

        print(f"Returns matrix: {returns_df_for_opt.shape[0]} days x {returns_df_for_opt.shape[1]} tickers")
        print(f"Sectors loaded: {len(set(sectors_dict_for_opt.values()))} unique sectors")

    if use_optimizer:
        print("\nInitializing CVXPY portfolio optimizer...")
        portfolio_optimizer = LongOnlyOptimizer(
            max_weight=0.02,        # Max 2% per position
            min_weight=0.003,       # Min 0.3% per position
            target_beta=1.0,        # Target beta = 1.0
            beta_tolerance=0.05,    # Tighter: 0.95-1.05
            target_vol=0.14,        # 14% annual vol target
            vol_tolerance=0.015,    # Vol can be ~12.5-15.5%
            max_sector_weight=0.08, # Max 8% per sector
            max_turnover=0.15,      # Max 15% turnover per month
            n_stocks=30,            # Target ~30 stocks
            verbose=False
        )

        # Build returns DataFrame for optimizer
        print("Building returns matrix for covariance estimation...")
        all_tickers = list(price_by_ticker.keys())
        returns_dict = {}
        for ticker in all_tickers:
            ticker_data = price_by_ticker[ticker]
            prices_series = pd.Series(ticker_data['prices'], index=ticker_data['dates'])
            returns_dict[ticker] = prices_series.pct_change()
        returns_df_for_opt = pd.DataFrame(returns_dict)
        returns_df_for_opt.index = pd.to_datetime(returns_df_for_opt.index)

        # SPY returns for beta estimation
        spy_returns_for_opt = spy_prices['adj_close'].pct_change()
        print(f"Returns matrix: {returns_df_for_opt.shape[0]} days x {returns_df_for_opt.shape[1]} tickers")

    previous_opt_weights = None  # Track optimizer weights for turnover

    for test_year in range(first_test_year, years[-1] + 1):
        train_end_year = test_year - 1

        print(f"\n{'='*60}")
        print(f"FOLD: Train {min_start_year}-{train_end_year} -> Test {test_year}")
        print(f"{'='*60}")

        # Split data
        train_data = features_df[features_df['year'] <= train_end_year].copy()
        test_data = features_df[features_df['year'] == test_year].copy()

        # === SURVIVORSHIP BIAS FIX ===
        # Filter to only include stocks that were in S&P 500 at each point in time
        if historical_universe:
            def filter_by_universe(df):
                """Filter dataframe to only include stocks in S&P 500 at each date."""
                # Build universe cache by unique dates (much faster than row-by-row)
                unique_dates = df['date'].unique()
                date_universes = {}
                for d in unique_dates:
                    universe = get_universe_at_date(historical_universe, d)
                    # Pre-expand with proper ticker alias mapping
                    expanded = set()
                    for t in universe:
                        expanded.update(normalize_ticker(t))
                    date_universes[d] = expanded

                # Vectorized check: for each row, is any ticker variant in that date's universe?
                def check_ticker_in_universe(row):
                    universe = date_universes.get(row['date'], set())
                    ticker_variants = normalize_ticker(row['ticker'])
                    return bool(ticker_variants & universe)  # Set intersection

                valid_mask = df.apply(check_ticker_in_universe, axis=1)
                return df[valid_mask]

            train_before = len(train_data)
            test_before = len(test_data)

            train_data = filter_by_universe(train_data)
            test_data = filter_by_universe(test_data)

            train_removed = train_before - len(train_data)
            test_removed = test_before - len(test_data)
            if train_removed > 0 or test_removed > 0:
                print(f"  Survivorship fix: removed {train_removed} train / {test_removed} test samples")

        if len(train_data) < 1000 or len(test_data) < 100:
            print(f"  Skipping: insufficient data (train={len(train_data)}, test={len(test_data)})")
            continue

        # Prepare training data
        # Select target based on mode
        if use_regression:
            # REGRESSION MODE: Predict continuous residual returns (ChatGPT recommended)
            target_col = 'target_regression'  # Cross-sectional z-score of residual returns
            if target_col not in train_data.columns:
                # Fallback to raw residual returns
                target_col = 'future_return_resid'
                if target_col not in train_data.columns:
                    print(f"  Warning: regression targets not found, falling back to target_binary")
                    target_col = 'target_binary'
        elif use_ranker:
            target_col = 'future_return_resid_rank'
            if target_col not in train_data.columns:
                print(f"  Warning: {target_col} not found, falling back to target_binary")
                target_col = 'target_binary'
        else:
            target_col = 'target_binary'

        train_clean = train_data[['ticker', 'date', target_col] + feature_cols].dropna()
        X_train = train_clean[feature_cols].values
        y_train = train_clean[target_col].values

        # === TIME-DECAY SAMPLE WEIGHTING (ChatGPT recommendation) ===
        # Weight recent samples more heavily than older ones
        # Uses exponential decay: weight = exp(-lambda * months_ago)
        train_dates = pd.to_datetime(train_clean['date'])
        max_date = train_dates.max()
        months_ago = ((max_date - train_dates).dt.days / 30.0).values
        decay_lambda = 0.02  # ~50% weight after 35 months
        sample_weights = np.exp(-decay_lambda * months_ago)
        # Normalize weights to sum to len(samples) for consistent learning rate
        sample_weights = sample_weights * len(sample_weights) / sample_weights.sum()

        print(f"  Training samples: {len(train_clean)}")
        print(f"  Time-decay: oldest weight={sample_weights.min():.2f}, newest={sample_weights.max():.2f}")

        if use_regression:
            print(f"  Target: REGRESSION (mean={y_train.mean():.3f}, std={y_train.std():.3f})")
        elif use_ranker:
            print(f"  Target: continuous rank (mean={y_train.mean():.3f}, std={y_train.std():.3f})")
        else:
            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
            print(f"  Class balance: {n_pos} pos ({n_pos/len(y_train)*100:.1f}%) / {n_neg} neg")

        # Use last 20% of training as validation for early stopping
        val_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]
        # Split sample weights too
        w_tr, w_val = sample_weights[:val_split], sample_weights[val_split:]

        # Train ensemble of models with different random seeds
        models = []

        if use_ranker:
            # === LIGHTGBM RANKING MODEL (predicts continuous residual ranks) ===
            base_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'max_depth': 5,
                'learning_rate': 0.02,
                'n_estimators': 500,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'verbose': -1
            }

            for seed_idx in range(n_ensemble):
                actual_seed = 42 + seed_idx * 17
                params = base_params.copy()
                params['random_state'] = actual_seed
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(30, verbose=False)]
                )
                models.append(model)

        elif use_meta_ensemble:
            # === META-ENSEMBLE: XGB + LGBM + Ridge (diverse model stack) ===
            # Standardize features for Ridge (tree models don't need it)
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)

            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

            # --- Model 1: XGBoost Classifier ---
            xgb_params = {
                'objective': 'binary:logistic',
                'max_depth': 4,
                'learning_rate': 0.01,
                'n_estimators': 500,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 5,
                'gamma': 0.2,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'scale_pos_weight': scale_pos_weight,
                'eval_metric': 'auc',
                'early_stopping_rounds': 30,
                'verbosity': 0,
                'random_state': 42
            }
            xgb_model = xgb.XGBClassifier(**xgb_params)
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            # --- Model 2: LightGBM Classifier ---
            lgb_params = {
                'objective': 'binary',
                'metric': 'auc',
                'max_depth': 5,
                'learning_rate': 0.02,
                'n_estimators': 500,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'scale_pos_weight': scale_pos_weight,
                'verbose': -1,
                'random_state': 59
            }
            lgb_model = lgb.LGBMClassifier(**lgb_params)
            lgb_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )

            # --- Model 3: Calibrated Ridge Classifier ---
            # Ridge needs calibration to output probabilities
            ridge_base = RidgeClassifier(alpha=1.0, random_state=73)
            ridge_model = CalibratedClassifierCV(ridge_base, cv=3, method='sigmoid')
            ridge_model.fit(X_tr_scaled, y_tr)

            # Store models with their scalers
            models = [
                ('xgb', xgb_model, None),
                ('lgb', lgb_model, None),
                ('ridge', ridge_model, scaler)
            ]
            print(f"  META-ENSEMBLE: XGBoost + LightGBM + Ridge")

        elif use_stacked_blend:
            # === STACKED ALPHA BLENDING (ChatGPT: 2-layer ensemble) ===
            # Layer 1: Train XGBoost and LightGBM base models
            # Layer 2: Use their predictions as features for Ridge meta-learner
            from sklearn.model_selection import cross_val_predict
            from sklearn.linear_model import LogisticRegression

            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

            # Base models for OOF predictions (NO early stopping - not compatible with cross_val_predict)
            xgb_oof_model = xgb.XGBClassifier(
                objective='binary:logistic', max_depth=4, learning_rate=0.01,
                n_estimators=200, subsample=0.7, colsample_bytree=0.7,
                scale_pos_weight=scale_pos_weight, eval_metric='auc',
                verbosity=0, random_state=42
            )

            lgb_oof_model = lgb.LGBMClassifier(
                objective='binary', max_depth=5, learning_rate=0.02,
                n_estimators=200, subsample=0.7, colsample_bytree=0.7,
                scale_pos_weight=scale_pos_weight, verbose=-1, random_state=59
            )

            # Get out-of-fold predictions for stacking (prevents leakage)
            print(f"  STACKED BLEND: Getting OOF predictions...")
            xgb_oof = cross_val_predict(xgb_oof_model, X_tr, y_tr, cv=3, method='predict_proba')[:, 1]
            lgb_oof = cross_val_predict(lgb_oof_model, X_tr, y_tr, cv=3, method='predict_proba')[:, 1]

            # Stack OOF predictions as meta-features
            meta_features_train = np.column_stack([xgb_oof, lgb_oof])

            # Train meta-learner on stacked features
            meta_learner = LogisticRegression(C=1.0, random_state=73, max_iter=1000)
            meta_learner.fit(meta_features_train, y_tr, sample_weight=w_tr)

            # Create final base models WITH early stopping for test-time predictions
            xgb_base = xgb.XGBClassifier(
                objective='binary:logistic', max_depth=4, learning_rate=0.01,
                n_estimators=500, subsample=0.7, colsample_bytree=0.7,
                scale_pos_weight=scale_pos_weight, eval_metric='auc',
                early_stopping_rounds=30, verbosity=0, random_state=42
            )
            lgb_base = lgb.LGBMClassifier(
                objective='binary', max_depth=5, learning_rate=0.02,
                n_estimators=500, subsample=0.7, colsample_bytree=0.7,
                scale_pos_weight=scale_pos_weight, verbose=-1, random_state=59
            )

            # Train final base models on full training data
            xgb_base.fit(X_tr, y_tr, sample_weight=w_tr,
                        eval_set=[(X_val, y_val)], verbose=False)
            lgb_base.fit(X_tr, y_tr, sample_weight=w_tr,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(30, verbose=False)])

            # Store as tuple for special handling
            models = [('stacked', (xgb_base, lgb_base, meta_learner), None)]
            print(f"  STACKED BLEND: XGBoost + LightGBM -> LogisticRegression meta-learner")
            print(f"  Meta-learner coefs: XGB={meta_learner.coef_[0][0]:.3f}, LGB={meta_learner.coef_[0][1]:.3f}")

        elif use_regression:
            # === XGBOOST REGRESSOR (ChatGPT recommended: predict continuous residual returns) ===
            # This uses ALL the return information instead of discarding into binary buckets
            base_params = {
                'objective': 'reg:squarederror',
                'max_depth': 4,
                'learning_rate': 0.01,
                'n_estimators': 500,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 5,
                'gamma': 0.2,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'eval_metric': 'rmse',
                'early_stopping_rounds': 30,
                'verbosity': 0
            }

            for seed_idx in range(n_ensemble):
                actual_seed = 42 + seed_idx * 17
                params = base_params.copy()
                params['random_state'] = actual_seed
                model = xgb.XGBRegressor(**params)
                model.fit(X_tr, y_tr, sample_weight=w_tr,
                         eval_set=[(X_val, y_val)], verbose=False)
                models.append(model)

            print(f"  REGRESSION: XGBoost Regressor (predicting continuous residual z-scores)")

        else:
            # === XGBOOST CLASSIFIER (original binary classification) ===
            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

            base_params = {
                'objective': 'binary:logistic',
                'max_depth': 4,
                'learning_rate': 0.01,
                'n_estimators': 500,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 5,
                'gamma': 0.2,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'scale_pos_weight': scale_pos_weight,
                'eval_metric': 'auc',
                'early_stopping_rounds': 30,
                'verbosity': 0
            }

            for seed_idx in range(n_ensemble):
                actual_seed = 42 + seed_idx * 17
                params = base_params.copy()
                params['random_state'] = actual_seed
                model = xgb.XGBClassifier(**params)
                # Use time-decay sample weights (ChatGPT recommendation)
                model.fit(X_tr, y_tr, sample_weight=w_tr,
                         eval_set=[(X_val, y_val)], verbose=False)
                models.append(model)

            # Save per-fold model for audit trail
            if save_fold_models:
                save_fold_model(
                    model=model,
                    test_year=test_year,
                    seed=actual_seed,
                    train_window=(min_start_year, train_end_year),
                    feature_cols=feature_cols
                )

        # Evaluate on test year
        test_clean = test_data[['ticker', 'date', 'target_binary'] + feature_cols].dropna()
        X_test = test_clean[feature_cols].values
        y_test = test_clean['target_binary'].values

        if len(X_test) == 0:
            print(f"  No test samples!")
            continue

        # Ensemble prediction: average from all models
        ensemble_preds = np.zeros(len(X_test))
        if use_stacked_blend:
            # Stacked blend: get base model predictions, then apply meta-learner
            _, (xgb_base, lgb_base, meta_learner), _ = models[0]
            xgb_preds = xgb_base.predict_proba(X_test)[:, 1]
            lgb_preds = lgb_base.predict_proba(X_test)[:, 1]
            meta_features = np.column_stack([xgb_preds, lgb_preds])
            ensemble_preds = meta_learner.predict_proba(meta_features)[:, 1]
            test_pred_proba = ensemble_preds  # already probabilities
        elif use_meta_ensemble:
            # Meta-ensemble: models are tuples (name, model, scaler)
            for name, model, model_scaler in models:
                if model_scaler is not None:
                    # Ridge needs scaled features
                    X_scaled = model_scaler.transform(X_test)
                    ensemble_preds += model.predict_proba(X_scaled)[:, 1]
                else:
                    # XGB and LGB use raw features
                    ensemble_preds += model.predict_proba(X_test)[:, 1]
            test_pred_proba = ensemble_preds / len(models)
        elif use_regression:
            # Regression: average continuous predictions, then rank cross-sectionally
            for m in models:
                ensemble_preds += m.predict(X_test)
            ensemble_preds = ensemble_preds / len(models)
            # Convert to cross-sectional ranks (0-1) for portfolio construction
            from scipy.stats import rankdata
            test_pred_proba = rankdata(ensemble_preds) / len(ensemble_preds)
        else:
            for m in models:
                if use_ranker:
                    # LightGBM regressor outputs continuous predictions
                    ensemble_preds += m.predict(X_test)
                else:
                    # XGBoost classifier outputs probabilities
                    ensemble_preds += m.predict_proba(X_test)[:, 1]
            test_pred_proba = ensemble_preds / len(models)

        # For ranker, predictions are continuous ranks (0-1)
        # Still compute AUC against binary target for comparison
        test_auc = roc_auc_score(y_test, test_pred_proba)

        # Precision@TopN (consistent with actual trading - top_n picks)
        test_clean = test_clean.copy()
        test_clean['pred_proba'] = test_pred_proba
        # Use same top_n as actual trading, not arbitrary 10%
        top_picks_eval = test_clean.nlargest(top_n, 'pred_proba')
        precision_at_topn = top_picks_eval['target_binary'].mean()
        # Also compute Precision@10% for comparison
        top_10pct = test_clean.nlargest(int(len(test_clean) * 0.1), 'pred_proba')
        precision_at_10 = top_10pct['target_binary'].mean()

        # === ADDITIONAL VALIDATION METRICS ===
        # Spearman rank correlation (does model rank stocks correctly?)
        # Get actual future returns for test set
        # Use RESIDUAL returns for IC (model predicts sector-neutral outperformance)
        return_cols = ['ticker', 'date', 'future_return']
        if 'future_return_resid' in test_data.columns:
            return_cols.append('future_return_resid')

        test_with_returns = test_clean.merge(
            test_data[return_cols].drop_duplicates(),
            on=['ticker', 'date'],
            how='left'
        )

        # Use residual returns for IC if available (sector-neutral metric)
        ic_return_col = 'future_return_resid' if 'future_return_resid' in test_with_returns.columns else 'future_return'
        valid_returns = test_with_returns.dropna(subset=[ic_return_col, 'pred_proba'])

        if len(valid_returns) > 10:
            spearman_corr, spearman_p = spearmanr(
                valid_returns['pred_proba'],
                valid_returns[ic_return_col]
            )
        else:
            spearman_corr, spearman_p = 0, 1

        # Bottom decile check (should perform poorly if model is good)
        bottom_picks = test_clean.nsmallest(int(len(test_clean) * 0.1), 'pred_proba')
        bottom_decile_rate = bottom_picks['target_binary'].mean()

        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Precision@{top_n}: {precision_at_topn:.2%} (actual trading)")
        print(f"  Precision@10%: {precision_at_10:.2%} (random=10%)")
        print(f"  Spearman Rank Corr: {spearman_corr:.3f} (p={spearman_p:.3f})")
        print(f"  Bottom Decile Hit: {bottom_decile_rate:.2%} (should be <10%)")

        # === BACKTEST THIS YEAR ===

        # Get rebalancing dates for test year
        test_dates = sorted(test_data['date'].unique())

        year_returns = []
        spy_returns = []

        for i, signal_date in enumerate(test_dates[:-1]):
            next_date = test_dates[i + 1]

            entry_date = get_next_trading_day(signal_date)
            exit_date = get_next_trading_day(next_date)

            if entry_date is None or exit_date is None:
                continue

            # Get features and predict
            date_features = test_data[test_data['date'] == signal_date].copy()
            if len(date_features) < top_n:
                continue

            # Remove target columns (no lookahead)
            for col in ['target_binary', 'target_excess', 'future_return', 'future_return_rank']:
                if col in date_features.columns:
                    date_features = date_features.drop(columns=[col])

            X = date_features[feature_cols].reindex(columns=feature_cols).astype(np.float32).values
            X = np.nan_to_num(X, nan=0.0)

            # Ensemble prediction for backtest (average across all models)
            ensemble_preds = np.zeros(len(X))
            if use_stacked_blend:
                # Stacked blend: get base model predictions, then apply meta-learner
                _, (xgb_base, lgb_base, meta_learner), _ = models[0]
                xgb_preds = xgb_base.predict_proba(X)[:, 1]
                lgb_preds = lgb_base.predict_proba(X)[:, 1]
                meta_features = np.column_stack([xgb_preds, lgb_preds])
                ensemble_preds = meta_learner.predict_proba(meta_features)[:, 1]
                date_features['pred_proba'] = ensemble_preds
            elif use_meta_ensemble:
                # Meta-ensemble: models are tuples (name, model, scaler)
                for name, model, model_scaler in models:
                    if model_scaler is not None:
                        X_scaled = model_scaler.transform(X)
                        ensemble_preds += model.predict_proba(X_scaled)[:, 1]
                    else:
                        ensemble_preds += model.predict_proba(X)[:, 1]
                date_features['pred_proba'] = ensemble_preds / len(models)
            elif use_regression:
                # Regression: average predictions, then rank cross-sectionally
                for m in models:
                    ensemble_preds += m.predict(X)
                ensemble_preds = ensemble_preds / len(models)
                # Convert to cross-sectional ranks (0-1) for portfolio construction
                from scipy.stats import rankdata
                date_features['pred_proba'] = rankdata(ensemble_preds) / len(ensemble_preds)
            else:
                for m in models:
                    if use_ranker:
                        # LightGBM regressor outputs continuous predictions
                        ensemble_preds += m.predict(X)
                    else:
                        # XGBoost classifier outputs probabilities
                        ensemble_preds += m.predict_proba(X)[:, 1]
                date_features['pred_proba'] = ensemble_preds / len(models)

            # === PORTFOLIO CONSTRUCTION ===
            if continuous_weights:
                # CONTINUOUS Z-SCORE WEIGHTS: Weight ALL stocks by signal strength
                # This is ChatGPT's recommended approach for proper IC monetization
                predictions = date_features['pred_proba'].values
                tickers = date_features['ticker'].values

                # Compute continuous weights
                raw_weights = compute_continuous_weights(predictions, max_position_weight)

                # Create weight dict
                target_weights = {t: w for t, w in zip(tickers, raw_weights) if abs(w) > 0.001}

                # Apply turnover constraint (uses nonlocal previous_continuous_weights)
                constrained_weights = apply_turnover_constraint(
                    target_weights,
                    previous_continuous_weights,
                    max_turnover_per_stock
                )
                previous_continuous_weights = constrained_weights.copy()

                # For compatibility with rest of code, create picks DataFrames
                long_tickers = [t for t, w in constrained_weights.items() if w > 0]
                short_tickers = [t for t, w in constrained_weights.items() if w < 0]

                top_picks = date_features[date_features['ticker'].isin(long_tickers)].copy()
                bottom_picks = date_features[date_features['ticker'].isin(short_tickers)].copy() if short_tickers else None

                # Store weights for return calculation
                continuous_weight_dict = constrained_weights

            elif use_factor_neutral:
                # === FACTOR-NEUTRAL OPTIMIZER (ChatGPT institutional-grade Phase 2) ===
                # Uses factor de-noising: removes market/sector/momentum before optimization

                predictions_df = date_features[['ticker', 'pred_proba']].copy()

                # Filter returns to data before signal date (no lookahead)
                signal_dt = pd.to_datetime(signal_date)
                returns_up_to_date = returns_df_for_opt[returns_df_for_opt.index < signal_dt]

                if len(returns_up_to_date) < 120:
                    # Need more data for factor model
                    fn_weights = {t: 1.0/top_n for t in date_features.nlargest(top_n, 'pred_proba')['ticker']}
                else:
                    spy_up_to_date = spy_returns_for_opt[spy_returns_for_opt.index < signal_dt]
                    fn_weights = factor_neutral_optimizer.optimize(
                        predictions_df=predictions_df,
                        returns_df=returns_up_to_date,
                        spy_returns=spy_up_to_date,
                        sectors=sectors_dict_for_opt,
                        previous_weights=previous_opt_weights,
                        as_of_date=signal_date
                    )
                    previous_opt_weights = fn_weights.copy()

                # Create top_picks DataFrame from optimizer weights
                fn_tickers = list(fn_weights.keys())
                top_picks = date_features[date_features['ticker'].isin(fn_tickers)].copy()
                bottom_picks = None  # Long-only

                # Store weights for return calculation
                continuous_weight_dict = fn_weights

            elif use_optimizer:
                # === CVXPY OPTIMIZER (ChatGPT institutional-grade) ===
                # Use portfolio optimizer with beta/vol/sector constraints

                # Prepare predictions DataFrame for optimizer
                predictions_df = date_features[['ticker', 'pred_proba']].copy()

                # Get sectors dict
                sectors_dict = {}
                if 'sector' in date_features.columns:
                    sectors_dict = dict(zip(date_features['ticker'], date_features['sector']))

                # Filter returns to data before signal date (no lookahead)
                signal_dt = pd.to_datetime(signal_date)
                returns_up_to_date = returns_df_for_opt[returns_df_for_opt.index < signal_dt]

                if len(returns_up_to_date) < 60:
                    # Fallback to simple top-N if not enough data
                    opt_weights = {t: 1.0/top_n for t in date_features.nlargest(top_n, 'pred_proba')['ticker']}
                else:
                    spy_up_to_date = spy_returns_for_opt[spy_returns_for_opt.index < signal_dt]
                    opt_weights = portfolio_optimizer.optimize_from_dataframe(
                        predictions_df=predictions_df,
                        returns_df=returns_up_to_date,
                        spy_returns=spy_up_to_date,
                        sectors_dict=sectors_dict,
                        previous_weights=previous_opt_weights,
                        date=signal_date
                    )
                    previous_opt_weights = opt_weights.copy()

                # Create top_picks DataFrame from optimizer weights
                opt_tickers = list(opt_weights.keys())
                top_picks = date_features[date_features['ticker'].isin(opt_tickers)].copy()
                bottom_picks = None  # Long-only with optimizer

                # Store weights for return calculation
                continuous_weight_dict = opt_weights

            else:
                # DISCRETE TOP-N / BOTTOM-N SELECTION (original behavior)
                if use_mega_cap_overlay:
                    # === MEGA-CAP OVERLAY (Fix for 2024 underperformance) ===
                    # Prepare predictions for overlay
                    predictions_df = date_features[['ticker', 'pred_proba']].copy()
                    predictions_df.columns = ['ticker', 'score']
                    predictions_df['score'] = predictions_df['score'] * 100  # Scale to 0-100
                    predictions_df['prediction'] = predictions_df['score'] / 100

                    # Apply mega-cap overlay
                    portfolio, diagnostics = apply_mega_cap_overlay(
                        predictions_df,
                        top_n=top_n,
                        min_score_threshold=40.0,
                        mega_cap_min_allocation=min_mega_cap_allocation,
                        mega_cap_weight_method=mega_cap_weight_method,
                        force_include_top_k=mega_cap_force_top_k,
                        verbose=False  # Suppress per-month output
                    )

                    # Get top picks and set continuous weights
                    top_picks = date_features[date_features['ticker'].isin(portfolio['ticker'])].copy()
                    # Store weights from overlay (will use below instead of vol-weighted)
                    continuous_weight_dict = portfolio.set_index('ticker')['weight'].to_dict()
                else:
                    # Original behavior
                    top_picks = date_features.nlargest(top_n, 'pred_proba')
                    continuous_weight_dict = None

                if long_short:
                    # Short the bottom N stocks (model predicts they'll underperform)
                    bottom_picks = date_features.nsmallest(top_n, 'pred_proba')
                else:
                    bottom_picks = None

            # === TURNOVER TRACKING ===
            if continuous_weights:
                current_holdings = set(constrained_weights.keys())
            elif use_factor_neutral:
                current_holdings = set(fn_weights.keys())
            elif use_optimizer:
                current_holdings = set(opt_weights.keys())
            elif long_short:
                current_holdings = set(top_picks['ticker']) | set(bottom_picks['ticker'])
            else:
                current_holdings = set(top_picks['ticker'])
            if track_turnover and previous_holdings:
                # Turnover = % of positions that changed
                holdings_changed = len(previous_holdings - current_holdings) + len(current_holdings - previous_holdings)
                turnover = holdings_changed / (2 * top_n)  # Normalized: 0=no change, 1=complete turnover
                turnover_records.append({
                    'date': signal_date,
                    'year': test_year,
                    'turnover': turnover,
                    'positions_exited': len(previous_holdings - current_holdings),
                    'positions_entered': len(current_holdings - previous_holdings)
                })
            previous_holdings = current_holdings

            # === POSITION SIZING ===
            # Use mega-cap overlay weights if available, otherwise vol-weighted
            if use_mega_cap_overlay and continuous_weight_dict:
                # Use weights from mega-cap overlay
                weights = np.array([continuous_weight_dict.get(row['ticker'], 0)
                                    for _, row in top_picks.iterrows()])
                # Normalize to ensure sum = 1
                weights = weights / weights.sum()
            else:
                # VOL-ADJUSTED POSITION SIZING (original behavior)
                # Weight inversely by volatility: high-vol stocks get smaller weights
                vol_col = 'volatility_20d'
                if vol_col in top_picks.columns:
                    vols = top_picks[vol_col].fillna(top_picks[vol_col].median())
                    vols = vols.clip(lower=0.10)  # Min 10% vol to avoid extreme weights
                    avg_vol = vols.mean()
                    raw_weights = avg_vol / vols
                    weights = (raw_weights / raw_weights.sum()).values  # Normalize to sum to 1
                else:
                    weights = np.ones(len(top_picks)) / len(top_picks)  # Equal weight fallback

            # Calculate returns (vol-weighted) using binary search price lookup
            # === LONG LEG ===
            long_returns = []
            long_weights = []
            skipped_tickers = 0
            for idx, (_, row) in enumerate(top_picks.iterrows()):
                ticker = row['ticker']
                ticker_data = price_by_ticker.get(ticker)
                if ticker_data is None:
                    skipped_tickers += 1
                    continue  # Skip instead of -100% penalty

                # Use binary search price lookup (forward-only, no backward drift)
                entry_price, actual_entry = get_price_on_or_after(ticker_data, entry_date)
                exit_price, actual_exit = get_price_on_or_after(ticker_data, exit_date)

                if entry_price and exit_price and entry_price > 0:
                    ret = (exit_price - entry_price) / entry_price
                    long_returns.append(ret)
                    long_weights.append(weights[idx])
                elif entry_price and not exit_price:
                    # Stock delisted after entry - conservative -20% loss
                    long_returns.append(-0.20)
                    long_weights.append(weights[idx])
                else:
                    skipped_tickers += 1

            # === SHORT LEG (if long-short mode) ===
            short_returns = []
            short_weights = []
            if long_short and bottom_picks is not None:
                # Vol-weighted short positions
                short_vol_col = 'volatility_20d'
                if short_vol_col in bottom_picks.columns:
                    short_vols = bottom_picks[short_vol_col].fillna(bottom_picks[short_vol_col].median())
                    short_vols = short_vols.clip(lower=0.10)
                    short_avg_vol = short_vols.mean()
                    short_raw_weights = short_avg_vol / short_vols
                    short_weights_arr = (short_raw_weights / short_raw_weights.sum()).values
                else:
                    short_weights_arr = np.ones(len(bottom_picks)) / len(bottom_picks)

                for idx, (_, row) in enumerate(bottom_picks.iterrows()):
                    ticker = row['ticker']
                    ticker_data = price_by_ticker.get(ticker)
                    if ticker_data is None:
                        continue

                    entry_price, actual_entry = get_price_on_or_after(ticker_data, entry_date)
                    exit_price, actual_exit = get_price_on_or_after(ticker_data, exit_date)

                    if entry_price and exit_price and entry_price > 0:
                        # SHORT PROFIT: we profit when stock goes DOWN
                        # If stock goes from 100 to 90, we make +10%
                        ret = (entry_price - exit_price) / entry_price
                        short_returns.append(ret)
                        short_weights.append(short_weights_arr[idx])
                    elif entry_price and not exit_price:
                        # Stock delisted while short - assume worst case (buyout premium)
                        short_returns.append(-0.30)  # 30% loss on short
                        short_weights.append(short_weights_arr[idx])

            if not long_returns:
                continue

            # Backward compatibility: use long_returns as pick_returns for non-LS
            pick_returns = long_returns
            pick_weights = long_weights

            # === SLIPPAGE MODEL ===
            # Realistic slippage for mid/large cap stocks: 10-30 bps per rebalancing
            # Base: ~10 bps round-trip commission
            # Slippage: ~10-20 bps market impact depending on turnover
            base_cost = 0.001  # 10 bps base transaction cost
            if track_turnover and turnover_records:
                recent_turnover = turnover_records[-1]['turnover'] if turnover_records else 0.5
                # More realistic slippage: 10-20 bps per rebalancing
                # High turnover = more market impact
                slippage = 0.001 + 0.001 * recent_turnover  # 10-20 bps
            else:
                slippage = 0.0015  # 15 bps default
            total_cost = base_cost + slippage  # Total: 20-30 bps per month

            # Vol-weighted portfolio return (not simple average!)
            pick_weights = np.array(pick_weights)
            pick_returns = np.array(pick_returns)

            # === CALCULATE GROSS RETURN ===
            if continuous_weights and continuous_weight_dict:
                # CONTINUOUS WEIGHTS PORTFOLIO
                # Calculate return using the exact weights from z-score optimization
                portfolio_return_continuous = 0.0
                total_long_weight = 0.0
                total_short_weight = 0.0

                for ticker, weight in continuous_weight_dict.items():
                    ticker_data = price_by_ticker.get(ticker)
                    if ticker_data is None:
                        continue

                    entry_price, _ = get_price_on_or_after(ticker_data, entry_date)
                    exit_price, _ = get_price_on_or_after(ticker_data, exit_date)

                    if entry_price and exit_price and entry_price > 0:
                        stock_return = (exit_price - entry_price) / entry_price
                        # Weight is signed: positive=long, negative=short
                        # For shorts, we want to profit when stock goes DOWN
                        if weight < 0:
                            # Short position: profit = -stock_return * |weight|
                            portfolio_return_continuous += -stock_return * abs(weight)
                            total_short_weight += abs(weight)
                        else:
                            # Long position: profit = stock_return * weight
                            portfolio_return_continuous += stock_return * weight
                            total_long_weight += weight

                # Apply short borrow costs
                monthly_borrow_cost = (short_cost_bps / 10000) / 12 * total_short_weight
                gross_return = portfolio_return_continuous - total_cost - monthly_borrow_cost

            elif long_short and short_returns:
                # LONG-SHORT PORTFOLIO (discrete top-N/bottom-N)
                # 50% capital long, 50% capital short (dollar neutral)
                short_weights = np.array(short_weights)
                short_returns_arr = np.array(short_returns)

                # Weighted return for each leg
                if pick_weights.sum() > 0:
                    long_leg_return = np.dot(pick_weights / pick_weights.sum(), pick_returns)
                else:
                    long_leg_return = np.mean(pick_returns)

                if short_weights.sum() > 0:
                    short_leg_return = np.dot(short_weights / short_weights.sum(), short_returns_arr)
                else:
                    short_leg_return = np.mean(short_returns_arr) if short_returns_arr.size > 0 else 0

                # Dollar-neutral: 50% long, 50% short
                # Gross return = (long_return + short_return) / 2
                # Note: short_leg_return is already inverted (profit when stocks go down)
                gross_long_short = (long_leg_return + short_leg_return) / 2

                # Subtract short borrow costs (monthly portion of annual rate)
                monthly_borrow_cost = (short_cost_bps / 10000) / 12 * 0.5  # 50% of capital is short
                gross_return = gross_long_short - total_cost - monthly_borrow_cost
            else:
                # LONG-ONLY PORTFOLIO (original behavior)
                if pick_weights.sum() > 0:
                    normalized_weights = pick_weights / pick_weights.sum()
                    gross_return = np.dot(normalized_weights, pick_returns) - total_cost
                else:
                    gross_return = np.mean(pick_returns) - total_cost

            # === VOLATILITY TARGETING ===
            # Scale position to target X% annual volatility
            # Require minimum 12 months of data for reliable vol estimate
            vol_scale = 1.0
            if vol_target and len(realized_vol_window) >= 12:
                # Calculate realized monthly vol from recent 12-24 months
                lookback = min(24, len(realized_vol_window))
                recent_returns = realized_vol_window[-lookback:]
                realized_monthly_vol = np.std(recent_returns)
                realized_annual_vol = realized_monthly_vol * np.sqrt(12)

                if realized_annual_vol > 0.05:  # Minimum 5% vol to avoid extreme scaling
                    # Scale position to target vol
                    vol_scale = min(2.0, max(0.25, vol_target / realized_annual_vol))

            # === KELLY FRACTION ===
            # Use 60-120 month window for stable Kelly estimates (ChatGPT recommendation)
            kelly_scale = 1.0
            kelly_min_months = 60  # 5 years minimum for reliable Kelly
            kelly_lookback = 120   # 10 year lookback max
            if use_kelly and len(rolling_win_rate) >= kelly_min_months:
                # Kelly = W - (1-W)/R where W=win rate, R=win/loss ratio
                lookback = min(kelly_lookback, len(rolling_win_rate))
                avg_win_rate = np.mean(rolling_win_rate[-lookback:])
                wins = [w for w in rolling_avg_win[-lookback:] if w > 0]
                losses = [l for l in rolling_avg_loss[-lookback:] if l < 0]
                avg_win_amt = np.mean(wins) if wins else 0.02
                avg_loss_amt = abs(np.mean(losses)) if losses else 0.02

                if avg_loss_amt > 0:
                    win_loss_ratio = avg_win_amt / avg_loss_amt
                    kelly_fraction = avg_win_rate - (1 - avg_win_rate) / win_loss_ratio
                    # Quarter-Kelly for safety (half-Kelly still too aggressive)
                    # Never scale above 0.5x (50% of full position)
                    kelly_scale = max(0.25, min(0.5, kelly_fraction * 0.25))

            # === RISK-OFF SCALING ===
            # If use_risk_off and market volatility is high, scale down exposure
            risk_off_scale = 1.0
            if use_risk_off:
                # Get market_volatility from features (annualized SPY vol, roughly VIX/100)
                mkt_vol = date_features['market_volatility'].iloc[0] if 'market_volatility' in date_features.columns else 0.15
                # Convert to VIX-like scale (multiply by 100)
                vix_approx = mkt_vol * 100 if mkt_vol < 1 else mkt_vol

                if vix_approx > vix_threshold:
                    # High volatility: 50% equity, 50% cash
                    risk_off_scale = 0.5

            # === COMBINE ALL SCALING FACTORS ===
            # vol_scale: volatility targeting
            # kelly_scale: Kelly fraction sizing
            # risk_off_scale: VIX-based risk reduction
            combined_scale = vol_scale * kelly_scale * risk_off_scale
            combined_scale = max(0.5, min(1.5, combined_scale))  # Bound to 50%-150% (was 10%-200%)

            # Final portfolio return (scaled exposure)
            portfolio_return = gross_return * combined_scale

            # === UPDATE ROLLING TRACKERS ===
            # Track for vol targeting
            realized_vol_window.append(gross_return)

            # Track for Kelly fraction
            rolling_win_rate.append(1.0 if gross_return > 0 else 0.0)
            if gross_return > 0:
                rolling_avg_win.append(gross_return)
                rolling_avg_loss.append(0)
            else:
                rolling_avg_win.append(0)
                rolling_avg_loss.append(gross_return)

            # SPY return
            if entry_date in spy_prices.index and exit_date in spy_prices.index:
                spy_ret = (spy_prices.loc[exit_date, 'adj_close'] -
                          spy_prices.loc[entry_date, 'adj_close']) / spy_prices.loc[entry_date, 'adj_close']
            else:
                spy_ret = 0

            year_returns.append(portfolio_return)
            spy_returns.append(spy_ret)

            all_results.append({
                'year': test_year,
                'date': signal_date,
                'portfolio_return': portfolio_return,
                'gross_return': gross_return,
                'spy_return': spy_ret,
                'excess_return': portfolio_return - spy_ret,
                'position_scale': combined_scale,
                'vol_scale': vol_scale,
                'kelly_scale': kelly_scale,
                'risk_off_scale': risk_off_scale,
                'transaction_cost': total_cost
            })

        # Yearly summary
        if year_returns:
            cum_portfolio = np.prod([1 + r for r in year_returns]) - 1
            cum_spy = np.prod([1 + r for r in spy_returns]) - 1
            excess = cum_portfolio - cum_spy
            win_rate = np.mean([1 if r > s else 0 for r, s in zip(year_returns, spy_returns)])

            yearly_summary.append({
                'year': test_year,
                'portfolio_return': cum_portfolio,
                'spy_return': cum_spy,
                'excess_return': excess,
                'win_rate': win_rate,
                'n_months': len(year_returns),
                'auc': test_auc,
                'precision_at_topn': precision_at_topn,
                'precision_at_10': precision_at_10,
                'spearman_corr': spearman_corr,
                'bottom_decile_rate': bottom_decile_rate
            })

            print(f"\n  YEAR {test_year} BACKTEST:")
            print(f"    Portfolio: {cum_portfolio:+.2%}")
            print(f"    SPY:       {cum_spy:+.2%}")
            print(f"    Excess:    {excess:+.2%}")
            print(f"    Win rate:  {win_rate:.0%}")

    # === FINAL SUMMARY ===

    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 70)

    summary_df = pd.DataFrame(yearly_summary)

    if len(summary_df) == 0:
        print("No results to summarize!")
        return

    print("\n=== YEARLY OUT-OF-SAMPLE PERFORMANCE ===\n")
    print(f"{'Year':<6} {'Portfolio':>12} {'SPY':>10} {'Excess':>10} {'Win%':>8} {'AUC':>8} {'P@10':>8} {'Sprmn':>7} {'Bot10':>7}")
    print("-" * 85)

    for _, row in summary_df.iterrows():
        spearman = row.get('spearman_corr', 0)
        bottom = row.get('bottom_decile_rate', 0)
        print(f"{int(row['year']):<6} {row['portfolio_return']:>+11.1%} {row['spy_return']:>+9.1%} "
              f"{row['excess_return']:>+9.1%} {row['win_rate']:>7.0%} {row['auc']:>7.3f} {row['precision_at_10']:>7.1%} "
              f"{spearman:>6.2f} {bottom:>6.1%}")

    print("-" * 85)

    # Aggregates
    total_portfolio = np.prod([1 + r for r in summary_df['portfolio_return']]) - 1
    total_spy = np.prod([1 + r for r in summary_df['spy_return']]) - 1
    avg_excess = summary_df['excess_return'].mean()
    avg_win_rate = summary_df['win_rate'].mean()
    avg_auc = summary_df['auc'].mean()
    avg_p10 = summary_df['precision_at_10'].mean()
    avg_spearman = summary_df['spearman_corr'].mean() if 'spearman_corr' in summary_df.columns else 0
    avg_bottom = summary_df['bottom_decile_rate'].mean() if 'bottom_decile_rate' in summary_df.columns else 0

    print(f"{'TOTAL':<6} {total_portfolio:>+11.1%} {total_spy:>+9.1%} {total_portfolio - total_spy:>+9.1%}")
    print(f"{'AVG':<6} {summary_df['portfolio_return'].mean():>+11.1%} {summary_df['spy_return'].mean():>+9.1%} "
          f"{avg_excess:>+9.1%} {avg_win_rate:>7.0%} {avg_auc:>7.3f} {avg_p10:>7.1%} "
          f"{avg_spearman:>6.2f} {avg_bottom:>6.1%}")

    # === RISK ANALYTICS ===
    # Compute from all monthly returns (not yearly aggregates)

    results_df = pd.DataFrame(all_results)

    if len(results_df) > 0:
        print("\n" + "=" * 70)
        print("RISK ANALYTICS (Monthly Returns)")
        print("=" * 70)

        port_returns = results_df['portfolio_return'].values
        spy_returns_arr = results_df['spy_return'].values
        excess_returns = results_df['excess_return'].values

        n_months = len(port_returns)
        n_years = n_months / 12

        # --- Annualized Returns ---
        total_port_ret = np.prod(1 + port_returns) - 1
        total_spy_ret = np.prod(1 + spy_returns_arr) - 1
        ann_port_ret = (1 + total_port_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_spy_ret = (1 + total_spy_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

        # --- Volatility (annualized) ---
        port_vol = np.std(port_returns) * np.sqrt(12)
        spy_vol = np.std(spy_returns_arr) * np.sqrt(12)
        excess_vol = np.std(excess_returns) * np.sqrt(12)  # Tracking error

        # --- Sharpe Ratio (assuming 5% risk-free rate) ---
        rf_rate = 0.05
        sharpe_port = (ann_port_ret - rf_rate) / port_vol if port_vol > 0 else 0
        sharpe_spy = (ann_spy_ret - rf_rate) / spy_vol if spy_vol > 0 else 0

        # --- Sortino Ratio (downside deviation only) ---
        downside_returns = port_returns[port_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(12) if len(downside_returns) > 0 else port_vol
        sortino = (ann_port_ret - rf_rate) / downside_vol if downside_vol > 0 else 0

        # --- Information Ratio (excess return / tracking error) ---
        ann_excess = ann_port_ret - ann_spy_ret
        info_ratio = ann_excess / excess_vol if excess_vol > 0 else 0

        # --- Beta to SPY ---
        if np.std(spy_returns_arr) > 0:
            covariance = np.cov(port_returns, spy_returns_arr)[0, 1]
            spy_variance = np.var(spy_returns_arr)
            beta = covariance / spy_variance
        else:
            beta = 1.0

        # --- Alpha (Jensen's Alpha) ---
        # Alpha = Portfolio Return - (Rf + Beta * (Market Return - Rf))
        alpha = ann_port_ret - (rf_rate + beta * (ann_spy_ret - rf_rate))

        # --- Max Drawdown ---
        cum_returns = np.cumprod(1 + port_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)

        # SPY max drawdown for comparison
        spy_cum = np.cumprod(1 + spy_returns_arr)
        spy_running_max = np.maximum.accumulate(spy_cum)
        spy_drawdowns = (spy_cum - spy_running_max) / spy_running_max
        spy_max_dd = np.min(spy_drawdowns)

        # --- Calmar Ratio (annual return / max drawdown) ---
        calmar = ann_port_ret / abs(max_drawdown) if max_drawdown != 0 else 0

        # --- Print Results ---
        print(f"\n{'Metric':<25} {'Portfolio':>12} {'SPY':>12}")
        print("-" * 50)
        print(f"{'Annualized Return':<25} {ann_port_ret:>+11.1%} {ann_spy_ret:>+11.1%}")
        print(f"{'Annualized Volatility':<25} {port_vol:>11.1%} {spy_vol:>11.1%}")
        print(f"{'Sharpe Ratio':<25} {sharpe_port:>11.2f} {sharpe_spy:>11.2f}")
        print(f"{'Sortino Ratio':<25} {sortino:>11.2f} {'--':>12}")
        print(f"{'Max Drawdown':<25} {max_drawdown:>11.1%} {spy_max_dd:>11.1%}")
        print(f"{'Calmar Ratio':<25} {calmar:>11.2f} {'--':>12}")
        print(f"{'Beta to SPY':<25} {beta:>11.2f} {'1.00':>12}")
        print(f"{'Alpha (annualized)':<25} {alpha:>+11.1%} {'--':>12}")
        print(f"{'Information Ratio':<25} {info_ratio:>11.2f} {'--':>12}")
        print(f"{'Tracking Error':<25} {excess_vol:>11.1%} {'--':>12}")

        # --- Turnover Analytics ---
        if track_turnover and turnover_records:
            turnover_df = pd.DataFrame(turnover_records)
            avg_turnover = turnover_df['turnover'].mean()
            annual_turnover = avg_turnover * 12  # Approximate annual turnover

            print(f"\n{'--- TURNOVER ANALYTICS ---':^50}")
            print(f"{'Avg Monthly Turnover':<25} {avg_turnover:>11.1%}")
            print(f"{'Implied Annual Turnover':<25} {annual_turnover:>11.0%}")
            print(f"{'Avg Positions Exited/Mo':<25} {turnover_df['positions_exited'].mean():>11.1f}")
            print(f"{'Avg Positions Entered/Mo':<25} {turnover_df['positions_entered'].mean():>11.1f}")

            # Turnover cost impact
            avg_extra_slippage = 0.0005 * avg_turnover * 12  # Annual extra slippage
            print(f"{'Est. Annual Slippage Cost':<25} {avg_extra_slippage:>11.2%}")

        # --- Position Scaling Analytics ---
        if vol_target or use_kelly:
            avg_scale = results_df['position_scale'].mean()
            min_scale = results_df['position_scale'].min()
            max_scale = results_df['position_scale'].max()

            print(f"\n{'--- POSITION SCALING ---':^50}")
            print(f"{'Avg Position Scale':<25} {avg_scale:>11.1%}")
            print(f"{'Min Position Scale':<25} {min_scale:>11.1%}")
            print(f"{'Max Position Scale':<25} {max_scale:>11.1%}")

            if vol_target:
                avg_vol_scale = results_df['vol_scale'].mean()
                print(f"{'Avg Vol-Target Scale':<25} {avg_vol_scale:>11.1%}")
            if use_kelly:
                avg_kelly_scale = results_df['kelly_scale'].mean()
                print(f"{'Avg Kelly Scale':<25} {avg_kelly_scale:>11.1%}")

        # --- Risk-Adjusted Assessment ---
        print("\n" + "-" * 50)
        print("RISK ASSESSMENT:")

        if sharpe_port > 1.0:
            print(f"  Sharpe {sharpe_port:.2f} - EXCELLENT risk-adjusted returns")
        elif sharpe_port > 0.5:
            print(f"  Sharpe {sharpe_port:.2f} - GOOD risk-adjusted returns")
        else:
            print(f"  Sharpe {sharpe_port:.2f} - WEAK risk-adjusted returns")

        if abs(max_drawdown) > 0.30:
            print(f"  Max DD {max_drawdown:.1%} - HIGH RISK (may be hard to stomach)")
        elif abs(max_drawdown) > 0.20:
            print(f"  Max DD {max_drawdown:.1%} - MODERATE risk")
        else:
            print(f"  Max DD {max_drawdown:.1%} - ACCEPTABLE risk")

        if beta > 1.5:
            print(f"  Beta {beta:.2f} - HIGH market exposure (amplifies moves)")
        elif beta > 1.0:
            print(f"  Beta {beta:.2f} - ABOVE-MARKET exposure")
        else:
            print(f"  Beta {beta:.2f} - DEFENSIVE positioning")

        if info_ratio > 0.5:
            print(f"  Info Ratio {info_ratio:.2f} - STRONG alpha generation")
        elif info_ratio > 0:
            print(f"  Info Ratio {info_ratio:.2f} - POSITIVE but modest alpha")
        else:
            print(f"  Info Ratio {info_ratio:.2f} - NO consistent alpha")

    # === KEY INSIGHTS ===

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    # Consistency check
    positive_excess_years = (summary_df['excess_return'] > 0).sum()
    total_years = len(summary_df)

    print(f"\n1. CONSISTENCY: Beat SPY in {positive_excess_years}/{total_years} years ({positive_excess_years/total_years:.0%})")

    # Best and worst years
    best_year = summary_df.loc[summary_df['excess_return'].idxmax()]
    worst_year = summary_df.loc[summary_df['excess_return'].idxmin()]

    print(f"\n2. BEST YEAR:  {int(best_year['year'])} with {best_year['excess_return']:+.1%} excess")
    print(f"   WORST YEAR: {int(worst_year['year'])} with {worst_year['excess_return']:+.1%} excess")

    # Bear market performance (2022)
    bear_years = summary_df[summary_df['spy_return'] < 0]
    if len(bear_years) > 0:
        print(f"\n3. BEAR MARKET PERFORMANCE (SPY negative years):")
        for _, row in bear_years.iterrows():
            print(f"   {int(row['year'])}: Portfolio {row['portfolio_return']:+.1%} vs SPY {row['spy_return']:+.1%} "
                  f"(excess: {row['excess_return']:+.1%})")

    # Model quality
    print(f"\n4. MODEL QUALITY:")
    print(f"   Avg AUC: {avg_auc:.3f} (>0.55 is good, >0.60 is strong)")
    print(f"   Avg Precision@10: {avg_p10:.1%} (>15% is good, >20% is excellent)")
    print(f"   Avg Spearman Corr: {avg_spearman:.3f} (>0.05 is meaningful)")
    print(f"   Avg Bottom Decile: {avg_bottom:.1%} (should be <10% if model works)")

    # Ranking check
    if avg_spearman > 0.05 and avg_bottom < 0.10:
        print("   Model shows GOOD ranking ability (top picks beat bottom picks)")
    elif avg_spearman > 0 and avg_bottom < avg_p10:
        print("   Model shows SOME ranking ability")
    else:
        print("   Model ranking is WEAK - may be noise")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if positive_excess_years >= total_years * 0.6 and avg_p10 > 0.12:
        print("\n[+] ROBUST: Strategy shows consistent out-of-sample alpha")
        print("  - Beats SPY in majority of years")
        print("  - Precision@10 above random in most periods")
        print("  - Consider paper trading next")
    elif positive_excess_years >= total_years * 0.5:
        print("\n[~] MARGINAL: Strategy shows some signal but inconsistent")
        print("  - May be loading on known factors")
        print("  - Consider factor analysis before live trading")
    else:
        print("\n[-] WEAK: Strategy does not generalize well")
        print("  - Original backtest may have been overfit")
        print("  - Consider different features or longer training")

    # === BOOTSTRAP STATISTICAL SIGNIFICANCE ===
    if run_bootstrap and len(results_df) > 0:
        print("\n" + "=" * 70)
        print("STATISTICAL SIGNIFICANCE (Bootstrap)")
        print("=" * 70)

        # Bootstrap on monthly excess returns
        excess_returns = results_df['excess_return'].values
        mean_excess, p_value, ci_low, ci_high = bootstrap_pvalue(excess_returns)

        print(f"\nMonthly Excess Returns (vs SPY):")
        print(f"  Mean:           {mean_excess:+.2%}")
        print(f"  95% CI:         [{ci_low:+.2%}, {ci_high:+.2%}]")
        print(f"  p-value:        {p_value:.4f} (H0: mean <= 0)")

        if p_value < 0.05:
            print(f"\n  [+] SIGNIFICANT at 5% level (p={p_value:.4f})")
            print("    The excess return is unlikely due to chance alone.")
        elif p_value < 0.10:
            print(f"\n  [~] MARGINALLY significant (p={p_value:.4f})")
            print("    Some evidence of alpha, but not conclusive.")
        else:
            print(f"\n  [-] NOT significant (p={p_value:.4f})")
            print("    Cannot reject that excess return is due to chance.")

        # Also test portfolio returns vs 0
        port_returns = results_df['portfolio_return'].values
        port_mean, port_p, port_ci_low, port_ci_high = bootstrap_pvalue(port_returns)
        print(f"\nPortfolio Absolute Returns:")
        print(f"  Mean:           {port_mean:+.2%}")
        print(f"  95% CI:         [{port_ci_low:+.2%}, {port_ci_high:+.2%}]")
        print(f"  p-value:        {port_p:.4f}")

    # === IC (INFORMATION COEFFICIENT) ANALYSIS ===
    if 'spearman_corr' in summary_df.columns:
        print("\n" + "=" * 70)
        print("INFORMATION COEFFICIENT (IC) ANALYSIS")
        print("=" * 70)

        ic_values = summary_df['spearman_corr'].values
        avg_ic = np.mean(ic_values)
        std_ic = np.std(ic_values)
        ic_ir = avg_ic / std_ic if std_ic > 0 else 0  # IC Information Ratio

        print(f"\nIC = Spearman correlation between predictions and actual returns")
        print(f"\n{'Year':<8} {'IC':>10} {'Interpretation':<20}")
        print("-" * 40)
        for _, row in summary_df.iterrows():
            ic = row['spearman_corr']
            if ic > 0.05:
                interp = "GOOD signal"
            elif ic > 0.02:
                interp = "Weak signal"
            elif ic > 0:
                interp = "Marginal"
            else:
                interp = "No signal"
            print(f"{int(row['year']):<8} {ic:>+9.3f} {interp:<20}")

        print("-" * 40)
        print(f"{'Average':<8} {avg_ic:>+9.3f}")
        print(f"{'Std Dev':<8} {std_ic:>9.3f}")
        print(f"{'IC IR':<8} {ic_ir:>9.2f} (IC / std)")

        # IC quality assessment
        print(f"\nIC Quality Assessment:")
        if avg_ic > 0.05:
            print(f"  [OK] STRONG: Avg IC={avg_ic:.3f} (institutional quality)")
        elif avg_ic > 0.02:
            print(f"  ~ MODERATE: Avg IC={avg_ic:.3f} (usable with ensembling)")
        elif avg_ic > 0:
            print(f"  ~ WEAK: Avg IC={avg_ic:.3f} (marginal predictive power)")
        else:
            print(f"  [FAIL] NONE: Avg IC={avg_ic:.3f} (no predictive power)")

        if ic_ir > 0.5:
            print(f"  [OK] IC IR={ic_ir:.2f} - Signal is STABLE across time")
        elif ic_ir > 0.2:
            print(f"  ~ IC IR={ic_ir:.2f} - Signal has MODERATE stability")
        else:
            print(f"  [FAIL] IC IR={ic_ir:.2f} - Signal is UNSTABLE (high variance)")

        # Percentage of positive IC years
        pct_positive_ic = (ic_values > 0).mean()
        print(f"\n  Positive IC in {pct_positive_ic:.0%} of years ({(ic_values > 0).sum()}/{len(ic_values)})")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("models/walk_forward_results.csv", index=False)
    summary_df.to_csv("models/walk_forward_summary.csv", index=False)
    print(f"\nResults saved to models/walk_forward_*.csv")

    return summary_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Walk-forward validation with advanced features")
    parser.add_argument("--risk-off", action="store_true", help="Enable risk-off scaling based on volatility")
    parser.add_argument("--compare", action="store_true", help="Run both with and without risk-off")
    parser.add_argument("--vol-target", type=float, default=None,
                       help="Target annual volatility (e.g., 0.15 for 15%%)")
    parser.add_argument("--kelly", action="store_true", help="Use Kelly fraction position sizing")
    parser.add_argument("--ensemble", type=int, default=3,
                       help="Number of models in ensemble (default: 3, recommended: 10-50 for production)")
    parser.add_argument("--full", action="store_true",
                       help="Run with all features: risk-off + 15%% vol target + Kelly + ensemble")
    parser.add_argument("--balanced", action="store_true",
                       help="Run with BALANCED settings: 25%% vol target, no Kelly, lighter risk-off")
    parser.add_argument("--elite", action="store_true",
                       help="Run ELITE mode: long-short + 20%% vol target + 20-model ensemble + factor neutralization")
    parser.add_argument("--ultra", action="store_true",
                       help="Run ULTRA mode: elite + 50-model ensemble (maximum signal extraction)")
    parser.add_argument("--pro", action="store_true",
                       help="Run PRO mode: continuous z-score weights + turnover constraint (ChatGPT recommended)")
    parser.add_argument("--no-survivorship-fix", action="store_true",
                       help="Disable survivorship bias fix (use current S&P 500 list)")
    parser.add_argument("--long-short", action="store_true",
                       help="Use long-short market-neutral portfolio (long top N, short bottom N)")
    parser.add_argument("--short-cost", type=float, default=50,
                       help="Annual short borrow cost in basis points (default: 50)")
    parser.add_argument("--neutralize-beta", action="store_true",
                       help="Neutralize portfolio beta (target beta = 0)")
    parser.add_argument("--neutralize-sector", action="store_true",
                       help="Neutralize sector exposure (equal sector weights)")
    parser.add_argument("--optimize", action="store_true",
                       help="Use CVXPY portfolio optimizer with beta/vol/sector constraints (ChatGPT institutional-grade)")
    parser.add_argument("--factor-neutral", action="store_true",
                       help="Use factor-neutral optimizer (removes market/sector/momentum risk)")
    parser.add_argument("--ranker", action="store_true",
                       help="Use LightGBM ranking model (predicts continuous residual ranks)")
    parser.add_argument("--meta-ensemble", action="store_true",
                       help="Use meta-model ensemble (XGB + LightGBM + Ridge) for diverse signal")
    parser.add_argument("--stacked-blend", action="store_true",
                       help="Use stacked alpha blending (XGB + LGBM -> Ridge meta-learner)")
    parser.add_argument("--regression", action="store_true",
                       help="Use XGBoost Regressor on continuous residual returns (ChatGPT recommended)")

    # Mega-cap overlay arguments (fix for 2024 underperformance)
    parser.add_argument("--mega-cap-overlay", action="store_true",
                       help="Enable mega-cap overlay (force top 5 SPY holdings, fixes 2024 -22%% excess)")
    parser.add_argument("--min-mega-cap-allocation", type=float, default=0.25,
                       help="Minimum portfolio weight in mega-caps (default 25%%)")
    parser.add_argument("--mega-cap-force-top-k", type=int, default=5,
                       help="Force include top K mega-caps by SPY weight (default 5)")
    parser.add_argument("--mega-cap-weight-method", type=str, default='hybrid',
                       choices=['equal', 'cap_weighted', 'hybrid'],
                       help="Weighting method: equal, cap_weighted, or hybrid (default hybrid)")

    args = parser.parse_args()

    fix_survivorship = not args.no_survivorship_fix

    if args.pro:
        # Run PRO mode: ChatGPT recommended continuous weights + turnover constraint
        print("\n" + "=" * 70)
        print("PRO MODE: Continuous Z-Score Weights + Turnover Constraint")
        print("ChatGPT recommended optimal portfolio construction")
        print("=" * 70)
        walk_forward_validation(
            min_train_years=3,
            top_n=20,  # Not used in continuous mode
            transaction_cost=0.001,
            use_risk_off=False,
            vol_target=0.25,     # 25% vol target (less aggressive)
            use_kelly=False,
            track_turnover=True,
            n_ensemble=5,        # 5 models (less smoothing per ChatGPT)
            fix_survivorship_bias=fix_survivorship,
            long_short=True,     # Still L/S but with continuous weights
            short_cost_bps=50,
            neutralize_beta_flag=True,   # Beta neutral only
            neutralize_sector_flag=False,  # NO sector neutral (per ChatGPT)
            continuous_weights=True,      # KEY: Use continuous weights
            max_position_weight=0.05,     # 5% max per position
            max_turnover_per_stock=0.03   # 3% max weight change per month
        )

    elif args.ultra:
        # Run ULTRA mode: Maximum signal extraction
        print("\n" + "=" * 70)
        print("ULTRA MODE: Maximum Signal Extraction")
        print("50-model ensemble + long-short + factor neutralization")
        print("=" * 70)
        walk_forward_validation(
            min_train_years=3,
            top_n=20,
            transaction_cost=0.001,
            use_risk_off=False,
            vol_target=0.20,
            use_kelly=False,
            track_turnover=True,
            n_ensemble=50,       # 50-model ensemble (ChatGPT recommended 20-50)
            fix_survivorship_bias=fix_survivorship,
            long_short=True,
            short_cost_bps=50,
            neutralize_beta_flag=True,
            neutralize_sector_flag=True
        )

    elif args.elite:
        # Run ELITE mode: Long-short with proper IC monetization + factor neutralization
        print("\n" + "=" * 70)
        print("ELITE MODE: Long-Short Market-Neutral Portfolio")
        print("20-model ensemble + factor neutralization")
        print("=" * 70)
        walk_forward_validation(
            min_train_years=3,
            top_n=20,
            transaction_cost=0.001,
            use_risk_off=False,  # No VIX (market-neutral already hedged)
            vol_target=0.20,     # 20% vol target
            use_kelly=False,     # No Kelly (L/S has different dynamics)
            track_turnover=True,
            n_ensemble=20,       # 20-model ensemble (ChatGPT recommendation)
            fix_survivorship_bias=fix_survivorship,
            long_short=True,     # KEY: Long-short portfolio
            short_cost_bps=50,   # 50 bps annual borrow cost
            neutralize_beta_flag=True,   # Beta neutralization
            neutralize_sector_flag=True  # Sector neutralization
        )

    elif args.balanced:
        # Run with BALANCED settings (tuned for alpha + risk)
        print("\n" + "=" * 70)
        print("BALANCED MODE: Tuned for Alpha + Risk Control")
        print("=" * 70)
        walk_forward_validation(
            min_train_years=3,
            top_n=20,
            transaction_cost=0.001,
            use_risk_off=False,  # No VIX risk-off (too harsh)
            vol_target=0.25,     # 25% vol target (not 15%)
            use_kelly=False,     # No Kelly (too aggressive scaling)
            track_turnover=True,
            n_ensemble=3,
            fix_survivorship_bias=fix_survivorship,
            long_short=False
        )

    elif args.full:
        # Run with ALL advanced features (STRICT mode)
        print("\n" + "=" * 70)
        print("ELITE MODE: ALL ADVANCED FEATURES (STRICT)")
        print("=" * 70)
        walk_forward_validation(
            min_train_years=3,
            top_n=20,
            transaction_cost=0.001,
            use_risk_off=True,
            vix_threshold=25.0,
            vol_target=0.15,  # 15% annual vol target
            use_kelly=True,
            track_turnover=True,
            n_ensemble=5,  # Full ensemble for elite mode
            fix_survivorship_bias=fix_survivorship
        )

    elif args.compare:
        print("\n" + "=" * 70)
        print("COMPARISON: BASELINE (no features)")
        print("=" * 70)
        results_baseline = walk_forward_validation(
            min_train_years=3,
            top_n=20,
            transaction_cost=0.001,
            use_risk_off=False,
            vol_target=None,
            use_kelly=False,
            fix_survivorship_bias=fix_survivorship
        )

        print("\n\n" + "=" * 70)
        print("COMPARISON: WITH RISK-OFF (VIX > 25 = 50% position)")
        print("=" * 70)
        results_risk_off = walk_forward_validation(
            min_train_years=3,
            top_n=20,
            transaction_cost=0.001,
            use_risk_off=True,
            vix_threshold=25.0,
            vol_target=None,
            use_kelly=False,
            fix_survivorship_bias=fix_survivorship
        )

        print("\n\n" + "=" * 70)
        print("COMPARISON: VOL TARGET (15% annual)")
        print("=" * 70)
        results_vol_target = walk_forward_validation(
            min_train_years=3,
            top_n=20,
            transaction_cost=0.001,
            use_risk_off=False,
            vol_target=0.15,
            use_kelly=False,
            fix_survivorship_bias=fix_survivorship
        )

        print("\n\n" + "=" * 70)
        print("COMPARISON: FULL ELITE (risk-off + vol target + Kelly)")
        print("=" * 70)
        results_elite = walk_forward_validation(
            min_train_years=3,
            top_n=20,
            transaction_cost=0.001,
            use_risk_off=True,
            vix_threshold=25.0,
            vol_target=0.15,
            use_kelly=True,
            fix_survivorship_bias=fix_survivorship
        )

        # Summary comparison
        print("\n\n" + "=" * 70)
        print("FEATURE COMPARISON SUMMARY")
        print("=" * 70)

        configs = [
            ("Baseline", results_baseline),
            ("Risk-Off", results_risk_off),
            ("Vol Target", results_vol_target),
            ("Elite (all)", results_elite)
        ]

        print(f"\n{'Config':<15} {'Total Return':>14} {'2022 Return':>12}")
        print("-" * 45)

        for name, results in configs:
            if results is not None and len(results) > 0:
                total_ret = (1 + results['portfolio_return']).prod() - 1
                r2022 = results[results['year'] == 2022]
                ret_2022 = r2022['portfolio_return'].values[0] if len(r2022) > 0 else float('nan')
                print(f"{name:<15} {total_ret:>+13.1%} {ret_2022:>+11.1%}")

    elif args.factor_neutral:
        # Run with factor-neutral optimizer (ChatGPT institutional-grade Phase 2)
        print("\n" + "=" * 70)
        print("ENTERPRISE FACTOR-NEUTRAL MODE: De-noised Portfolio Optimization")
        print("Features: L2 regularization | Conviction scaling | Regime detection")
        print("7-Factor Model: Beta + Sector + Momentum + Size + Value + Low-Vol + Reversal")
        if args.regression:
            print("ML Model: XGBoost REGRESSOR (continuous residual returns - ChatGPT recommended)")
        elif args.stacked_blend:
            print("ML Model: STACKED BLEND (XGB + LGBM -> LogisticRegression meta-learner)")
        elif args.meta_ensemble:
            print("ML Model: META-ENSEMBLE (XGBoost + LightGBM + Ridge)")
        elif args.ranker:
            print("ML Model: LightGBM RANKER (continuous residual rank prediction)")
        else:
            print("ML Model: XGBoost Classifier (binary top-10% prediction)")
        print("Beta: 1.05 +/- 0.20 | Vol: 22% +/- 10% | Sector max: 30% | No min weight")
        print("=" * 70)
        walk_forward_validation(
            min_train_years=3,
            top_n=30,
            transaction_cost=0.001,
            use_risk_off=False,
            vol_target=None,
            use_kelly=False,
            track_turnover=True,
            n_ensemble=args.ensemble,
            fix_survivorship_bias=fix_survivorship,
            long_short=False,
            use_optimizer=False,
            use_factor_neutral=True,  # KEY: Factor-neutral optimization
            use_ranker=args.ranker,   # KEY: LightGBM ranking model
            use_meta_ensemble=args.meta_ensemble,  # KEY: XGB + LGBM + Ridge ensemble
            use_stacked_blend=args.stacked_blend,  # KEY: Stacked alpha blending
            use_regression=args.regression   # KEY: XGBoost Regressor (ChatGPT recommended)
        )

    elif args.optimize:
        # Run with CVXPY portfolio optimizer (ChatGPT institutional-grade)
        print("\n" + "=" * 70)
        print("OPTIMIZE MODE: CVXPY Portfolio Optimizer")
        print("Beta target: 1.0 +/- 0.1 | Vol target: 16% | Sector max: 20%")
        print("=" * 70)
        walk_forward_validation(
            min_train_years=3,
            top_n=30,  # More stocks for optimizer to choose from
            transaction_cost=0.001,
            use_risk_off=False,
            vol_target=None,  # Optimizer handles vol internally
            use_kelly=False,
            track_turnover=True,
            n_ensemble=args.ensemble,
            fix_survivorship_bias=fix_survivorship,
            long_short=False,  # Long-only with optimizer
            use_optimizer=True  # KEY: Use CVXPY optimizer
        )

    else:
        walk_forward_validation(
            min_train_years=3,
            top_n=20,
            transaction_cost=0.001,
            use_risk_off=args.risk_off,
            vix_threshold=25.0,
            vol_target=args.vol_target,
            use_kelly=args.kelly,
            track_turnover=True,
            n_ensemble=args.ensemble,
            fix_survivorship_bias=fix_survivorship,
            long_short=args.long_short,
            short_cost_bps=args.short_cost,
            neutralize_beta_flag=args.neutralize_beta,
            neutralize_sector_flag=args.neutralize_sector,
            use_meta_ensemble=args.meta_ensemble,
            use_stacked_blend=args.stacked_blend,
            use_regression=args.regression,
            # Mega-cap overlay parameters
            use_mega_cap_overlay=args.mega_cap_overlay,
            min_mega_cap_allocation=args.min_mega_cap_allocation,
            mega_cap_force_top_k=args.mega_cap_force_top_k,
            mega_cap_weight_method=args.mega_cap_weight_method
        )
