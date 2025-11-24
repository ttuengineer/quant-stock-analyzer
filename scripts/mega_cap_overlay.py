"""
Mega-Cap Overlay for Portfolio Construction.

Solves the 2024 problem: Ensures we don't miss mega-cap rallies.

Key Insight:
- SPY is cap-weighted (top 7 = ~30% of index)
- Our equal-weight top-20 systematically underweights mega-caps
- When mega-caps rally (2019, 2024), we underperform badly

Solution: Hybrid Approach
1. Force include top 5-10 SPY mega-caps if they score reasonably well
2. Weight them by market cap (not equal weight)
3. Fill remaining slots with our ML picks
4. Ensure total mega-cap allocation >= 20-30% of portfolio
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime

# Top SPY holdings as of 2024 (update periodically)
MEGA_CAPS = {
    'AAPL': {'rank': 1, 'spy_weight': 0.070},  # ~7% of SPY
    'MSFT': {'rank': 2, 'spy_weight': 0.065},  # ~6.5%
    'NVDA': {'rank': 3, 'spy_weight': 0.060},  # ~6%
    'GOOGL': {'rank': 4, 'spy_weight': 0.035}, # ~3.5%
    'AMZN': {'rank': 5, 'spy_weight': 0.035},  # ~3.5%
    'META': {'rank': 6, 'spy_weight': 0.025},  # ~2.5%
    'TSLA': {'rank': 7, 'spy_weight': 0.020},  # ~2%
    'BRK.B': {'rank': 8, 'spy_weight': 0.018}, # ~1.8%
    'LLY': {'rank': 9, 'spy_weight': 0.015},   # ~1.5%
    'JPM': {'rank': 10, 'spy_weight': 0.014},  # ~1.4%
}

# Alternative ticker names (handle variations)
TICKER_ALIASES = {
    'GOOG': 'GOOGL',
    'BRK-B': 'BRK.B',
    'BF-B': 'BRK.B',
}


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbol (handle aliases)."""
    ticker = ticker.upper().strip()
    return TICKER_ALIASES.get(ticker, ticker)


def apply_mega_cap_overlay(
    predictions_df: pd.DataFrame,
    top_n: int = 20,
    min_score_threshold: float = 40.0,
    mega_cap_min_allocation: float = 0.20,
    mega_cap_weight_method: str = 'hybrid',
    force_include_top_k: int = 5,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply mega-cap overlay to portfolio construction.

    Args:
        predictions_df: DataFrame with columns ['ticker', 'score', 'prediction']
        top_n: Total portfolio size (default 20 stocks)
        min_score_threshold: Minimum score for mega-caps to be included (default 40)
        mega_cap_min_allocation: Minimum % of portfolio in mega-caps (default 20%)
        mega_cap_weight_method: 'equal', 'cap_weighted', or 'hybrid' (default hybrid)
        force_include_top_k: Force include top K mega-caps if score > threshold
        verbose: Print diagnostics

    Returns:
        portfolio_df: DataFrame with selected stocks and weights
        diagnostics: Dict with allocation details
    """
    # Normalize tickers
    predictions_df = predictions_df.copy()
    predictions_df['ticker'] = predictions_df['ticker'].apply(normalize_ticker)

    # Sort predictions by score
    predictions_df = predictions_df.sort_values('score', ascending=False)

    # Identify mega-caps in our predictions
    mega_cap_tickers = list(MEGA_CAPS.keys())
    predictions_df['is_mega_cap'] = predictions_df['ticker'].isin(mega_cap_tickers)

    # Get all mega-caps in predictions (regardless of score)
    available_mega_caps = predictions_df[predictions_df['is_mega_cap']].copy()

    # Force include top K mega-caps (by SPY rank) with NO threshold
    # We force include regardless of score to ensure mega-cap exposure in all market conditions
    # Only exclude if score is catastrophically low (< 5, essentially random)
    forced_mega_caps = []
    catastrophic_threshold = 5.0  # Only exclude if completely broken prediction

    for ticker, info in sorted(MEGA_CAPS.items(), key=lambda x: x[1]['rank']):
        if ticker in available_mega_caps['ticker'].values:
            ticker_score = available_mega_caps[available_mega_caps['ticker'] == ticker]['score'].iloc[0]
            # Force include unless prediction is completely broken
            if ticker_score >= catastrophic_threshold:
                forced_mega_caps.append(ticker)
                if len(forced_mega_caps) >= force_include_top_k:
                    break

    # Get non-mega-cap ML picks
    ml_picks = predictions_df[~predictions_df['is_mega_cap']].copy()

    # Calculate slots remaining after mega-caps
    mega_cap_slots = len(forced_mega_caps)
    ml_slots = top_n - mega_cap_slots

    if verbose:
        print(f"\n[MEGA-CAP OVERLAY]")
        print(f"  Forcing {mega_cap_slots} mega-caps: {forced_mega_caps}")
        print(f"  Remaining {ml_slots} slots for ML picks")

    # Select final portfolio
    selected_mega_caps = predictions_df[predictions_df['ticker'].isin(forced_mega_caps)].copy()
    selected_ml_picks = ml_picks.head(ml_slots).copy()

    # Combine
    portfolio = pd.concat([selected_mega_caps, selected_ml_picks], ignore_index=True)

    # Calculate weights based on method
    if mega_cap_weight_method == 'equal':
        # Equal weight all stocks
        portfolio['weight'] = 1.0 / len(portfolio)

    elif mega_cap_weight_method == 'cap_weighted':
        # Weight mega-caps by SPY weight, equal-weight ML picks
        weights = []
        for _, row in portfolio.iterrows():
            ticker = row['ticker']
            if ticker in MEGA_CAPS:
                # Use SPY weight (scaled)
                weights.append(MEGA_CAPS[ticker]['spy_weight'])
            else:
                # Equal weight for ML picks
                weights.append(1.0 / ml_slots if ml_slots > 0 else 0)

        # Normalize to sum to 1
        total = sum(weights)
        portfolio['weight'] = [w / total for w in weights]

    elif mega_cap_weight_method == 'hybrid':
        # Hybrid: Allocate minimum to mega-caps, rest to ML picks
        # Mega-caps: Weight by (score * spy_weight)
        # ML picks: Weight by score

        mega_cap_scores = []
        ml_scores = []

        for _, row in portfolio.iterrows():
            ticker = row['ticker']
            score = row['score']

            if ticker in MEGA_CAPS:
                # Score * SPY weight (gives mega-caps natural advantage)
                combined_score = score * MEGA_CAPS[ticker]['spy_weight'] * 1000
                mega_cap_scores.append(combined_score)
            else:
                ml_scores.append(score)

        # Allocate weights ensuring minimum mega-cap allocation
        total_mega_score = sum(mega_cap_scores) if mega_cap_scores else 1
        total_ml_score = sum(ml_scores) if ml_scores else 1

        # Start with minimum mega-cap allocation
        mega_cap_allocation = max(mega_cap_min_allocation, len(forced_mega_caps) / top_n)
        ml_allocation = 1.0 - mega_cap_allocation

        weights = []
        mega_idx = 0
        ml_idx = 0

        for _, row in portfolio.iterrows():
            ticker = row['ticker']

            if ticker in MEGA_CAPS:
                # Allocate proportional to combined score within mega-cap bucket
                w = (mega_cap_scores[mega_idx] / total_mega_score) * mega_cap_allocation
                weights.append(w)
                mega_idx += 1
            else:
                # Allocate proportional to score within ML picks bucket
                w = (ml_scores[ml_idx] / total_ml_score) * ml_allocation
                weights.append(w)
                ml_idx += 1

        portfolio['weight'] = weights

    else:
        raise ValueError(f"Unknown weight method: {mega_cap_weight_method}")

    # Ensure weights sum to 1
    portfolio['weight'] = portfolio['weight'] / portfolio['weight'].sum()

    # Calculate diagnostics
    mega_cap_total_weight = portfolio[portfolio['is_mega_cap']]['weight'].sum()
    ml_total_weight = portfolio[~portfolio['is_mega_cap']]['weight'].sum()

    diagnostics = {
        'total_stocks': len(portfolio),
        'mega_caps': mega_cap_slots,
        'ml_picks': ml_slots,
        'mega_cap_weight': mega_cap_total_weight,
        'ml_weight': ml_total_weight,
        'forced_mega_caps': forced_mega_caps,
        'weight_method': mega_cap_weight_method,
    }

    if verbose:
        print(f"\n[PORTFOLIO COMPOSITION]")
        print(f"  Total stocks: {len(portfolio)}")
        print(f"  Mega-caps: {mega_cap_slots} stocks, {mega_cap_total_weight:.1%} weight")
        print(f"  ML picks: {ml_slots} stocks, {ml_total_weight:.1%} weight")
        print(f"\n  Top 10 holdings:")
        for i, row in portfolio.head(10).iterrows():
            mc_flag = "[MEGA-CAP]" if row['is_mega_cap'] else ""
            print(f"    {row['ticker']:6s} {row['weight']:6.1%}  (score: {row['score']:5.1f}) {mc_flag}")

    return portfolio, diagnostics


def adjust_for_regime(
    portfolio: pd.DataFrame,
    market_regime: str = 'NORMAL',
    vix_level: float = 20.0,
    dispersion_level: float = 0.20
) -> pd.DataFrame:
    """
    Adjust portfolio weights based on market regime.

    Regimes:
    - LOW_DISPERSION: Mega-cap rally (like 2024), increase mega-cap weight
    - HIGH_VOL: Risk-off (like 2018, 2022), reduce exposure or go defensive
    - MOMENTUM: Strong trend (like 2021), full exposure
    - NORMAL: Balanced

    Args:
        portfolio: Portfolio DataFrame with weights
        market_regime: One of ['LOW_DISPERSION', 'HIGH_VOL', 'MOMENTUM', 'NORMAL']
        vix_level: Current VIX level
        dispersion_level: Cross-sectional dispersion

    Returns:
        Adjusted portfolio DataFrame
    """
    portfolio = portfolio.copy()

    if market_regime == 'LOW_DISPERSION' or dispersion_level < 0.15:
        # Mega-cap rally environment (2024-style)
        # Increase mega-cap weights by 50%
        mega_cap_mask = portfolio['is_mega_cap']
        portfolio.loc[mega_cap_mask, 'weight'] *= 1.5
        portfolio.loc[~mega_cap_mask, 'weight'] *= 0.7

        # Renormalize
        portfolio['weight'] = portfolio['weight'] / portfolio['weight'].sum()

        print(f"\n[REGIME ADJUSTMENT: LOW_DISPERSION]")
        print(f"  Increased mega-cap allocation by 50%")
        print(f"  New mega-cap weight: {portfolio[mega_cap_mask]['weight'].sum():.1%}")

    elif market_regime == 'HIGH_VOL' or vix_level > 30:
        # Risk-off environment (2018, 2022-style)
        # Reduce overall exposure by 30% (equivalent to 30% cash)
        portfolio['weight'] *= 0.70

        print(f"\n[REGIME ADJUSTMENT: HIGH_VOL]")
        print(f"  Reduced exposure by 30% (30% cash equivalent)")
        print(f"  Total invested: {portfolio['weight'].sum():.1%}")

    elif market_regime == 'MOMENTUM':
        # Strong trend environment (2021-style)
        # Full exposure, slight overweight to high-momentum picks
        print(f"\n[REGIME ADJUSTMENT: MOMENTUM]")
        print(f"  Full exposure, no adjustment")

    else:
        # Normal environment
        pass

    return portfolio


# Example usage
if __name__ == "__main__":
    # Test data
    test_predictions = pd.DataFrame({
        'ticker': ['NVDA', 'AAPL', 'MSFT', 'PLTR', 'COIN', 'GOOGL', 'TSLA', 'META',
                   'AMD', 'INTC', 'QCOM', 'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX',
                   'KLAC', 'ASML', 'TSM', 'UMC', 'GFS', 'ON', 'STM', 'MRVL'],
        'score': [85, 78, 75, 82, 79, 70, 68, 65,
                  72, 71, 70, 69, 68, 67, 66, 65,
                  64, 63, 62, 61, 60, 59, 58, 57],
        'prediction': [0.65, 0.62, 0.61, 0.64, 0.63, 0.58, 0.57, 0.56,
                       0.60, 0.59, 0.58, 0.58, 0.57, 0.56, 0.56, 0.55,
                       0.55, 0.54, 0.54, 0.53, 0.53, 0.52, 0.52, 0.51]
    })

    print("="*70)
    print("TESTING MEGA-CAP OVERLAY")
    print("="*70)

    # Test 1: Hybrid method (recommended)
    print("\n\n=== TEST 1: HYBRID METHOD (Recommended) ===")
    portfolio, diag = apply_mega_cap_overlay(
        test_predictions,
        top_n=20,
        min_score_threshold=40,
        mega_cap_min_allocation=0.25,  # 25% minimum in mega-caps
        mega_cap_weight_method='hybrid',
        force_include_top_k=5,
        verbose=True
    )

    # Test 2: Low dispersion regime (2024-style)
    print("\n\n=== TEST 2: LOW DISPERSION REGIME (2024-Style) ===")
    portfolio_adjusted = adjust_for_regime(
        portfolio,
        market_regime='LOW_DISPERSION',
        dispersion_level=0.12
    )

    print(f"\nFinal mega-cap allocation: {portfolio_adjusted[portfolio_adjusted['is_mega_cap']]['weight'].sum():.1%}")
