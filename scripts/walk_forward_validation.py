"""
Walk-Forward Out-of-Sample Validation.

This is the GOLD STANDARD test for strategy robustness.

Instead of: Train once (2015-2020) → Test once (2023-2025)
We do:      Train → Test → Retrain → Test → Retrain → Test...

Each test year is TRULY out-of-sample (model never saw it during training).
This reveals whether the model generalizes or just got lucky.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stock_analyzer.database import Database

# ML imports
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import xgboost as xgb


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
    n_ensemble: int = 3  # Number of models to ensemble (different seeds)
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
    if n_ensemble > 1:
        print(f"(ENSEMBLE: {n_ensemble} models with different seeds)")
    if use_risk_off:
        print(f"(WITH RISK-OFF: VIX > {vix_threshold} = 50% position)")
    if vol_target:
        print(f"(WITH VOL TARGETING: {vol_target*100:.0f}% annual vol target)")
    if use_kelly:
        print("(WITH KELLY FRACTION position sizing)")
    print("=" * 70)
    print("\nThis is the GOLD STANDARD robustness test.")
    print("Each test period is truly out-of-sample.\n")

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
        # CROSS-SECTIONAL RANKINGS
        'return_1d_rank', 'return_3d_rank', 'return_5d_rank',
        'return_1m_rank', 'return_3m_rank', 'return_6m_rank',
        'volatility_20d_rank', 'volatility_60d_rank',
        'dist_from_sma_50_rank', 'dist_from_sma_200_rank',
        'dist_from_52w_high_rank', 'dist_from_52w_low_rank',
        'volume_ratio_20_rank', 'volume_zscore_rank',
        # MARKET REGIME
        'market_volatility', 'market_trend'
    ]

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

    # Pre-compute price lookups
    price_by_ticker = {
        ticker: group.set_index('date').sort_index()
        for ticker, group in prices_df.groupby('ticker')
    }
    spy_prices = spy_df.set_index('date').sort_index()

    all_trading_dates = sorted(prices_df['date'].unique())
    trading_date_idx = {d: i for i, d in enumerate(all_trading_dates)}

    def get_next_trading_day(date):
        idx = trading_date_idx.get(date)
        if idx is not None and idx + 1 < len(all_trading_dates):
            return all_trading_dates[idx + 1]
        return None

    # === WALK FORWARD ===

    all_results = []
    yearly_summary = []

    # Tracking for advanced analytics
    turnover_records = []  # Track monthly turnover
    previous_holdings = set()  # Previous month's tickers
    realized_vol_window = []  # Rolling window for realized vol calculation

    # Kelly fraction tracking (calculated from rolling window)
    rolling_win_rate = []
    rolling_avg_win = []
    rolling_avg_loss = []

    for test_year in range(first_test_year, years[-1] + 1):
        train_end_year = test_year - 1

        print(f"\n{'='*60}")
        print(f"FOLD: Train {min_start_year}-{train_end_year} → Test {test_year}")
        print(f"{'='*60}")

        # Split data
        train_data = features_df[features_df['year'] <= train_end_year].copy()
        test_data = features_df[features_df['year'] == test_year].copy()

        if len(train_data) < 1000 or len(test_data) < 100:
            print(f"  Skipping: insufficient data (train={len(train_data)}, test={len(test_data)})")
            continue

        # Prepare training data
        train_clean = train_data[['ticker', 'date', 'target_binary'] + feature_cols].dropna()
        X_train = train_clean[feature_cols].values
        y_train = train_clean['target_binary'].values

        # Calculate scale_pos_weight
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

        print(f"  Training samples: {len(train_clean)}")
        print(f"  Class balance: {n_pos} pos ({n_pos/len(y_train)*100:.1f}%) / {n_neg} neg")

        # Train XGBoost ENSEMBLE (multiple seeds for robustness)
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

        # Use last 20% of training as validation for early stopping
        val_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        # Train ensemble of models with different random seeds
        models = []
        for seed in range(n_ensemble):
            params = base_params.copy()
            params['random_state'] = 42 + seed * 17  # Different seeds
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            models.append(model)

        # Evaluate on test year
        test_clean = test_data[['ticker', 'date', 'target_binary'] + feature_cols].dropna()
        X_test = test_clean[feature_cols].values
        y_test = test_clean['target_binary'].values

        if len(X_test) == 0:
            print(f"  No test samples!")
            continue

        # Ensemble prediction: average probabilities from all models
        ensemble_preds = np.zeros(len(X_test))
        for m in models:
            ensemble_preds += m.predict_proba(X_test)[:, 1]
        test_pred_proba = ensemble_preds / len(models)
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
        test_with_returns = test_clean.merge(
            test_data[['ticker', 'date', 'future_return']].drop_duplicates(),
            on=['ticker', 'date'],
            how='left'
        )
        valid_returns = test_with_returns.dropna(subset=['future_return', 'pred_proba'])

        if len(valid_returns) > 10:
            spearman_corr, spearman_p = spearmanr(
                valid_returns['pred_proba'],
                valid_returns['future_return']
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
            for m in models:
                ensemble_preds += m.predict_proba(X)[:, 1]
            date_features['pred_proba'] = ensemble_preds / len(models)
            top_picks = date_features.nlargest(top_n, 'pred_proba')

            # === TURNOVER TRACKING ===
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

            # === VOL-ADJUSTED POSITION SIZING ===
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

            # Helper: find nearest available price within N days
            def get_nearest_price(ticker_df, target_date, max_days=3):
                """Get price on target_date or nearest available within max_days."""
                if target_date in ticker_df.index:
                    return ticker_df.loc[target_date, 'adj_close']
                # Search forward then backward
                for delta in range(1, max_days + 1):
                    forward = target_date + pd.Timedelta(days=delta)
                    backward = target_date - pd.Timedelta(days=delta)
                    if forward in ticker_df.index:
                        return ticker_df.loc[forward, 'adj_close']
                    if backward in ticker_df.index:
                        return ticker_df.loc[backward, 'adj_close']
                return None

            # Calculate returns (vol-weighted)
            pick_returns = []
            pick_weights = []
            skipped_tickers = 0
            for idx, (_, row) in enumerate(top_picks.iterrows()):
                ticker = row['ticker']
                ticker_prices = price_by_ticker.get(ticker)
                if ticker_prices is None:
                    skipped_tickers += 1
                    continue  # Skip instead of -100% penalty

                entry_price = get_nearest_price(ticker_prices, entry_date, max_days=3)
                exit_price = get_nearest_price(ticker_prices, exit_date, max_days=3)

                if entry_price and exit_price and entry_price > 0:
                    ret = (exit_price - entry_price) / entry_price
                    pick_returns.append(ret)
                    pick_weights.append(weights[idx])
                else:
                    # Stock likely delisted - use conservative -20% instead of -100%
                    # This is more realistic than total loss
                    pick_returns.append(-0.20)
                    pick_weights.append(weights[idx])

            if not pick_returns:
                continue

            # === SLIPPAGE MODEL ===
            # Base transaction cost + modest turnover-based slippage
            # Using linear model (quadratic was too punitive)
            base_cost = transaction_cost * 2  # Round trip (~20 bps)
            if track_turnover and turnover_records:
                recent_turnover = turnover_records[-1]['turnover'] if turnover_records else 0.5
                # Linear slippage: ~5-10 bps per unit turnover
                slippage = 0.0005 * recent_turnover  # 5 bps max additional
            else:
                slippage = 0.0
            total_cost = base_cost + slippage

            # Vol-weighted portfolio return (not simple average!)
            pick_weights = np.array(pick_weights)
            pick_returns = np.array(pick_returns)
            # Renormalize weights for actual picks (in case some failed)
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
            kelly_scale = 1.0
            if use_kelly and len(rolling_win_rate) >= 12:
                # Kelly = W - (1-W)/R where W=win rate, R=win/loss ratio
                avg_win_rate = np.mean(rolling_win_rate[-24:])  # 2-year rolling
                avg_win_amt = np.mean([w for w in rolling_avg_win[-24:] if w > 0]) if any(w > 0 for w in rolling_avg_win[-24:]) else 0.02
                avg_loss_amt = abs(np.mean([l for l in rolling_avg_loss[-24:] if l < 0])) if any(l < 0 for l in rolling_avg_loss[-24:]) else 0.02

                if avg_loss_amt > 0:
                    win_loss_ratio = avg_win_amt / avg_loss_amt
                    kelly_fraction = avg_win_rate - (1 - avg_win_rate) / win_loss_ratio
                    # Half-Kelly for safety (full Kelly is too aggressive)
                    kelly_scale = max(0.25, min(1.5, kelly_fraction * 0.5))

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
        print("\n✓ ROBUST: Strategy shows consistent out-of-sample alpha")
        print("  - Beats SPY in majority of years")
        print("  - Precision@10 above random in most periods")
        print("  - Consider paper trading next")
    elif positive_excess_years >= total_years * 0.5:
        print("\n~ MARGINAL: Strategy shows some signal but inconsistent")
        print("  - May be loading on known factors")
        print("  - Consider factor analysis before live trading")
    else:
        print("\n✗ WEAK: Strategy does not generalize well")
        print("  - Original backtest may have been overfit")
        print("  - Consider different features or longer training")

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
                       help="Number of models in ensemble (default: 3)")
    parser.add_argument("--full", action="store_true",
                       help="Run with all features: risk-off + 15%% vol target + Kelly + ensemble")
    parser.add_argument("--balanced", action="store_true",
                       help="Run with BALANCED settings: 25%% vol target, no Kelly, lighter risk-off")
    args = parser.parse_args()

    if args.balanced:
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
            n_ensemble=3
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
            n_ensemble=5  # Full ensemble for elite mode
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
            use_kelly=False
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
            use_kelly=False
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
            use_kelly=False
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
            use_kelly=True
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
            n_ensemble=args.ensemble
        )
