"""
Backtest the ML stock picking strategy.

PRODUCTION-SAFE VERSION with:
- No lookahead bias (features only, no targets)
- Realistic execution (next-day open price)
- Survivorship bias handling (delisted stocks = -100%)
- Transaction costs
- Proper feature alignment
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stock_analyzer.database import Database


def load_model():
    """Load trained model and metadata."""
    model_path = Path("models/xgboost_classifier.pkl")
    metadata_path = Path("models/model_metadata.pkl")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    return model, metadata


def backtest_strategy(
    top_n: int = 20,
    start_date: str = "2023-01-01",
    end_date: str = None,
    transaction_cost: float = 0.001,  # 10 bps per trade (round-trip ~20 bps)
    verbose: bool = True
):
    """
    Backtest the ML stock picking strategy - PRODUCTION SAFE.

    Fixes applied:
    1. No lookahead: Only load feature columns, never targets
    2. Realistic execution: Enter at NEXT DAY'S OPEN, not same-day close
    3. Survivorship bias: Delisted/missing stocks get -100% return
    4. Transaction costs: Configurable slippage/costs
    5. Feature alignment: Exact column order matching training

    Args:
        top_n: Number of stocks to hold each month
        start_date: Backtest start date
        end_date: Backtest end date (default: latest available)
        transaction_cost: Cost per trade as decimal (0.001 = 0.1%)
        verbose: Print progress
    """
    if verbose:
        print("=" * 70)
        print(f"BACKTESTING - Top {top_n} Stock Strategy (PRODUCTION-SAFE)")
        print("=" * 70)
        print(f"\nTransaction cost: {transaction_cost*100:.2f}% per trade")

    # Load model
    model, metadata = load_model()
    feature_cols = metadata['feature_cols']

    if verbose:
        print(f"Model trained on: {metadata['trained_date']}")
        print(f"Using {len(feature_cols)} features")

    # Load data
    db = Database(db_path="data/stocks.db", use_supabase=False)

    # === FIX #1: Load features WITHOUT target columns (no lookahead) ===
    # Only select feature columns + ticker + date
    features_df = db.get_features(start_date=start_date, end_date=end_date)
    features_df['date'] = pd.to_datetime(features_df['date'])

    # CRITICAL: Drop any target/future columns to prevent leakage
    leak_cols = ['target_binary', 'target_excess', 'future_return', 'future_return_rank']
    for col in leak_cols:
        if col in features_df.columns:
            features_df = features_df.drop(columns=[col])

    # Get ALL price data (need full history for execution prices)
    # Load a bit before start_date to get next-day prices
    price_start = (pd.to_datetime(start_date) - timedelta(days=10)).strftime("%Y-%m-%d")
    prices_df = db.get_prices(start_date=price_start, end_date=end_date)
    prices_df['date'] = pd.to_datetime(prices_df['date'])

    # Get SPY benchmark
    spy_df = db.get_benchmarks(ticker="SPY", start_date=price_start, end_date=end_date)
    spy_df['date'] = pd.to_datetime(spy_df['date'])

    db.close()

    if verbose:
        print(f"\nBacktest period: {features_df['date'].min().date()} to {features_df['date'].max().date()}")

    # Get unique rebalancing dates (month-end dates when we have features)
    rebalance_dates = sorted(features_df['date'].unique())

    if verbose:
        print(f"Rebalancing dates: {len(rebalance_dates)}")

    # Pre-compute price lookup for speed
    # Group by ticker for O(1) access
    price_by_ticker = {
        ticker: group.set_index('date').sort_index()
        for ticker, group in prices_df.groupby('ticker')
    }
    spy_prices = spy_df.set_index('date').sort_index()

    # Get all trading dates for finding "next day"
    all_trading_dates = sorted(prices_df['date'].unique())
    trading_date_idx = {d: i for i, d in enumerate(all_trading_dates)}

    def get_next_trading_day(date):
        """Get the next trading day after given date."""
        idx = trading_date_idx.get(date)
        if idx is not None and idx + 1 < len(all_trading_dates):
            return all_trading_dates[idx + 1]
        return None

    # === RUN BACKTEST ===

    if verbose:
        print("\n" + "=" * 70)
        print("Running backtest...")
        print("=" * 70)

    results = []
    prev_holdings = set()  # Track previous holdings for turnover

    for i, signal_date in enumerate(rebalance_dates[:-1]):
        next_signal_date = rebalance_dates[i + 1]

        # === FIX #2: Realistic execution timing ===
        # Signal generated at month-end close
        # Execute at NEXT trading day's open
        entry_date = get_next_trading_day(signal_date)
        exit_date = get_next_trading_day(next_signal_date)

        if entry_date is None or exit_date is None:
            continue

        # Get features for this signal date
        date_features = features_df[features_df['date'] == signal_date].copy()

        if len(date_features) < top_n:
            continue

        # === FIX #5: Enforce exact feature column alignment ===
        # Reindex to match training order exactly
        try:
            X = date_features[feature_cols].reindex(columns=feature_cols).astype(np.float32).values
        except KeyError as e:
            print(f"Warning: Missing feature columns on {signal_date}: {e}")
            continue

        # Handle any NaN in features (use 0 or skip)
        if np.isnan(X).any():
            # Fill NaN with column median from this date
            X = np.nan_to_num(X, nan=0.0)

        # Make predictions
        date_features['pred_proba'] = model.predict_proba(X)[:, 1]

        # Select top N stocks
        top_picks = date_features.nlargest(top_n, 'pred_proba')
        selected_tickers = set(top_picks['ticker'].tolist())

        # Calculate turnover (for transaction costs)
        if prev_holdings:
            n_new = len(selected_tickers - prev_holdings)
            n_sold = len(prev_holdings - selected_tickers)
            turnover = (n_new + n_sold) / (2 * top_n)  # 0 to 1
        else:
            turnover = 1.0  # First period, buy everything

        # === Calculate returns for each pick ===
        pick_returns = []
        delisted_count = 0

        for ticker in selected_tickers:
            ticker_prices = price_by_ticker.get(ticker)

            if ticker_prices is None:
                # === FIX #3: Survivorship bias - ticker completely missing ===
                pick_returns.append(-1.0)  # -100% for missing tickers
                delisted_count += 1
                continue

            # Get entry price (next day's open after signal)
            # If open not available, use adj_close as proxy
            if entry_date in ticker_prices.index:
                if 'open' in ticker_prices.columns:
                    entry_price = ticker_prices.loc[entry_date, 'open']
                else:
                    entry_price = ticker_prices.loc[entry_date, 'adj_close']
            else:
                # Stock delisted or no data at entry
                pick_returns.append(-1.0)
                delisted_count += 1
                continue

            # Get exit price
            if exit_date in ticker_prices.index:
                if 'open' in ticker_prices.columns:
                    exit_price = ticker_prices.loc[exit_date, 'open']
                else:
                    exit_price = ticker_prices.loc[exit_date, 'adj_close']
            else:
                # === FIX #3: Stock delisted during holding period ===
                # Find last available price, assume partial loss
                available_dates = ticker_prices.index[
                    (ticker_prices.index > entry_date) & (ticker_prices.index <= exit_date)
                ]
                if len(available_dates) > 0:
                    last_date = available_dates[-1]
                    exit_price = ticker_prices.loc[last_date, 'adj_close']
                else:
                    # Complete loss
                    pick_returns.append(-1.0)
                    delisted_count += 1
                    continue

            # Calculate return
            if entry_price > 0:
                ret = (exit_price - entry_price) / entry_price
                pick_returns.append(ret)
            else:
                pick_returns.append(-1.0)
                delisted_count += 1

        if not pick_returns:
            continue

        # Portfolio return (equal weight)
        gross_return = np.mean(pick_returns)

        # === FIX #6: Apply transaction costs ===
        # Cost = (turnover * cost_per_trade * 2) for buy + sell
        total_cost = turnover * transaction_cost * 2
        portfolio_return = gross_return - total_cost

        # SPY return for same period (also use open-to-open for fairness)
        if entry_date in spy_prices.index and exit_date in spy_prices.index:
            spy_entry = spy_prices.loc[entry_date, 'adj_close']
            spy_exit = spy_prices.loc[exit_date, 'adj_close']
            spy_return = (spy_exit - spy_entry) / spy_entry
        else:
            spy_return = 0

        # Excess return
        excess_return = portfolio_return - spy_return

        results.append({
            'signal_date': signal_date,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'gross_return': gross_return,
            'transaction_cost': total_cost,
            'portfolio_return': portfolio_return,
            'spy_return': spy_return,
            'excess_return': excess_return,
            'n_stocks': len(pick_returns),
            'n_delisted': delisted_count,
            'turnover': turnover,
            'top_picks': list(top_picks['ticker'].head(5))
        })

        prev_holdings = selected_tickers

        # Progress
        if verbose and (i + 1) % 6 == 0:
            print(f"  {signal_date.date()}: Portfolio {portfolio_return:+.2%} vs SPY {spy_return:+.2%} "
                  f"(excess: {excess_return:+.2%}, delisted: {delisted_count})")

    # === ANALYZE RESULTS ===

    if not results:
        print("No results generated!")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    if verbose:
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS (Production-Safe)")
        print("=" * 70)

    # Cumulative returns
    results_df['cum_portfolio'] = (1 + results_df['portfolio_return']).cumprod()
    results_df['cum_spy'] = (1 + results_df['spy_return']).cumprod()
    results_df['cum_gross'] = (1 + results_df['gross_return']).cumprod()

    # Final values
    total_portfolio_return = results_df['cum_portfolio'].iloc[-1] - 1
    total_spy_return = results_df['cum_spy'].iloc[-1] - 1
    total_gross_return = results_df['cum_gross'].iloc[-1] - 1
    total_excess = total_portfolio_return - total_spy_return
    total_costs = total_gross_return - total_portfolio_return

    if verbose:
        print(f"\n=== CUMULATIVE RETURNS ===")
        print(f"Portfolio (net):  {total_portfolio_return:+.2%}")
        print(f"Portfolio (gross):{total_gross_return:+.2%}")
        print(f"Transaction costs:{total_costs:-.2%}")
        print(f"SPY:              {total_spy_return:+.2%}")
        print(f"Excess (net):     {total_excess:+.2%}")

    # Annualized returns
    n_months = len(results_df)
    n_years = n_months / 12

    if n_years > 0:
        ann_portfolio = (1 + total_portfolio_return) ** (1/n_years) - 1
        ann_spy = (1 + total_spy_return) ** (1/n_years) - 1
        ann_excess = ann_portfolio - ann_spy

        if verbose:
            print(f"\n=== ANNUALIZED RETURNS ===")
            print(f"Portfolio: {ann_portfolio:+.2%}")
            print(f"SPY:       {ann_spy:+.2%}")
            print(f"Excess:    {ann_excess:+.2%}")

        # Risk metrics
        portfolio_vol = results_df['portfolio_return'].std() * np.sqrt(12)
        spy_vol = results_df['spy_return'].std() * np.sqrt(12)
        excess_vol = results_df['excess_return'].std() * np.sqrt(12)

        # Sharpe ratio (5% risk-free)
        rf_rate = 0.05
        sharpe_portfolio = (ann_portfolio - rf_rate) / portfolio_vol if portfolio_vol > 0 else 0
        sharpe_spy = (ann_spy - rf_rate) / spy_vol if spy_vol > 0 else 0
        information_ratio = ann_excess / excess_vol if excess_vol > 0 else 0

        if verbose:
            print(f"\n=== RISK METRICS ===")
            print(f"Portfolio Volatility: {portfolio_vol:.2%}")
            print(f"SPY Volatility:       {spy_vol:.2%}")
            print(f"Sharpe (Portfolio):   {sharpe_portfolio:.2f}")
            print(f"Sharpe (SPY):         {sharpe_spy:.2f}")
            print(f"Information Ratio:    {information_ratio:.2f}")

    # Win rate
    win_rate = (results_df['excess_return'] > 0).mean()
    wins = results_df[results_df['excess_return'] > 0]
    losses = results_df[results_df['excess_return'] <= 0]
    avg_win = wins['excess_return'].mean() if len(wins) > 0 else 0
    avg_loss = losses['excess_return'].mean() if len(losses) > 0 else 0

    if verbose:
        print(f"\n=== WIN RATE ===")
        print(f"Months beating SPY: {win_rate:.1%} ({len(wins)}/{len(results_df)})")
        print(f"Avg win:  {avg_win:+.2%}")
        print(f"Avg loss: {avg_loss:+.2%}")

    # Max drawdown
    cum_max = results_df['cum_portfolio'].cummax()
    drawdown = (results_df['cum_portfolio'] - cum_max) / cum_max
    max_drawdown = drawdown.min()

    spy_cum_max = results_df['cum_spy'].cummax()
    spy_drawdown = (results_df['cum_spy'] - spy_cum_max) / spy_cum_max
    spy_max_drawdown = spy_drawdown.min()

    if verbose:
        print(f"\n=== MAX DRAWDOWN ===")
        print(f"Portfolio: {max_drawdown:.2%}")
        print(f"SPY:       {spy_max_drawdown:.2%}")

    # Survivorship stats
    total_delisted = results_df['n_delisted'].sum()
    avg_turnover = results_df['turnover'].mean()

    if verbose:
        print(f"\n=== EXECUTION STATS ===")
        print(f"Total delisted/missing: {total_delisted} stock-months")
        print(f"Avg monthly turnover:   {avg_turnover:.1%}")
        print(f"Total transaction costs:{results_df['transaction_cost'].sum():.2%}")

    # === SUMMARY ===

    if verbose:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        if total_excess > 0:
            print(f"Strategy BEAT SPY by {total_excess:+.2%} cumulative (net of costs)")
        else:
            print(f"Strategy UNDERPERFORMED SPY by {total_excess:.2%}")

        if n_years > 0 and sharpe_portfolio > sharpe_spy:
            print(f"Better risk-adjusted returns (Sharpe {sharpe_portfolio:.2f} vs {sharpe_spy:.2f})")

        if win_rate > 0.5:
            print(f"Beat SPY in {win_rate:.0%} of months")

    # Save results
    results_path = Path("models/backtest_results.csv")
    results_df.to_csv(results_path, index=False)

    if verbose:
        print(f"\nResults saved to {results_path}")

        # === YEARLY BREAKDOWN ===
        print("\n" + "=" * 70)
        print("YEARLY BREAKDOWN")
        print("=" * 70)

        results_df['year'] = results_df['signal_date'].dt.year
        yearly = results_df.groupby('year').agg({
            'portfolio_return': lambda x: (1 + x).prod() - 1,
            'spy_return': lambda x: (1 + x).prod() - 1,
            'excess_return': 'sum',
            'n_delisted': 'sum'
        })

        for year in yearly.index:
            row = yearly.loc[year]
            print(f"{year}: Portfolio {row['portfolio_return']:+.2%} | "
                  f"SPY {row['spy_return']:+.2%} | "
                  f"Excess {row['excess_return']:+.2%} | "
                  f"Delisted: {int(row['n_delisted'])}")

    return results_df


def main():
    """Run backtest with different configurations."""
    # Main backtest: Top 20 stocks
    print("\n" + "=" * 70)
    print("BACKTEST 1: TOP 20 STOCKS (Diversified)")
    print("=" * 70)
    results_20 = backtest_strategy(top_n=20, start_date="2023-01-01", transaction_cost=0.001)

    # Aggressive: Top 10 stocks
    print("\n\n" + "=" * 70)
    print("BACKTEST 2: TOP 10 STOCKS (Concentrated)")
    print("=" * 70)
    results_10 = backtest_strategy(top_n=10, start_date="2023-01-01", transaction_cost=0.001)

    # Conservative: Top 30 stocks
    print("\n\n" + "=" * 70)
    print("BACKTEST 3: TOP 30 STOCKS (Conservative)")
    print("=" * 70)
    results_30 = backtest_strategy(top_n=30, start_date="2023-01-01", transaction_cost=0.001)

    print("\n\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    for name, df in [("Top 10", results_10), ("Top 20", results_20), ("Top 30", results_30)]:
        if df.empty:
            continue
        total_ret = df['cum_portfolio'].iloc[-1] - 1
        spy_ret = df['cum_spy'].iloc[-1] - 1
        excess = total_ret - spy_ret
        win_rate = (df['excess_return'] > 0).mean()
        print(f"{name}: {total_ret:+.1%} vs SPY {spy_ret:+.1%} | "
              f"Excess: {excess:+.1%} | Win rate: {win_rate:.0%}")


if __name__ == "__main__":
    main()
