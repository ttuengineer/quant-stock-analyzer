"""
Generate This Month's Stock Picks.

Uses the trained model to rank all S&P 500 stocks and output
the top 20 picks for the current month.

Includes RISK-OFF protection:
- Checks VIX level (high volatility = reduce exposure)
- Checks SPY trend (below 200-day MA = defensive mode)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import hashlib
import yfinance as yf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stock_analyzer.database import Database


def get_market_regime():
    """
    Check market conditions for risk-off signals.

    Returns:
        dict with:
        - vix: current VIX level
        - spy_above_200ma: True if SPY > 200-day MA
        - risk_level: 'NORMAL', 'ELEVATED', or 'HIGH_RISK'
        - position_scale: 1.0, 0.5, or 0.25 based on risk
    """
    print("\n=== CHECKING MARKET REGIME ===")

    # Get VIX
    try:
        vix = yf.Ticker("^VIX")
        vix_data = vix.history(period="5d")
        current_vix = vix_data['Close'].iloc[-1]
        print(f"VIX: {current_vix:.1f}")
    except:
        current_vix = 20  # Default if can't fetch
        print("VIX: Unable to fetch (assuming 20)")

    # Get SPY and 200-day MA
    try:
        spy = yf.Ticker("SPY")
        spy_data = spy.history(period="1y")
        current_spy = spy_data['Close'].iloc[-1]
        ma_200 = spy_data['Close'].tail(200).mean()
        spy_above_200ma = current_spy > ma_200
        spy_pct_from_ma = (current_spy - ma_200) / ma_200 * 100
        print(f"SPY: ${current_spy:.2f} ({spy_pct_from_ma:+.1f}% from 200-MA)")
        print(f"SPY > 200-MA: {'Yes' if spy_above_200ma else 'NO - CAUTION'}")
    except:
        spy_above_200ma = True
        spy_pct_from_ma = 0
        print("SPY: Unable to fetch (assuming above 200-MA)")

    # Determine risk level
    if current_vix > 30 or not spy_above_200ma:
        risk_level = "HIGH_RISK"
        position_scale = 0.25  # Only 25% exposure
        print(f"\n*** HIGH RISK ENVIRONMENT ***")
        print(f"Recommendation: Reduce position sizes to 25%")
    elif current_vix > 20:
        risk_level = "ELEVATED"
        position_scale = 0.5  # 50% exposure
        print(f"\n* ELEVATED RISK *")
        print(f"Recommendation: Reduce position sizes to 50%")
    else:
        risk_level = "NORMAL"
        position_scale = 1.0  # Full exposure
        print(f"\nMarket regime: NORMAL")
        print(f"Full position sizes OK")

    return {
        'vix': current_vix,
        'spy_above_200ma': spy_above_200ma,
        'spy_pct_from_ma': spy_pct_from_ma,
        'risk_level': risk_level,
        'position_scale': position_scale
    }


def load_model():
    """Load trained model and metadata, compute model hash for audit trail."""
    model_path = Path("models/xgboost_classifier.pkl")
    metadata_path = Path("models/model_metadata.pkl")

    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run train_model.py first.")

    with open(model_path, 'rb') as f:
        model_bytes = f.read()
        model = pickle.loads(model_bytes)
        # Compute hash for audit trail
        model_hash = hashlib.sha256(model_bytes).hexdigest()[:12]

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    metadata['model_hash'] = model_hash
    return model, metadata


def compute_live_features(tickers: list) -> pd.DataFrame:
    """
    Compute features for current date using live data.

    Uses yf.download in batch for speed and fixes off-by-one return logic.
    """
    print(f"\nFetching live data for {len(tickers)} stocks (batch download)...")

    # Download 1 year of daily data for all tickers in one request (MUCH faster)
    raw = yf.download(tickers, period="1y", group_by='ticker', threads=True, progress=True)

    all_features = []
    failed = []

    def get_ticker_df(tkr):
        """Safely get DataFrame for a single ticker."""
        try:
            if len(tickers) == 1:
                df = raw.copy()  # type: ignore
            else:
                df = raw[tkr].copy()  # type: ignore
            df = df.dropna(how='all')
            return df
        except Exception:
            return None

    print("Processing features...")
    for i, ticker in enumerate(tickers):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(tickers)}")

        df = get_ticker_df(ticker)
        if df is None or df.shape[0] < 100:
            failed.append(ticker)
            continue

        try:
            df = df.sort_index()
            close = df['Close']
            current_price = float(close.iloc[-1])

            # === RETURNS (fixed off-by-one using pct_change) ===
            pct = close.pct_change().fillna(0)

            # Compound returns over N days
            return_1d = float(pct.iloc[-1])
            return_3d = float((1 + pct.iloc[-3:]).prod() - 1) if len(pct) >= 3 else None  # type: ignore
            return_5d = float((1 + pct.iloc[-5:]).prod() - 1) if len(pct) >= 5 else None  # type: ignore
            return_1m = float((1 + pct.iloc[-21:]).prod() - 1) if len(pct) >= 21 else None  # type: ignore
            return_3m = float((1 + pct.iloc[-63:]).prod() - 1) if len(pct) >= 63 else None  # type: ignore
            return_6m = float((1 + pct.iloc[-126:]).prod() - 1) if len(pct) >= 126 else None  # type: ignore

            # === VOLATILITY (annualized) ===
            ret_20 = pct.tail(20)
            vol_20d = float(ret_20.std() * np.sqrt(252)) if len(ret_20) > 1 else None
            ret_60 = pct.tail(60)
            vol_60d = float(ret_60.std() * np.sqrt(252)) if len(ret_60) > 1 else None

            # === SMA DISTANCES ===
            sma_50 = float(close.tail(50).mean()) if len(close) >= 50 else None
            sma_200 = float(close.tail(200).mean()) if len(close) >= 200 else None
            dist_sma_50 = (current_price - sma_50) / sma_50 if sma_50 and sma_50 > 0 else None
            dist_sma_200 = (current_price - sma_200) / sma_200 if sma_200 and sma_200 > 0 else None

            # === 52-WEEK HIGH/LOW ===
            high_52w = float(close.max())
            low_52w = float(close.min())
            dist_52w_high = (current_price - high_52w) / high_52w if high_52w != 0 else None
            dist_52w_low = (current_price - low_52w) / low_52w if low_52w != 0 else None

            # === VOLUME FEATURES ===
            vol_series = df['Volume'].dropna()
            current_vol = float(vol_series.iloc[-1]) if len(vol_series) > 0 else None
            avg_vol_20 = float(vol_series.tail(20).mean()) if len(vol_series) >= 1 else None
            vol_std_20 = float(vol_series.tail(20).std()) if len(vol_series) >= 2 else None
            volume_ratio_20 = (current_vol / avg_vol_20) if avg_vol_20 and avg_vol_20 > 0 else None  # type: ignore
            volume_zscore = ((current_vol - avg_vol_20) / vol_std_20) if (vol_std_20 and vol_std_20 > 0 and current_vol is not None and avg_vol_20 is not None) else None  # type: ignore

            all_features.append({
                'ticker': ticker,
                'current_price': current_price,
                'return_1d': return_1d,
                'return_3d': return_3d,
                'return_5d': return_5d,
                'return_1m': return_1m,
                'return_3m': return_3m,
                'return_6m': return_6m,
                'volatility_20d': vol_20d,
                'volatility_60d': vol_60d,
                'dist_from_sma_50': dist_sma_50,
                'dist_from_sma_200': dist_sma_200,
                'dist_from_52w_high': dist_52w_high,
                'dist_from_52w_low': dist_52w_low,
                'volume_ratio_20': volume_ratio_20,
                'volume_zscore': volume_zscore
            })

        except Exception:
            failed.append(ticker)
            continue

    if failed:
        print(f"  Failed to process: {len(failed)} stocks")

    df = pd.DataFrame(all_features)

    # Add cross-sectional rankings (same names as training)
    rank_cols = [
        'return_1d', 'return_3d', 'return_5d',
        'return_1m', 'return_3m', 'return_6m',
        'volatility_20d', 'volatility_60d',
        'dist_from_sma_50', 'dist_from_sma_200',
        'dist_from_52w_high', 'dist_from_52w_low',
        'volume_ratio_20', 'volume_zscore'
    ]

    for col in rank_cols:
        if col in df.columns:
            df[f'{col}_rank'] = df[col].rank(pct=True, na_option='keep')

    return df


def get_sp500_tickers():
    """Get current S&P 500 tickers from database."""
    db = Database(db_path="data/stocks.db", use_supabase=False)

    # Get unique tickers from prices table
    prices = db.get_prices()
    tickers = prices['ticker'].unique().tolist()
    db.close()

    # Remove any benchmark tickers
    tickers = [t for t in tickers if t not in ['SPY', '^VIX', '^GSPC']]

    return tickers


def generate_picks(top_n: int = 20, use_live_data: bool = False):
    """
    Generate stock picks for current month.

    Args:
        top_n: Number of stocks to recommend
        use_live_data: If True, fetch fresh data from Yahoo Finance
                       If False, use latest data from database
    """
    print("=" * 70)
    print("MONTHLY STOCK PICKS GENERATOR")
    print("=" * 70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Check market regime
    regime = get_market_regime()

    # Load model
    print("\n=== LOADING MODEL ===")
    model, metadata = load_model()
    feature_cols = metadata['feature_cols']
    model_hash = metadata.get('model_hash', 'unknown')
    print(f"Model trained: {metadata['trained_date']}")
    print(f"Model hash: {model_hash}")
    print(f"Features: {len(feature_cols)}")

    # Get features
    if use_live_data:
        print("\n=== FETCHING LIVE DATA ===")
        tickers = get_sp500_tickers()
        features_df = compute_live_features(tickers)

        # Add market regime features (use current values)
        features_df['market_volatility'] = regime['vix'] / 100  # Normalize
        features_df['market_trend'] = regime['spy_pct_from_ma'] / 100
    else:
        print("\n=== USING DATABASE (latest available date) ===")
        db = Database(db_path="data/stocks.db", use_supabase=False)
        features_df = db.get_features()
        db.close()

        # Get latest date
        features_df['date'] = pd.to_datetime(features_df['date'])
        latest_date = features_df['date'].max()
        features_df = features_df[features_df['date'] == latest_date].copy()
        print(f"Using data from: {latest_date.date()}")

    print(f"Stocks available: {len(features_df)}")

    # Prepare features for prediction
    # Remove any target columns
    for col in ['target_binary', 'target_excess', 'future_return', 'future_return_rank']:
        if col in features_df.columns:
            features_df = features_df.drop(columns=[col])

    # Ensure all feature columns exist
    missing_cols = [c for c in feature_cols if c not in features_df.columns]
    if missing_cols:
        print(f"\nWarning: Missing features: {missing_cols}")
        for col in missing_cols:
            features_df[col] = 0  # Fill with 0

    # Make predictions
    print("\n=== GENERATING PREDICTIONS ===")
    X = features_df[feature_cols].reindex(columns=feature_cols).astype(np.float32).values
    X = np.nan_to_num(X, nan=0.0)

    features_df['pred_proba'] = model.predict_proba(X)[:, 1]

    # Rank and select top N
    features_df = features_df.sort_values('pred_proba', ascending=False)
    top_picks = features_df.head(top_n).copy()

    # === VOLATILITY-ADJUSTED POSITION SIZING ===
    # High-vol stocks get smaller weights, low-vol stocks get larger weights
    # Formula: weight = base * (avg_vol / stock_vol), then normalize to 100%

    vol_col = 'volatility_20d'
    if vol_col in top_picks.columns:
        vols = top_picks[vol_col].fillna(top_picks[vol_col].median())
        vols = vols.clip(lower=0.10)  # Floor at 10% to avoid division issues
        avg_vol = vols.mean()

        # Inverse vol weighting: lower vol = higher weight
        raw_weights = avg_vol / vols
        # Normalize to sum to 1
        top_picks['vol_weight'] = raw_weights / raw_weights.sum()
        # Apply risk scale
        top_picks['final_weight'] = top_picks['vol_weight'] * regime['position_scale'] * 100
    else:
        # Equal weight fallback
        top_picks['vol_weight'] = 1 / top_n
        top_picks['final_weight'] = regime['position_scale'] * 100 / top_n

    # === OUTPUT RESULTS ===

    print("\n" + "=" * 70)
    print(f"TOP {top_n} STOCK PICKS (Volatility-Adjusted Weights)")
    print("=" * 70)

    if regime['risk_level'] != 'NORMAL':
        print(f"\n*** {regime['risk_level']} - Position sizes reduced to {regime['position_scale']*100:.0f}% ***\n")

    print(f"\n{'Rank':<6} {'Ticker':<8} {'Score':>8} {'Weight':>8} {'Price':>10} {'Vol':>8} {'52W High':>10}")
    print("-" * 76)

    for i, (_, row) in enumerate(top_picks.iterrows(), 1):
        ticker = row['ticker']
        score = row['pred_proba']
        weight = row['final_weight']
        price = row.get('current_price', row.get('adj_close', 'N/A'))
        vol = row.get('volatility_20d', 0) or 0
        dist_high = row.get('dist_from_52w_high', 0) or 0

        price_str = f"${price:.2f}" if isinstance(price, (int, float)) else "N/A"

        print(f"{i:<6} {ticker:<8} {score:>7.1%} {weight:>7.1f}% {price_str:>10} {vol:>7.0f}% {dist_high:>+9.1%}")

    # Save to file
    output_dir = Path("picks")
    output_dir.mkdir(exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    output_file = output_dir / f"picks_{date_str}.csv"

    # Prepare output dataframe with full audit trail
    output_df = top_picks[['ticker', 'pred_proba']].copy()
    output_df['rank'] = range(1, len(output_df) + 1)
    output_df['weight_pct'] = top_picks['final_weight'].values
    output_df['volatility'] = top_picks.get('volatility_20d', pd.Series([None]*len(top_picks))).values
    output_df['date'] = date_str
    output_df['data_source'] = 'live' if use_live_data else 'database'
    output_df['risk_level'] = regime['risk_level']
    output_df['position_scale'] = regime['position_scale']
    output_df['vix'] = regime['vix']
    output_df['spy_trend'] = regime['spy_pct_from_ma']
    output_df['model_hash'] = model_hash
    output_df['model_trained'] = metadata['trained_date']
    output_df['generated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    output_df.to_csv(output_file, index=False)
    print(f"\n\nPicks saved to: {output_file}")

    # === PORTFOLIO ALLOCATION ===

    print("\n" + "=" * 70)
    print("SUGGESTED PORTFOLIO ALLOCATION (Vol-Adjusted)")
    print("=" * 70)

    total_weight = top_picks['final_weight'].sum()
    cash_pct = 100 - total_weight
    min_weight = top_picks['final_weight'].min()
    max_weight = top_picks['final_weight'].max()

    print(f"\nRisk Level: {regime['risk_level']}")
    print(f"Position Scale: {regime['position_scale']*100:.0f}%")
    print(f"\nTotal equity exposure: {total_weight:.1f}%")
    print(f"Cash reserve: {cash_pct:.1f}%")
    print(f"\nWeight range: {min_weight:.1f}% - {max_weight:.1f}%")
    print(f"(High-vol stocks get smaller weights)")

    if regime['risk_level'] != 'NORMAL':
        print(f"\n*** Due to {regime['risk_level']} conditions, holding {cash_pct:.0f}% cash ***")

    # List tickers for easy copy-paste
    print("\n" + "=" * 70)
    print("TICKERS (copy-paste ready)")
    print("=" * 70)
    ticker_list = ", ".join(top_picks['ticker'].tolist())
    print(f"\n{ticker_list}")

    return top_picks


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate monthly stock picks")
    parser.add_argument("--top", type=int, default=20, help="Number of picks (default: 20)")
    parser.add_argument("--live", action="store_true", help="Fetch live data from Yahoo Finance")

    args = parser.parse_args()

    generate_picks(top_n=args.top, use_live_data=args.live)


if __name__ == "__main__":
    main()
