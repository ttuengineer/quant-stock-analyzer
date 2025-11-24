"""
Debug script to analyze 2024 portfolio selections.

This will help us understand:
1. What stocks were selected each month in 2024
2. Were mega-caps (AAPL, MSFT, NVDA, GOOGL, META) included?
3. What were their prediction scores?
4. Did the overlay actually activate?
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stock_analyzer.database import Database
from stock_analyzer.features import FeatureEngineer
from mega_cap_overlay import apply_mega_cap_overlay, MEGA_CAPS

def load_2024_model():
    """Load the 2024 trained model."""
    model_files = list(Path("models/folds").glob("fold_2024_seed_*.pkl"))
    if not model_files:
        print("ERROR: No 2024 models found!")
        return None

    print(f"\nFound {len(model_files)} 2024 models")

    # Load the first one for testing
    with open(model_files[0], 'rb') as f:
        model = pickle.load(f)

    print(f"Loaded model: {model_files[0].name}")
    return model

def get_2024_predictions(date_str='2024-01-31'):
    """Get predictions for a specific date in 2024."""
    print(f"\n{'='*70}")
    print(f"ANALYZING PREDICTIONS FOR {date_str}")
    print(f"{'='*70}")

    # Load model
    model = load_2024_model()
    if model is None:
        return None

    # Load data
    print("\nLoading data...")
    db = Database()
    engineer = FeatureEngineer()

    # Get data for prediction date
    date = pd.to_datetime(date_str)

    # Load historical data (need 252 days before for features)
    start_date = date - pd.Timedelta(days=400)

    df = db.fetch_stock_data(start_date=start_date, end_date=date)

    if df.empty:
        print("ERROR: No data loaded!")
        return None

    print(f"Loaded {len(df)} rows of data")

    # Engineer features
    print("\nEngineering features...")
    features_df = engineer.engineer_features(df)

    # Get features for the specific date
    date_features = features_df[features_df.index.get_level_values('date') == date].copy()

    if date_features.empty:
        print(f"ERROR: No features for {date_str}")
        return None

    print(f"Found features for {len(date_features)} stocks on {date_str}")

    # Make predictions
    print("\nMaking predictions...")

    feature_cols = [col for col in date_features.columns
                   if col not in ['ticker', 'forward_return', 'target', 'sector', 'industry']]

    X = date_features[feature_cols].fillna(0)

    predictions = model.predict_proba(X)[:, 1]

    date_features['pred_proba'] = predictions
    date_features['score'] = predictions * 100
    date_features = date_features.reset_index()

    # Sort by score
    date_features = date_features.sort_values('score', ascending=False)

    # Check for mega-caps
    mega_cap_tickers = list(MEGA_CAPS.keys())
    date_features['is_mega_cap'] = date_features['ticker'].isin(mega_cap_tickers)

    print("\n" + "="*70)
    print("TOP 30 PREDICTIONS")
    print("="*70)
    print(f"\n{'Rank':<6} {'Ticker':<8} {'Score':<8} {'MegaCap':<10} {'Sector':<20}")
    print("-" * 70)

    for i, row in date_features.head(30).iterrows():
        mega_flag = "[MEGA-CAP]" if row['is_mega_cap'] else ""
        sector = row.get('sector', 'Unknown')[:18]
        print(f"{i+1:<6} {row['ticker']:<8} {row['score']:<8.1f} {mega_flag:<10} {sector:<20}")

    # Mega-cap analysis
    print("\n" + "="*70)
    print("MEGA-CAP ANALYSIS")
    print("="*70)

    mega_cap_preds = date_features[date_features['is_mega_cap']].copy()

    if len(mega_cap_preds) == 0:
        print("\n❌ NO MEGA-CAPS FOUND IN PREDICTIONS!")
        return date_features

    print(f"\nFound {len(mega_cap_preds)} mega-caps in predictions:\n")
    print(f"{'Ticker':<8} {'Score':<8} {'Rank':<8} {'Above 40?':<12}")
    print("-" * 50)

    for _, row in mega_cap_preds.iterrows():
        ticker = row['ticker']
        score = row['score']
        rank = date_features[date_features['ticker'] == ticker].index[0] + 1
        above_40 = "✅ YES" if score >= 40 else "❌ NO"
        print(f"{ticker:<8} {score:<8.1f} {rank:<8} {above_40:<12}")

    # Test overlay
    print("\n" + "="*70)
    print("TESTING MEGA-CAP OVERLAY")
    print("="*70)

    predictions_df = date_features[['ticker', 'score', 'pred_proba']].copy()
    predictions_df.columns = ['ticker', 'score', 'prediction']

    portfolio, diagnostics = apply_mega_cap_overlay(
        predictions_df,
        top_n=20,
        min_score_threshold=40.0,
        mega_cap_min_allocation=0.25,
        mega_cap_weight_method='hybrid',
        force_include_top_k=5,
        verbose=True
    )

    print("\n" + "="*70)
    print("FINAL PORTFOLIO (Top 20)")
    print("="*70)

    for i, row in portfolio.head(20).iterrows():
        mega_flag = "[MEGA-CAP]" if row['is_mega_cap'] else ""
        print(f"{row['ticker']:<8} Weight: {row['weight']:<6.1%}  Score: {row['score']:<6.1f}  {mega_flag}")

    return date_features

if __name__ == "__main__":
    # Test multiple 2024 dates
    test_dates = [
        '2024-01-31',  # January (AI hype starting)
        '2024-06-28',  # Mid-year
        '2024-11-29',  # End of year
    ]

    for date in test_dates:
        try:
            result = get_2024_predictions(date)
            print("\n" + "="*70)
            print()
        except Exception as e:
            print(f"\nERROR processing {date}: {e}")
            import traceback
            traceback.print_exc()
