"""
Train XGBoost classifier to predict stock outperformance.

Predicts: P(stock beats SPY over next month)
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt


def main():
    """Train the prediction model."""
    print("=" * 70)
    print("MODEL TRAINING - XGBoost Classifier (Top-Decile Prediction)")
    print("=" * 70)
    print("\nTraining model to predict: P(stock is in TOP 10% of 3-month returns)\n")

    # Initialize database
    print("Loading features from database...")
    db = Database(db_path="data/stocks.db", use_supabase=False)

    # Load all features
    features_df = db.get_features()
    db.close()

    print(f"Loaded {len(features_df)} samples")
    print(f"Date range: {features_df['date'].min()} to {features_df['date'].max()}")

    # === PREPARE DATA ===

    # Define feature columns - using BOTH raw values AND rankings!
    # Raw values preserve magnitude, rankings show relative position
    feature_cols = [
        # === RAW VALUES (preserve magnitude - important for extremes!) ===
        # Short-term reversal (very predictive!)
        'return_1d', 'return_3d', 'return_5d',
        # Momentum
        'return_1m', 'return_3m', 'return_6m',
        # Volatility
        'volatility_20d', 'volatility_60d',
        # Technical levels
        'dist_from_sma_50', 'dist_from_sma_200',
        'dist_from_52w_high', 'dist_from_52w_low',
        # Volume shocks
        'volume_ratio_20', 'volume_zscore',

        # === CROSS-SECTIONAL RANKINGS (relative position vs peers) ===
        'return_1d_rank', 'return_3d_rank', 'return_5d_rank',
        'return_1m_rank', 'return_3m_rank', 'return_6m_rank',
        'volatility_20d_rank', 'volatility_60d_rank',
        'dist_from_sma_50_rank', 'dist_from_sma_200_rank',
        'dist_from_52w_high_rank', 'dist_from_52w_low_rank',
        'volume_ratio_20_rank', 'volume_zscore_rank',

        # === INDUSTRY-NEUTRAL RESIDUALS (professional upgrade!) ===
        'return_1d_resid', 'return_3d_resid', 'return_5d_resid',
        'return_1m_resid', 'return_3m_resid', 'return_6m_resid',
        'volatility_20d_resid', 'volatility_60d_resid',
        'dist_from_sma_50_resid', 'dist_from_sma_200_resid',
        'volume_ratio_20_resid', 'volume_zscore_resid',

        # === MARKET REGIME ===
        'market_volatility', 'market_trend'
    ]

    # Filter to only columns that exist in data
    available_cols = [col for col in feature_cols if col in features_df.columns]
    missing_cols = set(feature_cols) - set(available_cols)
    if missing_cols:
        print(f"Note: {len(missing_cols)} features not available")
    feature_cols = available_cols

    print(f"\nUsing {len(feature_cols)} features:")
    print("  - RAW: reversal (1-5d), momentum, volatility, technicals, volume")
    print("  - RANKS: cross-sectional rankings vs peers")
    print("  - REGIME: market volatility + trend")
    print("\nPredicting: 3-MONTH outperformance vs SPY")

    # Drop samples with missing features or targets
    clean_df = features_df[['ticker', 'date', 'target_binary'] + feature_cols].copy()
    clean_df = clean_df.dropna()

    print(f"\nAfter cleaning: {len(clean_df)} samples")

    # === TRAIN/VALIDATION/TEST SPLIT (time-based) ===

    # Sort by date
    clean_df['date'] = pd.to_datetime(clean_df['date'])
    clean_df = clean_df.sort_values('date')

    # Split dates
    train_cutoff = '2021-01-01'
    val_cutoff = '2023-01-01'

    train = clean_df[clean_df['date'] < train_cutoff]
    val = clean_df[(clean_df['date'] >= train_cutoff) & (clean_df['date'] < val_cutoff)]
    test = clean_df[clean_df['date'] >= val_cutoff]

    print(f"\nTrain: {len(train)} samples ({train['date'].min()} to {train['date'].max()})")
    print(f"Val:   {len(val)} samples ({val['date'].min()} to {val['date'].max()})")
    print(f"Test:  {len(test)} samples ({test['date'].min()} to {test['date'].max()})")

    # Prepare X, y
    X_train = train[feature_cols].values
    y_train = train['target_binary'].values

    X_val = val[feature_cols].values
    y_val = val['target_binary'].values

    X_test = test[feature_cols].values
    y_test = test['target_binary'].values

    # === CLASS BALANCE CHECK ===
    # With top-decile labeling, we expect ~10% positive class
    n_pos = int(y_train.sum())  # type: ignore
    n_neg = len(y_train) - n_pos
    pos_ratio = n_pos / len(y_train)
    scale_pos_weight = n_neg / n_pos  # Balance the classes

    print(f"\n=== CLASS BALANCE ===")
    print(f"Positive (top 10%): {n_pos} ({pos_ratio*100:.1f}%)")
    print(f"Negative (rest):    {n_neg} ({(1-pos_ratio)*100:.1f}%)")
    print(f"scale_pos_weight:   {scale_pos_weight:.2f}")

    # === TRAIN XGBOOST ===

    print("\n" + "=" * 70)
    print("Training XGBoost Classifier...")
    print("=" * 70)

    # XGBoost parameters - tuned for TOP-DECILE prediction
    # Key: scale_pos_weight to handle class imbalance
    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,              # Slightly deeper for complex patterns
        'learning_rate': 0.01,       # Slower learning
        'n_estimators': 1000,        # More trees, but early stopping will kick in
        'subsample': 0.7,            # Row sampling
        'colsample_bytree': 0.7,     # Column sampling
        'min_child_weight': 5,       # Allow smaller leaves
        'gamma': 0.2,                # Some regularization
        'reg_alpha': 0.1,            # L1 regularization
        'reg_lambda': 1.0,           # L2 regularization
        'scale_pos_weight': scale_pos_weight,  # Handle class imbalance!
        'random_state': 42,
        'eval_metric': 'auc',
        'early_stopping_rounds': 50  # Stop if no improvement for 50 rounds
    }

    model = xgb.XGBClassifier(**params)

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    print("\nTraining complete!")

    # === EVALUATE ===

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    # Predictions
    train_pred_proba = model.predict_proba(X_train)[:, 1]
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    test_pred_proba = model.predict_proba(X_test)[:, 1]

    train_pred = (train_pred_proba > 0.5).astype(int)
    val_pred = (val_pred_proba > 0.5).astype(int)
    test_pred = (test_pred_proba > 0.5).astype(int)

    # Metrics
    print("\n=== AUC-ROC (higher is better, 0.5 = random) ===")
    train_auc = roc_auc_score(y_train, train_pred_proba)  # type: ignore
    val_auc = roc_auc_score(y_val, val_pred_proba)  # type: ignore
    test_auc = roc_auc_score(y_test, test_pred_proba)  # type: ignore

    print(f"Train AUC: {train_auc:.4f}")
    print(f"Val AUC:   {val_auc:.4f}")
    print(f"Test AUC:  {test_auc:.4f}")

    print("\n=== Accuracy (hit rate) ===")
    train_acc = accuracy_score(y_train, train_pred)  # type: ignore
    val_acc = accuracy_score(y_val, val_pred)  # type: ignore
    test_acc = accuracy_score(y_test, test_pred)  # type: ignore

    print(f"Train Accuracy: {train_acc:.4f} ({train_acc*100:.1f}%)")
    print(f"Val Accuracy:   {val_acc:.4f} ({val_acc*100:.1f}%)")
    print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.1f}%)")

    print("\n=== Test Set Classification Report ===")
    print(classification_report(y_test, test_pred, target_names=['Lose', 'Win']))  # type: ignore

    # Top decile performance (PRECISION@10)
    # With top-decile labeling: actual=1 means stock was in top 10% future returns
    print("\n=== PRECISION@10 (Key Metric!) ===")
    test_df = test.copy()
    test_df['pred_proba'] = test_pred_proba
    test_df['actual'] = y_test

    # Top 10% of MODEL'S predictions
    top_picks = test_df.nlargest(int(len(test_df) * 0.1), 'pred_proba')
    precision_at_10 = top_picks['actual'].mean()

    # Top 5% of predictions (more aggressive)
    top_5pct = test_df.nlargest(int(len(test_df) * 0.05), 'pred_proba')
    precision_at_5 = top_5pct['actual'].mean()

    print(f"Precision@10%: {precision_at_10:.4f} ({precision_at_10*100:.1f}%)")
    print(f"Precision@5%:  {precision_at_5:.4f} ({precision_at_5*100:.1f}%)")
    print(f"(Random would be 10%, good model should be 15-25%+)")
    print(f"Lift over random: {precision_at_10 / 0.1:.2f}x")

    top_decile_hit_rate = precision_at_10  # For backward compatibility

    # === FEATURE IMPORTANCE ===

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("\nTop 10 most important features:")
    print(feature_importance.head(10).to_string(index=False))

    # === SAVE MODEL ===

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "xgboost_classifier.pkl"
    metadata_path = model_dir / "model_metadata.pkl"

    print(f"\nSaving model to {model_path}...")

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'train_auc': train_auc,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'test_accuracy': test_acc,
        'top_decile_hit_rate': top_decile_hit_rate,
        'feature_importance': feature_importance.to_dict(),
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_samples': len(train),
        'params': params
    }

    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print("Model saved!")

    # === SUMMARY ===

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if test_auc > 0.55:
        print("Model shows predictive power (AUC > 0.55)")
    else:
        print("Model AUC is weak - but check Precision@10!")

    # With top-decile labeling, Precision@10 is the key metric
    # Random = 10%, so >15% is good, >20% is excellent
    if precision_at_10 > 0.20:
        print(f"EXCELLENT! Precision@10 = {precision_at_10*100:.1f}% (2x+ random)")
    elif precision_at_10 > 0.15:
        print(f"GOOD! Precision@10 = {precision_at_10*100:.1f}% (1.5x random)")
    elif precision_at_10 > 0.10:
        print(f"MARGINAL: Precision@10 = {precision_at_10*100:.1f}% (slightly better than random)")
    else:
        print(f"WEAK: Precision@10 = {precision_at_10*100:.1f}% (no better than random)")

    print(f"\nModel ready for backtesting!")
    print("Next step:")
    print("  python scripts/backtest_strategy.py")


if __name__ == "__main__":
    main()
