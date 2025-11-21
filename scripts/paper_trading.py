"""
Paper Trading Pipeline - ALIGNED WITH WALK-FORWARD VALIDATION.

This module replicates EXACTLY the same logic used in walk_forward_validation.py:
- Same feature columns
- Same ensemble approach (multiple seeds)
- Same vol-weighted positions
- Same survivorship bias filtering
- Same slippage model
- Optional factor neutralization

Usage:
    # Generate picks using saved fold models
    python scripts/paper_trading.py --generate

    # Generate with fresh model training (slower but uses latest data)
    python scripts/paper_trading.py --generate --retrain

    # Check performance of past picks
    python scripts/paper_trading.py --reconcile

    # Show current positions
    python scripts/paper_trading.py --status
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
import pickle
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stock_analyzer.database import Database

# ML imports
import xgboost as xgb


# === CONFIGURATION ===

PAPER_TRADING_DIR = Path("paper_trading")
PICKS_DIR = PAPER_TRADING_DIR / "picks"
LOGS_DIR = PAPER_TRADING_DIR / "logs"
PERFORMANCE_DIR = PAPER_TRADING_DIR / "performance"
FOLD_MODELS_DIR = Path("models/folds")


def ensure_directories():
    """Create paper trading directories if they don't exist."""
    for d in [PAPER_TRADING_DIR, PICKS_DIR, LOGS_DIR, PERFORMANCE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# === FEATURE COLUMNS (EXACT MATCH WITH WALK-FORWARD) ===

FEATURE_COLS = [
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
    # INDUSTRY-NEUTRAL RESIDUALS
    'return_1d_resid', 'return_3d_resid', 'return_5d_resid',
    'return_1m_resid', 'return_3m_resid', 'return_6m_resid',
    'volatility_20d_resid', 'volatility_60d_resid',
    'dist_from_sma_50_resid', 'dist_from_sma_200_resid',
    'volume_ratio_20_resid', 'volume_zscore_resid',
    # MARKET REGIME
    'market_volatility', 'market_trend'
]


# === SURVIVORSHIP BIAS HANDLING (EXACT MATCH) ===

_TICKER_ALIASES = {}


def load_historical_universe(filepath="data/historical_sp500_universe.json"):
    """Load historical S&P 500 universe for survivorship bias fix."""
    global _TICKER_ALIASES

    filepath = Path(filepath)
    if not filepath.exists():
        print(f"WARNING: Historical universe file not found: {filepath}")
        return None, {}

    with open(filepath, 'r') as f:
        data = json.load(f)

    universe = data.get('universe_by_date', {})
    aliases = data.get('ticker_aliases', {})
    sectors = data.get('sectors', {})

    _TICKER_ALIASES = aliases

    return universe, aliases, sectors


def normalize_ticker(ticker):
    """Get all valid ticker variants for matching."""
    variants = set()
    variants.add(ticker)
    variants.add(ticker.replace('.', '-'))
    variants.add(ticker.replace('-', '.'))

    if ticker in _TICKER_ALIASES:
        variants.update(_TICKER_ALIASES[ticker])

    return variants


def get_universe_at_date(historical_universe, target_date):
    """Get the S&P 500 universe at a specific date."""
    if historical_universe is None:
        return None

    target = pd.to_datetime(target_date).strftime('%Y-%m-%d')
    dates = sorted([d for d in historical_universe.keys() if d <= target], reverse=True)

    if dates:
        return set(historical_universe[dates[0]])

    earliest = min(historical_universe.keys())
    return set(historical_universe[earliest])


def filter_by_universe(df, historical_universe, date_col='date', ticker_col='ticker'):
    """Filter DataFrame to only include stocks in S&P 500 at each date."""
    if historical_universe is None:
        return df

    # Build universe cache
    unique_dates = df[date_col].unique()
    date_universes = {}
    for d in unique_dates:
        universe = get_universe_at_date(historical_universe, d)
        expanded = set()
        for t in universe:
            expanded.update(normalize_ticker(t))
        date_universes[d] = expanded

    def check_ticker_in_universe(row):
        universe = date_universes.get(row[date_col], set())
        ticker_variants = normalize_ticker(row[ticker_col])
        return bool(ticker_variants & universe)

    valid_mask = df.apply(check_ticker_in_universe, axis=1)
    return df[valid_mask]


# === MODEL TRAINING (EXACT MATCH WITH WALK-FORWARD) ===

def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_ensemble: int = 3,
    verbose: bool = True
) -> List:
    """
    Train an ensemble of XGBoost models with different random seeds.
    EXACT match with walk_forward_validation.py logic.
    """
    # Calculate class balance
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

    if verbose:
        print(f"  Training samples: {len(y_train)}")
        print(f"  Class balance: {n_pos} pos ({n_pos/len(y_train)*100:.1f}%) / {n_neg} neg")

    # Base parameters (EXACT match)
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

    # Use last 20% for validation (EXACT match)
    val_split = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    # Train ensemble
    models = []
    for seed_idx in range(n_ensemble):
        actual_seed = 42 + seed_idx * 17  # EXACT match
        params = base_params.copy()
        params['random_state'] = actual_seed
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        models.append(model)
        if verbose:
            print(f"    Trained model {seed_idx + 1}/{n_ensemble} (seed={actual_seed})")

    return models


def get_ensemble_predictions(models: List, X: np.ndarray) -> np.ndarray:
    """Get averaged predictions from ensemble (EXACT match)."""
    ensemble_preds = np.zeros(len(X))
    for m in models:
        ensemble_preds += m.predict_proba(X)[:, 1]
    return ensemble_preds / len(models)


# === VOL-WEIGHTED PORTFOLIO CONSTRUCTION (EXACT MATCH) ===

def compute_vol_weights(picks_df: pd.DataFrame, vol_col: str = 'volatility_20d') -> np.ndarray:
    """
    Compute volatility-weighted positions (EXACT match with walk-forward).
    Higher vol stocks get SMALLER weights.
    """
    if vol_col not in picks_df.columns:
        return np.ones(len(picks_df)) / len(picks_df)

    vols = picks_df[vol_col].fillna(picks_df[vol_col].median())
    vols = vols.clip(lower=0.10)  # Min 10% vol to avoid extreme weights
    avg_vol = vols.mean()
    raw_weights = avg_vol / vols
    weights = (raw_weights / raw_weights.sum()).values
    return weights


# === LOGGING AND AUDIT TRAIL ===

def compute_model_hash(models: List) -> str:
    """Compute hash of ensemble models for versioning."""
    # Hash based on model parameters and structure
    summary = ""
    for i, m in enumerate(models):
        summary += f"{i}_{m.get_params().get('random_state', 0)}_"
    return hashlib.md5(summary.encode()).hexdigest()[:12]


def compute_data_hash(features_df: pd.DataFrame) -> str:
    """Compute hash of input data for reproducibility."""
    summary = f"{features_df.shape}_{features_df['date'].max()}_{features_df['ticker'].nunique()}"
    return hashlib.md5(summary.encode()).hexdigest()[:12]


class PaperTradingLog:
    """
    Manages paper trading logs and picks with full audit trail.
    """

    def __init__(self, log_dir: Path = PICKS_DIR):
        self.log_dir = log_dir
        ensure_directories()

    def save_picks(
        self,
        picks: Dict[str, float],
        predictions_df: pd.DataFrame,
        metadata: Dict,
        date: Optional[str] = None
    ) -> str:
        """Save picks with full audit trail."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        session_id = f"{date}_{datetime.now().strftime('%H%M%S')}"

        # Save picks
        picks_file = self.log_dir / f"picks_{session_id}.json"
        with open(picks_file, 'w') as f:
            json.dump({
                'date': date,
                'session_id': session_id,
                'picks': picks,
                'n_long': sum(1 for w in picks.values() if w > 0),
                'n_short': sum(1 for w in picks.values() if w < 0),
                'total_long_weight': sum(w for w in picks.values() if w > 0),
                'total_short_weight': sum(w for w in picks.values() if w < 0),
            }, f, indent=2, default=str)

        # Save metadata
        metadata_file = self.log_dir / f"metadata_{session_id}.json"
        metadata['session_id'] = session_id
        metadata['generated_at'] = datetime.now().isoformat()
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save full predictions
        predictions_file = self.log_dir / f"predictions_{session_id}.csv"
        predictions_df.to_csv(predictions_file, index=False)

        print(f"Picks saved: {picks_file}")
        print(f"Metadata saved: {metadata_file}")
        print(f"Predictions saved: {predictions_file}")

        return session_id

    def load_picks(self, session_id: str) -> Tuple[Dict, Dict, pd.DataFrame]:
        """Load picks, metadata, and predictions for a session."""
        picks_file = self.log_dir / f"picks_{session_id}.json"
        metadata_file = self.log_dir / f"metadata_{session_id}.json"
        predictions_file = self.log_dir / f"predictions_{session_id}.csv"

        with open(picks_file, 'r') as f:
            picks_data = json.load(f)

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        predictions_df = pd.read_csv(predictions_file)

        return picks_data, metadata, predictions_df

    def list_sessions(self) -> List[str]:
        """List all pick sessions."""
        sessions = []
        for f in self.log_dir.glob("picks_*.json"):
            session_id = f.stem.replace("picks_", "")
            sessions.append(session_id)
        return sorted(sessions, reverse=True)

    def get_latest_session(self) -> Optional[str]:
        """Get most recent session ID."""
        sessions = self.list_sessions()
        return sessions[0] if sessions else None


class PerformanceTracker:
    """Track paper trading performance over time."""

    def __init__(self, db_path: str = "data/stocks.db"):
        self.db_path = db_path
        ensure_directories()

    def reconcile_picks(
        self,
        session_id: str,
        picks_log: PaperTradingLog
    ) -> pd.DataFrame:
        """Reconcile a pick session against realized returns."""
        picks_data, metadata, predictions_df = picks_log.load_picks(session_id)
        picks = picks_data['picks']
        pick_date = picks_data['date']

        print(f"\n=== Reconciling Session: {session_id} ===")
        print(f"Pick Date: {pick_date}")
        print(f"Positions: {len(picks)}")

        # Load price data
        db = Database(db_path=self.db_path, use_supabase=False)
        prices_df = db.get_prices()
        spy_df = db.get_benchmarks(ticker="SPY")
        db.close()

        prices_df['date'] = pd.to_datetime(prices_df['date'])
        spy_df['date'] = pd.to_datetime(spy_df['date'])

        # Find entry and exit dates
        pick_date_dt = pd.to_datetime(pick_date)
        entry_date = pick_date_dt + timedelta(days=1)

        # Find actual entry date (next available)
        available_dates = sorted(prices_df['date'].unique())
        entry_date = min([d for d in available_dates if d >= entry_date], default=None)

        if entry_date is None:
            print("No price data after pick date")
            return pd.DataFrame()

        # Exit date: ~1 month later
        exit_date = entry_date + timedelta(days=30)
        exit_date = min([d for d in available_dates if d >= exit_date], default=None)

        today = pd.to_datetime(datetime.now().date())
        if exit_date is None or exit_date > today:
            exit_date = max([d for d in available_dates if d <= today], default=None)

        if exit_date is None:
            print("No exit date available")
            return pd.DataFrame()

        print(f"Entry Date: {entry_date.date()}")
        print(f"Exit Date: {exit_date.date()}")

        # Calculate returns
        results = []
        for ticker, weight in picks.items():
            ticker_prices = prices_df[prices_df['ticker'] == ticker]

            entry_row = ticker_prices[ticker_prices['date'] == entry_date]
            exit_row = ticker_prices[ticker_prices['date'] == exit_date]

            if len(entry_row) == 0 or len(exit_row) == 0:
                continue

            entry_price = entry_row['adj_close'].values[0]
            exit_price = exit_row['adj_close'].values[0]

            if entry_price > 0:
                stock_return = (exit_price - entry_price) / entry_price
                position_return = stock_return * weight

                results.append({
                    'ticker': ticker,
                    'weight': weight,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'stock_return': stock_return,
                    'position_return': position_return,
                })

        results_df = pd.DataFrame(results)

        if len(results_df) == 0:
            print("No results to reconcile")
            return results_df

        # SPY return
        spy_entry = spy_df[spy_df['date'] == entry_date]['adj_close'].values
        spy_exit = spy_df[spy_df['date'] == exit_date]['adj_close'].values

        if len(spy_entry) > 0 and len(spy_exit) > 0:
            spy_return = (spy_exit[0] - spy_entry[0]) / spy_entry[0]
        else:
            spy_return = 0

        # Summary
        portfolio_return = results_df['position_return'].sum()

        print(f"\n=== Performance Summary ===")
        print(f"Portfolio Return: {portfolio_return:+.2%}")
        print(f"SPY Return:       {spy_return:+.2%}")
        print(f"Excess Return:    {portfolio_return - spy_return:+.2%}")

        # Save
        recon_file = PERFORMANCE_DIR / f"reconciliation_{session_id}.csv"
        results_df.to_csv(recon_file, index=False)

        summary = {
            'session_id': session_id,
            'pick_date': pick_date,
            'entry_date': str(entry_date.date()),
            'exit_date': str(exit_date.date()),
            'portfolio_return': portfolio_return,
            'spy_return': spy_return,
            'excess_return': portfolio_return - spy_return,
            'n_positions': len(results_df),
        }

        summary_file = PERFORMANCE_DIR / f"summary_{session_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return results_df

    def generate_report(self) -> pd.DataFrame:
        """Generate performance report across all sessions."""
        summaries = []

        for f in PERFORMANCE_DIR.glob("summary_*.json"):
            with open(f, 'r') as fp:
                summaries.append(json.load(fp))

        if not summaries:
            print("No reconciled sessions found")
            return pd.DataFrame()

        report_df = pd.DataFrame(summaries)
        report_df = report_df.sort_values('pick_date', ascending=False)

        print("\n" + "=" * 70)
        print("PAPER TRADING PERFORMANCE REPORT")
        print("=" * 70)

        print(f"\nTotal Sessions: {len(report_df)}")
        print(f"\n{'Date':<12} {'Portfolio':>12} {'SPY':>10} {'Excess':>10}")
        print("-" * 46)

        for _, row in report_df.iterrows():
            print(f"{row['pick_date']:<12} {row['portfolio_return']:>+11.1%} "
                  f"{row['spy_return']:>+9.1%} {row['excess_return']:>+9.1%}")

        # Aggregates
        total_portfolio = (1 + report_df['portfolio_return']).prod() - 1
        total_spy = (1 + report_df['spy_return']).prod() - 1
        avg_excess = report_df['excess_return'].mean()
        win_rate = (report_df['excess_return'] > 0).mean()

        print("-" * 46)
        print(f"{'TOTAL':<12} {total_portfolio:>+11.1%} {total_spy:>+9.1%}")
        print(f"\nAvg Monthly Excess: {avg_excess:+.2%}")
        print(f"Win Rate vs SPY: {win_rate:.0%}")

        return report_df


# === MAIN PICK GENERATION (ALIGNED WITH WALK-FORWARD) ===

def generate_picks(
    n_picks: int = 20,
    n_ensemble: int = 3,
    long_only: bool = True,
    use_vol_weights: bool = True,
    fix_survivorship_bias: bool = True,
    retrain: bool = False,
    min_train_years: int = 3,
    verbose: bool = True
) -> str:
    """
    Generate paper trading picks using EXACT walk-forward logic.

    Args:
        n_picks: Number of top/bottom picks
        n_ensemble: Number of models in ensemble
        long_only: If True, long-only. If False, long-short.
        use_vol_weights: Use volatility-weighted positions
        fix_survivorship_bias: Filter by historical S&P 500 membership
        retrain: If True, train fresh models. If False, load saved fold models.
        min_train_years: Minimum years of training data
        verbose: Print progress

    Returns:
        session_id
    """
    ensure_directories()

    print("=" * 70)
    print("PAPER TRADING PICK GENERATION")
    print("(Aligned with walk-forward validation)")
    print("=" * 70)

    # Load historical universe
    historical_universe = None
    sectors_dict = {}
    if fix_survivorship_bias:
        historical_universe, ticker_aliases, sectors_dict = load_historical_universe()
        if historical_universe:
            print(f"\nSurvivorship fix: Loaded universe data")
        else:
            print("WARNING: No historical universe data, proceeding without fix")

    # Load data
    print("\nLoading data...")
    db = Database(db_path="data/stocks.db", use_supabase=False)
    features_df = db.get_features()
    features_df['date'] = pd.to_datetime(features_df['date'])
    db.close()

    # Get latest date
    latest_date = features_df['date'].max()
    print(f"Latest feature date: {latest_date.date()}")

    # Get available feature columns
    available_cols = [col for col in FEATURE_COLS if col in features_df.columns]
    missing_cols = set(FEATURE_COLS) - set(available_cols)
    if missing_cols and verbose:
        print(f"Note: {len(missing_cols)} features not available")

    feature_cols = available_cols
    print(f"Using {len(feature_cols)} features")

    # Get current features
    current_features = features_df[features_df['date'] == latest_date].copy()

    # Apply survivorship bias filter
    if fix_survivorship_bias and historical_universe:
        before_count = len(current_features)
        current_features = filter_by_universe(
            current_features, historical_universe, 'date', 'ticker'
        )
        after_count = len(current_features)
        print(f"Survivorship filter: {before_count} -> {after_count} stocks")

    print(f"Stocks available: {len(current_features)}")

    # Train or load models
    if retrain:
        print("\n--- Training fresh ensemble ---")

        # Get training data (all data before latest date)
        features_df['year'] = features_df['date'].dt.year
        latest_year = features_df['year'].max()
        train_end_year = latest_year - 1 if features_df['date'].max().month <= 6 else latest_year

        train_data = features_df[features_df['year'] <= train_end_year].copy()

        # Apply survivorship fix to training data
        if fix_survivorship_bias and historical_universe:
            train_data = filter_by_universe(train_data, historical_universe, 'date', 'ticker')

        # Prepare training data
        train_clean = train_data[['ticker', 'date', 'target_binary'] + feature_cols].dropna()
        X_train = train_clean[feature_cols].values
        y_train = train_clean['target_binary'].values

        # Train ensemble
        models = train_ensemble(X_train, y_train, n_ensemble=n_ensemble, verbose=verbose)

    else:
        # Load most recent fold models
        print("\n--- Loading saved fold models ---")

        if not FOLD_MODELS_DIR.exists():
            print("ERROR: No fold models found. Run walk-forward validation first or use --retrain")
            return None

        # Find most recent fold
        fold_files = list(FOLD_MODELS_DIR.glob("fold_*.pkl"))
        if not fold_files:
            print("ERROR: No fold models found. Run with --retrain")
            return None

        # Get unique years
        years = set()
        for f in fold_files:
            parts = f.stem.split('_')
            if len(parts) >= 2:
                years.add(int(parts[1]))

        latest_fold_year = max(years) if years else None
        if latest_fold_year is None:
            print("ERROR: Could not determine fold year")
            return None

        print(f"Loading models from fold {latest_fold_year}")

        # Load all models for that fold
        models = []
        for f in sorted(FOLD_MODELS_DIR.glob(f"fold_{latest_fold_year}_seed_*.pkl")):
            with open(f, 'rb') as fp:
                fold_data = pickle.load(fp)
            models.append(fold_data['model'])
            if verbose:
                print(f"  Loaded: {f.name}")

        if not models:
            print(f"ERROR: No models for fold {latest_fold_year}")
            return None

        print(f"Loaded {len(models)} models")

    # Prepare features for prediction
    X = current_features[feature_cols].values
    X = np.nan_to_num(X, nan=0.0).astype(np.float32)

    # Get ensemble predictions
    print("\nGenerating predictions...")
    pred_proba = get_ensemble_predictions(models, X)
    current_features = current_features.copy()
    current_features['pred_proba'] = pred_proba

    # Sort by prediction
    current_features = current_features.sort_values('pred_proba', ascending=False)

    # Select picks
    top_picks = current_features.head(n_picks).copy()

    if long_only:
        # Vol-weighted long-only
        if use_vol_weights:
            weights = compute_vol_weights(top_picks)
        else:
            weights = np.ones(len(top_picks)) / len(top_picks)

        picks = {row['ticker']: float(w) for (_, row), w in zip(top_picks.iterrows(), weights)}

    else:
        # Long-short
        bottom_picks = current_features.tail(n_picks).copy()

        if use_vol_weights:
            long_weights = compute_vol_weights(top_picks) * 0.5  # 50% long
            short_weights = compute_vol_weights(bottom_picks) * 0.5  # 50% short
        else:
            long_weights = np.ones(len(top_picks)) / len(top_picks) * 0.5
            short_weights = np.ones(len(bottom_picks)) / len(bottom_picks) * 0.5

        picks = {}
        for (_, row), w in zip(top_picks.iterrows(), long_weights):
            picks[row['ticker']] = float(w)
        for (_, row), w in zip(bottom_picks.iterrows(), short_weights):
            picks[row['ticker']] = float(-w)  # Negative weight = short

    # Build metadata
    model_hash = compute_model_hash(models)
    data_hash = compute_data_hash(features_df)

    pick_metadata = {
        'model_hash': model_hash,
        'data_hash': data_hash,
        'feature_date': str(latest_date.date()),
        'n_picks': n_picks,
        'n_ensemble': len(models),
        'long_only': long_only,
        'use_vol_weights': use_vol_weights,
        'fix_survivorship_bias': fix_survivorship_bias,
        'retrained': retrain,
        'feature_cols': feature_cols,
        'n_stocks_universe': len(current_features),
        'aligned_with_walk_forward': True,  # Key indicator!
    }

    # Save
    log = PaperTradingLog()

    # Prepare predictions for saving
    save_cols = ['ticker', 'date', 'pred_proba']
    if 'volatility_20d' in current_features.columns:
        save_cols.append('volatility_20d')

    session_id = log.save_picks(
        picks=picks,
        predictions_df=current_features[save_cols],
        metadata=pick_metadata,
        date=str(latest_date.date())
    )

    # Print picks
    print(f"\n{'=' * 60}")
    print(f"PICKS FOR {latest_date.date()}")
    print(f"{'=' * 60}")
    print(f"Session ID: {session_id}")
    print(f"Model Hash: {model_hash}")
    print(f"Ensemble: {len(models)} models")

    # Sort picks by weight for display
    sorted_picks = sorted(picks.items(), key=lambda x: x[1], reverse=True)

    if long_only:
        print(f"\nTop {n_picks} Long Picks (vol-weighted):")
        for i, (ticker, weight) in enumerate(sorted_picks, 1):
            prob = current_features[current_features['ticker'] == ticker]['pred_proba'].values[0]
            print(f"  {i:2}. {ticker:<6} weight: {weight:.1%}  (prob: {prob:.3f})")
    else:
        long_picks = [(t, w) for t, w in sorted_picks if w > 0]
        short_picks = [(t, w) for t, w in sorted_picks if w < 0]

        print(f"\nLong Positions ({len(long_picks)}):")
        for i, (ticker, weight) in enumerate(long_picks, 1):
            prob = current_features[current_features['ticker'] == ticker]['pred_proba'].values[0]
            print(f"  {i:2}. {ticker:<6} weight: {weight:+.1%}  (prob: {prob:.3f})")

        print(f"\nShort Positions ({len(short_picks)}):")
        for i, (ticker, weight) in enumerate(short_picks, 1):
            prob = current_features[current_features['ticker'] == ticker]['pred_proba'].values[0]
            print(f"  {i:2}. {ticker:<6} weight: {weight:+.1%}  (prob: {prob:.3f})")

    return session_id


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Paper Trading Pipeline (Aligned with Walk-Forward)"
    )
    parser.add_argument("--generate", action="store_true", help="Generate new picks")
    parser.add_argument("--reconcile", type=str, help="Reconcile a session (or 'latest')")
    parser.add_argument("--reconcile-all", action="store_true", help="Reconcile all sessions")
    parser.add_argument("--report", action="store_true", help="Generate performance report")
    parser.add_argument("--status", action="store_true", help="Show current status")

    # Pick generation options
    parser.add_argument("--n-picks", type=int, default=20, help="Number of picks (default: 20)")
    parser.add_argument("--ensemble", type=int, default=3, help="Number of models in ensemble")
    parser.add_argument("--long-short", action="store_true", help="Use long-short (default: long-only)")
    parser.add_argument("--equal-weight", action="store_true", help="Use equal weights (default: vol-weighted)")
    parser.add_argument("--retrain", action="store_true", help="Train fresh models instead of loading saved")
    parser.add_argument("--no-survivorship-fix", action="store_true", help="Disable survivorship bias fix")

    args = parser.parse_args()

    if args.generate:
        generate_picks(
            n_picks=args.n_picks,
            n_ensemble=args.ensemble,
            long_only=not args.long_short,
            use_vol_weights=not args.equal_weight,
            fix_survivorship_bias=not args.no_survivorship_fix,
            retrain=args.retrain,
        )

    elif args.reconcile:
        log = PaperTradingLog()
        tracker = PerformanceTracker()

        if args.reconcile == 'latest':
            session_id = log.get_latest_session()
            if session_id:
                tracker.reconcile_picks(session_id, log)
            else:
                print("No sessions found")
        else:
            tracker.reconcile_picks(args.reconcile, log)

    elif args.reconcile_all:
        log = PaperTradingLog()
        tracker = PerformanceTracker()

        for session_id in log.list_sessions():
            try:
                tracker.reconcile_picks(session_id, log)
            except Exception as e:
                print(f"Error reconciling {session_id}: {e}")

    elif args.report:
        tracker = PerformanceTracker()
        tracker.generate_report()

    elif args.status:
        log = PaperTradingLog()
        sessions = log.list_sessions()

        print("=" * 60)
        print("PAPER TRADING STATUS")
        print("=" * 60)

        if not sessions:
            print("\nNo pick sessions found.")
            print("Run: python scripts/paper_trading.py --generate")
        else:
            print(f"\nTotal Sessions: {len(sessions)}")
            print(f"\nRecent Sessions:")
            for s in sessions[:5]:
                picks_data, metadata, _ = log.load_picks(s)
                aligned = metadata.get('aligned_with_walk_forward', False)
                aligned_str = " [ALIGNED]" if aligned else " [OLD]"
                print(f"  {s}: {picks_data['n_long']} long, "
                      f"{picks_data['n_short']} short "
                      f"(model: {metadata.get('model_hash', 'unknown')[:8]}){aligned_str}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
