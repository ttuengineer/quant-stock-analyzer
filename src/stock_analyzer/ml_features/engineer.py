"""
Feature engineering for predictive ML models.

Computes momentum, value, quality, and market regime features from raw data.
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta

from ..database import Database
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """
    Engineer features for ML prediction.

    Computes features on monthly rebalancing dates (last trading day of each month).
    """

    def __init__(self, db: Database):
        """
        Initialize feature engineer.

        Args:
            db: Database instance
        """
        self.db = db

    def engineer_all_features(
        self,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        forward_window: int = 21  # ~1 month of trading days
    ):
        """
        Engineer all features for all stocks - OPTIMIZED VERSION.

        Uses vectorized pandas operations instead of slow loops.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info("=" * 60)
        logger.info("Starting feature engineering (FAST vectorized version)")
        logger.info("=" * 60)

        # 0. Load sector data for industry-neutral residuals
        logger.info("Loading sector data...")
        try:
            meta_df = self.db.get_meta()
            if meta_df is not None and len(meta_df) > 0 and 'sector' in meta_df.columns:
                self._sector_map = dict(zip(meta_df['ticker'], meta_df['sector']))
                unique_sectors = meta_df['sector'].nunique()
                logger.info(f"Loaded sector data for {len(self._sector_map)} stocks ({unique_sectors} sectors)")
            else:
                self._sector_map = {}
                logger.warning("No sector data found - skipping industry-neutral residuals")
        except Exception as e:
            self._sector_map = {}
            logger.warning(f"Failed to load sector data: {e}")

        # 1. Load all data
        logger.info("Loading price data...")
        prices = self.db.get_prices(start_date=start_date, end_date=end_date)
        prices['date'] = pd.to_datetime(prices['date'])
        prices = prices.sort_values(['ticker', 'date'])

        logger.info("Loading benchmark data (SPY)...")
        spy = self.db.get_benchmarks(ticker="SPY", start_date=start_date, end_date=end_date)
        spy['date'] = pd.to_datetime(spy['date'])
        spy = spy.sort_values('date')

        logger.info(f"Loaded {len(prices)} price records for {prices['ticker'].nunique()} stocks")

        # 2. Get monthly rebalancing dates
        logger.info("Identifying monthly rebalancing dates...")
        rebalance_dates = self._get_monthly_dates(prices)
        logger.info(f"Found {len(rebalance_dates)} monthly rebalancing dates")

        # 3. FAST: Compute all features using vectorized operations
        logger.info("Computing features (vectorized)...")
        features_df = self._compute_all_features_fast(prices, spy, rebalance_dates, forward_window)

        if features_df.empty:
            logger.warning("No features generated!")
            return pd.DataFrame()

        # Remove rows with NaN targets
        before_drop = len(features_df)
        features_df = features_df.dropna(subset=['future_return', 'target_binary'])
        after_drop = len(features_df)

        if before_drop > after_drop:
            logger.info(f"Dropped {before_drop - after_drop} samples with missing targets")

        logger.info(f"Engineered {len(features_df)} feature samples")

        # Convert Timestamp to string for SQLite compatibility
        features_df['date'] = features_df['date'].astype(str).str[:10]

        # 4. Save to database
        logger.info("Saving features to database...")
        self.db.insert_features(features_df)

        logger.info("=" * 60)
        logger.info("Feature engineering complete!")
        logger.info("=" * 60)

        return features_df

    def _compute_all_features_fast(self, prices, spy, rebalance_dates, forward_window):
        """
        FAST vectorized feature computation.
        """
        all_features = []

        # Pre-compute SPY features (same for all stocks on each date)
        spy_features = self._compute_spy_features(spy, rebalance_dates, forward_window)

        # Group prices by ticker for fast lookup
        ticker_groups = {ticker: group.set_index('date') for ticker, group in prices.groupby('ticker')}

        total_dates = len(rebalance_dates)
        for i, date in enumerate(rebalance_dates):
            if (i + 1) % 12 == 0:
                logger.info(f"Progress: {i+1}/{total_dates} dates ({date})")

            date = pd.to_datetime(date)
            date_features = []

            # Get SPY info for this date
            spy_info = spy_features.get(date, {})
            if not spy_info:
                continue

            for ticker, ticker_df in ticker_groups.items():
                try:
                    features = self._compute_ticker_features_fast(
                        ticker, date, ticker_df, spy_info, forward_window
                    )
                    if features:
                        date_features.append(features)
                except:
                    continue

            if date_features:
                df = pd.DataFrame(date_features)

                # Add cross-sectional rankings for ALL features
                # Keep BOTH raw values AND ranks (raw preserves magnitude!)
                rank_cols = [
                    # Short-term reversal
                    'return_1d', 'return_3d', 'return_5d',
                    # Momentum
                    'return_1m', 'return_3m', 'return_6m', 'return_12m',
                    # Volatility
                    'volatility_20d', 'volatility_60d',
                    # Technical
                    'dist_from_sma_50', 'dist_from_sma_200',
                    'dist_from_52w_high', 'dist_from_52w_low',
                    # Volume
                    'volume_ratio_20', 'volume_ratio_60', 'volume_zscore', 'volume_5d_ratio',
                    # Rolling z-scores (ChatGPT)
                    'return_1m_zscore', 'return_3m_zscore', 'vol_zscore_rolling', 'volume_zscore_rolling',
                    # Nonlinear interactions (ChatGPT)
                    'mom_vol_interaction', 'reversal_vol_interaction', 'sma_vol_interaction',
                    'high_mom_interaction', 'vol_regime_interaction'
                ]
                for col in rank_cols:
                    if col in df.columns:
                        df[f'{col}_rank'] = df[col].rank(pct=True, na_option='keep')

                # === INDUSTRY-NEUTRAL RESIDUALS (professional upgrade!) ===
                # For each feature, subtract the sector mean to remove industry effects
                # This reduces beta and improves signal-to-noise
                if hasattr(self, '_sector_map') and self._sector_map:
                    df['sector'] = df['ticker'].map(self._sector_map)

                    # Features to residualize
                    resid_cols = [
                        'return_1d', 'return_3d', 'return_5d',
                        'return_1m', 'return_3m', 'return_6m',
                        'volatility_20d', 'volatility_60d',
                        'dist_from_sma_50', 'dist_from_sma_200',
                        'volume_ratio_20', 'volume_zscore',
                        # Rolling z-scores (ChatGPT)
                        'return_1m_zscore', 'return_3m_zscore',
                        # Nonlinear interactions (ChatGPT)
                        'mom_vol_interaction', 'reversal_vol_interaction',
                        'vol_regime_interaction'
                    ]

                    for col in resid_cols:
                        if col in df.columns:
                            # Compute sector mean for this date
                            sector_means = df.groupby('sector')[col].transform('mean')
                            # Residual = raw value - sector mean
                            df[f'{col}_resid'] = df[col] - sector_means

                # === MONTHLY CROSS-SECTIONAL Z-SCORE NORMALIZATION (ChatGPT) ===
                # Normalize ALL features to z-scores within each month
                # This removes time-varying scale effects and makes features comparable
                zscore_cols = [
                    'return_1d', 'return_3d', 'return_5d',
                    'return_1m', 'return_3m', 'return_6m', 'return_12m',
                    'volatility_20d', 'volatility_60d',
                    'dist_from_sma_50', 'dist_from_sma_200',
                    'dist_from_52w_high', 'dist_from_52w_low',
                    'volume_ratio_20', 'volume_ratio_60', 'volume_zscore', 'volume_5d_ratio',
                    'return_1m_zscore', 'return_3m_zscore',
                    'mom_vol_interaction', 'reversal_vol_interaction',
                    'sma_vol_interaction', 'high_mom_interaction', 'vol_regime_interaction'
                ]
                for col in zscore_cols:
                    if col in df.columns:
                        col_mean = df[col].mean()
                        col_std = df[col].std()
                        if col_std > 0:
                            df[f'{col}_znorm'] = (df[col] - col_mean) / col_std
                        else:
                            df[f'{col}_znorm'] = 0.0

                # === MULTI-HORIZON SMOOTHED RESIDUAL RETURN TARGET ===
                # Use average of 1-month and 2-month returns for smoother, more stable signal
                # This reduces noise from single-month outliers
                if 'future_return' in df.columns:
                    # Create smoothed return (average of 1m and 2m if available)
                    if 'future_return_2m' in df.columns:
                        # Weighted average: 60% 1-month, 40% 2-month (favor shorter horizon)
                        df['future_return_smooth'] = df.apply(
                            lambda row: row['future_return'] if pd.isna(row['future_return_2m'])
                            else 0.6 * row['future_return'] + 0.4 * row['future_return_2m'],
                            axis=1
                        )
                    else:
                        df['future_return_smooth'] = df['future_return']

                    # Compute sector-neutral residual return on SMOOTHED returns
                    if 'sector' in df.columns and df['sector'].notna().any():
                        sector_mean_return = df.groupby('sector')['future_return_smooth'].transform('mean')
                        df['future_return_resid'] = df['future_return_smooth'] - sector_mean_return
                    else:
                        df['future_return_resid'] = df['future_return_smooth'] - df['future_return_smooth'].mean()

                    # Rank RESIDUAL returns within this date (0 to 1)
                    df['future_return_resid_rank'] = df['future_return_resid'].rank(pct=True, na_option='keep')

                    # Also keep raw return rank for comparison
                    df['future_return_rank'] = df['future_return'].rank(pct=True, na_option='keep')

                    # === VOLATILITY-SCALED TARGET (ChatGPT: stabilizes distribution) ===
                    # target_scaled = residual_return / realized_vol
                    # This reduces heteroscedasticity and makes high-vol stocks comparable to low-vol
                    if 'volatility_20d' in df.columns:
                        # Clip volatility to avoid division by tiny numbers
                        vol_clipped = df['volatility_20d'].clip(lower=0.10)  # Min 10% annualized vol
                        df['target_vol_scaled'] = df['future_return_resid'] / vol_clipped
                        # Winsorize to ±3 standard deviations to reduce outlier impact
                        target_mean = df['target_vol_scaled'].mean()
                        target_std = df['target_vol_scaled'].std()
                        df['target_vol_scaled'] = df['target_vol_scaled'].clip(
                            lower=target_mean - 3*target_std,
                            upper=target_mean + 3*target_std
                        )
                    else:
                        df['target_vol_scaled'] = df['future_return_resid']

                    # === CONTINUOUS REGRESSION TARGET (ChatGPT: uses all information) ===
                    # Cross-sectional z-score of residual returns (mean=0, std=1 each month)
                    resid_mean = df['future_return_resid'].mean()
                    resid_std = df['future_return_resid'].std()
                    if resid_std > 0:
                        df['target_regression'] = (df['future_return_resid'] - resid_mean) / resid_std
                    else:
                        df['target_regression'] = 0.0

                    # Target: Top 10% of smoothed RESIDUAL returns = 1
                    df['target_binary'] = (df['future_return_resid_rank'] >= 0.9).astype(int)
                else:
                    df['future_return_resid'] = None
                    df['target_binary'] = None

                all_features.append(df)

        if all_features:
            return pd.concat(all_features, ignore_index=True)
        return pd.DataFrame()

    def _compute_spy_features(self, spy, rebalance_dates, forward_window):
        """Pre-compute SPY features for all dates."""
        spy = spy.set_index('date').sort_index()
        spy_features = {}

        for date in rebalance_dates:
            date = pd.to_datetime(date)
            spy_hist = spy[spy.index <= date]

            if len(spy_hist) < 63:
                continue

            current = spy_hist.iloc[-1]['adj_close']

            # Market volatility (20-day)
            returns = spy_hist.tail(20)['adj_close'].pct_change().dropna()
            market_vol = returns.std() * np.sqrt(252) if len(returns) > 1 else None

            # Market trend (3-month return)
            past = spy_hist.iloc[-63]['adj_close']
            market_trend = (current - past) / past if past > 0 else None

            # Future SPY return for target calculation
            spy_future = spy[spy.index > date].head(forward_window + 5)
            if len(spy_future) >= 21:
                spy_future_price = spy_future.iloc[20]['adj_close']
                spy_return = (spy_future_price - current) / current
            else:
                spy_return = None

            spy_features[date] = {
                'current': current,
                'market_volatility': market_vol,
                'market_trend': market_trend,
                'spy_return': spy_return
            }

        return spy_features

    def _compute_ticker_features_fast(self, ticker, date, ticker_df, spy_info, forward_window):
        """Compute features for one ticker on one date - optimized."""
        # Get historical data up to date
        hist = ticker_df[ticker_df.index <= date]

        if len(hist) < 252:  # Need 1 year
            return None

        current_price = hist.iloc[-1]['adj_close']
        if not current_price or current_price <= 0:
            return None

        # === MOMENTUM RETURNS (raw values - keep magnitude!) ===
        def get_return(days):
            if len(hist) > days:
                past = hist.iloc[-days]['adj_close']
                if past and past > 0:
                    return (current_price - past) / past
            return None

        # Short-term reversal signals (1-5 days) - VERY PREDICTIVE!
        return_1d = get_return(1)
        return_3d = get_return(3)
        return_5d = get_return(5)

        # Medium-term momentum
        return_1m = get_return(21)
        return_3m = get_return(63)
        return_6m = get_return(126)
        return_12m = get_return(252)

        # === VOLATILITY ===
        ret_20 = hist.tail(20)['adj_close'].pct_change().dropna()
        vol_20d = ret_20.std() * np.sqrt(252) if len(ret_20) > 1 else None

        ret_60 = hist.tail(60)['adj_close'].pct_change().dropna()
        vol_60d = ret_60.std() * np.sqrt(252) if len(ret_60) > 1 else None

        # === SMA DISTANCES ===
        sma_50 = hist.tail(50)['adj_close'].mean()
        sma_200 = hist.tail(200)['adj_close'].mean()
        dist_sma_50 = (current_price - sma_50) / sma_50 if sma_50 > 0 else None
        dist_sma_200 = (current_price - sma_200) / sma_200 if sma_200 > 0 else None

        # === 52-WEEK HIGH/LOW ===
        high_52w = hist.tail(252)['adj_close'].max()
        low_52w = hist.tail(252)['adj_close'].min()
        dist_52w_high = (current_price - high_52w) / high_52w if high_52w > 0 else None
        dist_52w_low = (current_price - low_52w) / low_52w if low_52w > 0 else None

        # === VOLUME SHOCK FEATURES (very strong signals!) ===
        if 'volume' in hist.columns:
            current_volume = hist.iloc[-1]['volume']
            avg_vol_20 = hist.tail(20)['volume'].mean()
            avg_vol_60 = hist.tail(60)['volume'].mean()

            # Volume ratio (current vs average)
            volume_ratio_20 = current_volume / avg_vol_20 if avg_vol_20 > 0 else None
            volume_ratio_60 = current_volume / avg_vol_60 if avg_vol_60 > 0 else None

            # Volume z-score (how unusual is today's volume)
            vol_std_20 = hist.tail(20)['volume'].std()
            volume_zscore = (current_volume - avg_vol_20) / vol_std_20 if vol_std_20 > 0 else None

            # 5-day average volume ratio
            avg_vol_5 = hist.tail(5)['volume'].mean()
            volume_5d_ratio = avg_vol_5 / avg_vol_20 if avg_vol_20 > 0 else None
        else:
            volume_ratio_20 = None
            volume_ratio_60 = None
            volume_zscore = None
            volume_5d_ratio = None

        # === ROLLING Z-SCORE FEATURES (ChatGPT recommendation) ===
        # Standardize features using 60-day rolling window to remove drift
        def rolling_zscore(series, window=60):
            """Compute z-score relative to rolling window."""
            if len(series) < window:
                return None
            recent = series.tail(window)
            mean = recent.mean()
            std = recent.std()
            if std > 0:
                return (series.iloc[-1] - mean) / std
            return None

        # Returns rolling z-scores
        ret_series = hist['adj_close'].pct_change()
        return_1m_zscore = rolling_zscore(ret_series.rolling(21).sum(), 60)
        return_3m_zscore = rolling_zscore(ret_series.rolling(63).sum(), 120)

        # Volatility rolling z-score
        vol_series = ret_series.rolling(20).std() * np.sqrt(252)
        vol_zscore_rolling = rolling_zscore(vol_series, 60)

        # Volume rolling z-score
        if 'volume' in hist.columns:
            vol_ma = hist['volume'].rolling(20).mean()
            volume_zscore_rolling = rolling_zscore(vol_ma, 60)
        else:
            volume_zscore_rolling = None

        # === NONLINEAR INTERACTION FEATURES (ChatGPT recommendation) ===
        # These capture conditional effects (e.g., momentum is stronger with high volume)

        # Momentum × Volume (high volume confirms momentum)
        mom_vol_interaction = None
        if return_1m is not None and volume_ratio_20 is not None:
            mom_vol_interaction = return_1m * np.log1p(volume_ratio_20)

        # Reversal × Volatility (reversal stronger in high vol)
        reversal_vol_interaction = None
        if return_5d is not None and vol_20d is not None:
            reversal_vol_interaction = -return_5d * vol_20d  # Negative = mean reversion

        # Distance from SMA × Volume (breakouts with volume)
        sma_vol_interaction = None
        if dist_sma_50 is not None and volume_ratio_20 is not None:
            sma_vol_interaction = dist_sma_50 * np.log1p(volume_ratio_20)

        # High/Low × Momentum (52-week high with momentum = continuation)
        high_mom_interaction = None
        if dist_52w_high is not None and return_3m is not None:
            high_mom_interaction = -dist_52w_high * return_3m  # Near high + positive mom

        # Volatility regime interaction (low vol + momentum = stronger signal)
        vol_regime_interaction = None
        if vol_20d is not None and return_1m is not None and spy_info.get('market_volatility'):
            relative_vol = vol_20d / spy_info['market_volatility'] if spy_info['market_volatility'] > 0 else 1
            vol_regime_interaction = return_1m / (relative_vol + 0.5)  # Dampens high-vol signals

        # === TARGET: Multi-horizon smoothed return (reduces noise) ===
        # Use average of 21-day and 42-day returns for smoother signal
        future = ticker_df[ticker_df.index > date].head(forward_window * 2 + 10)

        future_return = None
        future_return_2m = None
        excess = None

        # 1-month return (primary)
        if len(future) >= forward_window and spy_info.get('spy_return') is not None:
            future_price_1m = future.iloc[forward_window - 1]['adj_close']
            stock_return_1m = (future_price_1m - current_price) / current_price
            excess = stock_return_1m - spy_info['spy_return']
            future_return = stock_return_1m

        # 2-month return (secondary, for smoothing)
        if len(future) >= forward_window * 2:
            future_price_2m = future.iloc[forward_window * 2 - 1]['adj_close']
            future_return_2m = (future_price_2m - current_price) / current_price

        return {
            'ticker': ticker,
            'date': date,
            # Short-term reversal (1-5 days)
            'return_1d': return_1d,
            'return_3d': return_3d,
            'return_5d': return_5d,
            # Medium-term momentum
            'return_1m': return_1m,
            'return_3m': return_3m,
            'return_6m': return_6m,
            'return_12m': return_12m,
            # Volatility
            'volatility_20d': vol_20d,
            'volatility_60d': vol_60d,
            # Technical levels
            'dist_from_sma_50': dist_sma_50,
            'dist_from_sma_200': dist_sma_200,
            'dist_from_52w_high': dist_52w_high,
            'dist_from_52w_low': dist_52w_low,
            # Volume shocks
            'volume_ratio_20': volume_ratio_20,
            'volume_ratio_60': volume_ratio_60,
            'volume_zscore': volume_zscore,
            'volume_5d_ratio': volume_5d_ratio,
            # Rolling z-scores (ChatGPT: removes drift, stabilizes signal)
            'return_1m_zscore': return_1m_zscore,
            'return_3m_zscore': return_3m_zscore,
            'vol_zscore_rolling': vol_zscore_rolling,
            'volume_zscore_rolling': volume_zscore_rolling,
            # Nonlinear interactions (ChatGPT: captures conditional effects)
            'mom_vol_interaction': mom_vol_interaction,
            'reversal_vol_interaction': reversal_vol_interaction,
            'sma_vol_interaction': sma_vol_interaction,
            'high_mom_interaction': high_mom_interaction,
            'vol_regime_interaction': vol_regime_interaction,
            # Market regime
            'market_volatility': spy_info.get('market_volatility'),
            'market_trend': spy_info.get('market_trend'),
            # Target (raw values - will be ranked cross-sectionally)
            'target_excess': excess,
            'future_return': future_return,  # 1-month raw return
            'future_return_2m': future_return_2m  # 2-month raw return for smoothing
        }

    def _get_monthly_dates(self, prices: pd.DataFrame) -> list:
        """Get last trading day of each month."""
        prices['date_dt'] = pd.to_datetime(prices['date'])
        prices['year_month'] = prices['date_dt'].dt.to_period('M')

        # Get last date of each month
        monthly = prices.groupby('year_month')['date'].max().values
        return sorted(monthly)

    def _compute_features_for_date(
        self,
        date: str,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        spy: pd.DataFrame,
        forward_window: int = 21
    ) -> pd.DataFrame:
        """
        Compute features for all stocks on a specific date.

        Args:
            date: Rebalancing date YYYY-MM-DD
            prices: All price data
            fundamentals: All fundamental data
            spy: SPY benchmark data
            forward_window: Days to look ahead for target

        Returns:
            DataFrame with features for all stocks on this date
        """
        features_list = []

        # Get list of stocks trading on this date
        stocks_on_date = prices[prices['date'] == date]['ticker'].unique()

        for ticker in stocks_on_date:
            try:
                ticker_features = self._compute_stock_features(
                    ticker=ticker,
                    date=date,
                    prices=prices,
                    fundamentals=fundamentals,
                    spy=spy,
                    forward_window=forward_window
                )

                if ticker_features is not None:
                    features_list.append(ticker_features)

            except Exception as e:
                # Skip stocks with errors
                continue

        if features_list:
            df = pd.DataFrame(features_list)

            # === ADD CROSS-SECTIONAL RANKING FEATURES ===
            # These rank each stock relative to all other stocks on the SAME date
            # This is critical - what matters is relative performance, not absolute!

            rank_cols = [
                'return_1m', 'return_3m', 'return_6m', 'return_12m',
                'volatility_20d', 'volatility_60d',
                'dist_from_sma_50', 'dist_from_sma_200',
                'dist_from_52w_high', 'dist_from_52w_low',
                'volume_ratio'
            ]

            for col in rank_cols:
                if col in df.columns:
                    # Compute percentile rank (0 to 1) within this date
                    df[f'{col}_rank'] = df[col].rank(pct=True, na_option='keep')

            return df
        else:
            return pd.DataFrame()

    def _compute_stock_features(
        self,
        ticker: str,
        date: str,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        spy: pd.DataFrame,
        forward_window: int
    ) -> Optional[dict]:
        """
        Compute all features for a single stock on a single date.

        Returns:
            Dictionary of features, or None if insufficient data
        """
        # Get historical prices for this stock up to date
        ticker_prices = prices[
            (prices['ticker'] == ticker) & (prices['date'] <= date)
        ].sort_values('date')

        if len(ticker_prices) < 252:  # Need at least 1 year of data
            return None

        # Current price
        current_price = ticker_prices.iloc[-1]['adj_close']

        if current_price is None or current_price <= 0:
            return None

        # === MOMENTUM FEATURES ===

        # Returns over various periods
        def get_return(days_back):
            if len(ticker_prices) > days_back:
                past_price = ticker_prices.iloc[-days_back]['adj_close']
                if past_price and past_price > 0:
                    return (current_price - past_price) / past_price
            return None

        return_1m = get_return(21)   # ~1 month
        return_3m = get_return(63)   # ~3 months
        return_6m = get_return(126)  # ~6 months
        return_12m = get_return(252) # ~12 months

        # Volatility (20-day and 60-day)
        returns_20d = ticker_prices.tail(20)['adj_close'].pct_change().dropna()
        volatility_20d = returns_20d.std() * np.sqrt(252) if len(returns_20d) > 1 else None

        returns_60d = ticker_prices.tail(60)['adj_close'].pct_change().dropna()
        volatility_60d = returns_60d.std() * np.sqrt(252) if len(returns_60d) > 1 else None

        # Distance from moving averages
        sma_50 = ticker_prices.tail(50)['adj_close'].mean() if len(ticker_prices) >= 50 else None
        sma_200 = ticker_prices.tail(200)['adj_close'].mean() if len(ticker_prices) >= 200 else None

        dist_from_sma_50 = (current_price - sma_50) / sma_50 if sma_50 and sma_50 > 0 else None
        dist_from_sma_200 = (current_price - sma_200) / sma_200 if sma_200 and sma_200 > 0 else None

        # === 52-WEEK HIGH/LOW FEATURES (very predictive!) ===
        if len(ticker_prices) >= 252:
            high_52w = ticker_prices.tail(252)['adj_close'].max()
            low_52w = ticker_prices.tail(252)['adj_close'].min()

            # Distance from 52-week high (negative = below high)
            dist_from_52w_high = (current_price - high_52w) / high_52w if high_52w > 0 else None

            # Distance from 52-week low (positive = above low)
            dist_from_52w_low = (current_price - low_52w) / low_52w if low_52w > 0 else None
        else:
            dist_from_52w_high = None
            dist_from_52w_low = None

        # === VOLUME FEATURES ===
        if 'volume' in ticker_prices.columns and len(ticker_prices) >= 20:
            avg_volume_20d = ticker_prices.tail(20)['volume'].mean()
            current_volume = ticker_prices.iloc[-1]['volume']
            volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else None
        else:
            volume_ratio = None

        # === VALUE & QUALITY FEATURES (from fundamentals) ===

        # Get most recent fundamentals before this date
        ticker_funds = fundamentals[
            (fundamentals['ticker'] == ticker) & (fundamentals['date'] <= date)
        ].sort_values('date')

        if not ticker_funds.empty:
            latest_fund = ticker_funds.iloc[-1]

            # Value metrics
            pe_ratio = latest_fund['pe_ratio']
            earnings_yield = 1.0 / pe_ratio if pe_ratio and pe_ratio > 0 else None

            pb_ratio = latest_fund['pb_ratio']
            book_to_price = 1.0 / pb_ratio if pb_ratio and pb_ratio > 0 else None

            # Quality metrics
            roe = latest_fund['roe']
            profit_margin = latest_fund['profit_margin']
            debt_to_equity = latest_fund['debt_to_equity']
        else:
            earnings_yield = None
            book_to_price = None
            roe = None
            profit_margin = None
            debt_to_equity = None

        # === MARKET REGIME FEATURES ===

        # Get SPY data up to this date
        spy_data = spy[spy['date'] <= date].sort_values('date')

        if len(spy_data) >= 20:
            spy_current = spy_data.iloc[-1]['adj_close']

            # Market volatility (20-day realized vol of SPY)
            spy_returns = spy_data.tail(20)['adj_close'].pct_change().dropna()
            market_volatility = spy_returns.std() * np.sqrt(252) if len(spy_returns) > 1 else None

            # Market trend (SPY 3-month return)
            if len(spy_data) > 63:
                spy_past = spy_data.iloc[-63]['adj_close']
                market_trend = (spy_current - spy_past) / spy_past if spy_past > 0 else None
            else:
                market_trend = None
        else:
            market_volatility = None
            market_trend = None

        # === SECTOR RELATIVE (placeholder - would need sector data) ===
        sector_return = None  # TODO: Compute sector average returns

        # === TARGET LABELS ===

        # Get future prices for this stock (forward_window days ahead)
        future_prices = prices[
            (prices['ticker'] == ticker) & (prices['date'] > date)
        ].sort_values('date').head(forward_window + 5)  # Extra buffer

        # Get future SPY prices
        future_spy = spy[spy['date'] > date].sort_values('date').head(forward_window + 5)

        if len(future_prices) >= forward_window and len(future_spy) >= forward_window:
            # Stock return over forward window
            future_price = future_prices.iloc[forward_window - 1]['adj_close']
            stock_return = (future_price - current_price) / current_price if future_price > 0 else None

            # SPY return over same period
            future_spy_date = future_prices.iloc[forward_window - 1]['date']
            spy_future = future_spy[future_spy['date'] <= future_spy_date]

            if not spy_future.empty:
                spy_future_price = spy_future.iloc[-1]['adj_close']
                spy_current_price = spy_data.iloc[-1]['adj_close']
                spy_return = (spy_future_price - spy_current_price) / spy_current_price if spy_current_price > 0 else None

                # Excess return (stock vs benchmark)
                if stock_return is not None and spy_return is not None:
                    target_1m_excess = stock_return - spy_return
                    target_binary = 1 if target_1m_excess > 0 else 0
                else:
                    target_1m_excess = None
                    target_binary = None
            else:
                target_1m_excess = None
                target_binary = None
        else:
            target_1m_excess = None
            target_binary = None

        # === ASSEMBLE FEATURE DICTIONARY ===

        features = {
            'ticker': ticker,
            'date': date,

            # Momentum
            'return_1m': return_1m,
            'return_3m': return_3m,
            'return_6m': return_6m,
            'return_12m': return_12m,
            'volatility_20d': volatility_20d,
            'volatility_60d': volatility_60d,
            'dist_from_sma_50': dist_from_sma_50,
            'dist_from_sma_200': dist_from_sma_200,

            # Price level (52-week high/low)
            'dist_from_52w_high': dist_from_52w_high,
            'dist_from_52w_low': dist_from_52w_low,

            # Volume
            'volume_ratio': volume_ratio,

            # Value
            'earnings_yield': earnings_yield,
            'book_to_price': book_to_price,

            # Quality
            'roe': roe,
            'debt_to_equity': debt_to_equity,
            'profit_margin': profit_margin,

            # Market regime
            'market_volatility': market_volatility,
            'market_trend': market_trend,

            # Sector relative
            'sector_return': sector_return,

            # Targets
            'target_1m_excess': target_1m_excess,
            'target_binary': target_binary
        }

        return features
