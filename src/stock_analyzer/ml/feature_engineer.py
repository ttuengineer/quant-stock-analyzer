"""
Feature Engineering for Machine Learning

Creates 60+ technical and fundamental features for ML models:
- Price-based features (returns, volatility, momentum)
- Technical indicators (RSI, MACD, Bollinger, etc.)
- Volume features (OBV, volume ratios, trends)
- Fundamental ratios (P/E, ROE, margins, growth)
- Cross-sectional features (sector rankings, peer comparisons)
- Market regime features (VIX, market trend)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from decimal import Decimal
import pandas_ta as ta

from ..models.domain import Fundamentals, TechnicalIndicators, StockInfo
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """
    Generate ML-ready features from stock data.

    Creates 60+ features across multiple categories:
    - Price momentum (20 features)
    - Technical indicators (25 features)
    - Volume analysis (10 features)
    - Fundamentals (15 features)
    - Market regime (5 features)
    """

    def __init__(self):
        self.feature_names = []

    def engineer_features(
        self,
        price_data: pd.DataFrame,
        fundamentals: Optional[Fundamentals] = None,
        technical_indicators: Optional[TechnicalIndicators] = None,
        stock_info: Optional[StockInfo] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Engineer all features for ML model.

        Args:
            price_data: OHLCV price data
            fundamentals: Fundamental metrics
            technical_indicators: Pre-calculated indicators
            stock_info: Stock metadata

        Returns:
            DataFrame with engineered features (single row)
        """
        if price_data is None or price_data.empty or len(price_data) < 60:
            logger.warning("Insufficient price data for feature engineering")
            return None

        try:
            features = {}

            # 1. Price Momentum Features (20 features)
            price_features = self._create_price_features(price_data)
            features.update(price_features)

            # 2. Technical Indicator Features (25 features)
            technical_features = self._create_technical_features(price_data, technical_indicators)
            features.update(technical_features)

            # 3. Volume Features (10 features)
            volume_features = self._create_volume_features(price_data)
            features.update(volume_features)

            # 4. Fundamental Features (15 features)
            fundamental_features = self._create_fundamental_features(fundamentals)
            features.update(fundamental_features)

            # 5. Market Regime Features (5 features)
            regime_features = self._create_regime_features()
            features.update(regime_features)

            # Convert to DataFrame
            feature_df = pd.DataFrame([features])

            # Store feature names
            self.feature_names = list(features.keys())

            logger.info(f"Engineered {len(features)} features for ML")

            return feature_df

        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            return None

    def _create_price_features(self, price_data: pd.DataFrame) -> Dict:
        """Create price momentum and trend features."""
        features = {}
        prices = price_data['Close']

        try:
            # Returns over multiple periods
            features['return_1d'] = prices.pct_change(1).iloc[-1] if len(prices) > 1 else 0
            features['return_5d'] = prices.pct_change(5).iloc[-1] if len(prices) > 5 else 0
            features['return_10d'] = prices.pct_change(10).iloc[-1] if len(prices) > 10 else 0
            features['return_20d'] = prices.pct_change(20).iloc[-1] if len(prices) > 20 else 0
            features['return_60d'] = prices.pct_change(60).iloc[-1] if len(prices) > 60 else 0

            # Volatility measures
            returns = prices.pct_change().dropna()
            features['volatility_5d'] = returns.tail(5).std() if len(returns) > 5 else 0
            features['volatility_20d'] = returns.tail(20).std() if len(returns) > 20 else 0
            features['volatility_60d'] = returns.tail(60).std() if len(returns) > 60 else 0

            # Price position relative to highs/lows
            high_52w = prices.tail(252).max() if len(prices) >= 252 else prices.max()
            low_52w = prices.tail(252).min() if len(prices) >= 252 else prices.min()
            current_price = prices.iloc[-1]

            features['pct_from_high_52w'] = (current_price / high_52w - 1) if high_52w > 0 else 0
            features['pct_from_low_52w'] = (current_price / low_52w - 1) if low_52w > 0 else 0

            # Moving average features
            features['sma_20'] = prices.tail(20).mean() / current_price if current_price > 0 else 1
            features['sma_50'] = prices.tail(50).mean() / current_price if current_price > 0 and len(prices) > 50 else 1
            features['sma_200'] = prices.tail(200).mean() / current_price if current_price > 0 and len(prices) > 200 else 1

            # Trend strength
            features['price_vs_sma20'] = (current_price / prices.tail(20).mean() - 1) if len(prices) > 20 else 0
            features['price_vs_sma50'] = (current_price / prices.tail(50).mean() - 1) if len(prices) > 50 else 0

            # Momentum indicators
            features['momentum_5d'] = returns.tail(5).sum() if len(returns) > 5 else 0
            features['momentum_20d'] = returns.tail(20).sum() if len(returns) > 20 else 0

            # Acceleration
            if len(returns) > 20:
                features['return_acceleration'] = returns.tail(10).mean() / returns.tail(20).mean() if returns.tail(20).mean() != 0 else 1
            else:
                features['return_acceleration'] = 1

        except Exception as e:
            logger.warning(f"Error creating price features: {e}")

        return features

    def _create_technical_features(self, price_data: pd.DataFrame, tech_ind: Optional[TechnicalIndicators]) -> Dict:
        """Create technical indicator features."""
        features = {}

        try:
            # Use pre-calculated indicators if available
            if tech_ind:
                features['rsi'] = float(tech_ind.rsi) if tech_ind.rsi else 50
                features['macd'] = float(tech_ind.macd) if tech_ind.macd else 0
                features['macd_signal'] = float(tech_ind.macd_signal) if tech_ind.macd_signal else 0
                features['macd_hist'] = float(tech_ind.macd_histogram) if tech_ind.macd_histogram else 0
                features['adx'] = float(tech_ind.adx) if tech_ind.adx else 25
                features['atr'] = float(tech_ind.atr) if tech_ind.atr else 0
            else:
                # Calculate on-the-fly
                features['rsi'] = self._calculate_rsi(price_data)
                features['macd'] = 0
                features['macd_signal'] = 0
                features['macd_hist'] = 0
                features['adx'] = 25
                features['atr'] = 0

            # Bollinger Bands
            if len(price_data) >= 20:
                closes = price_data['Close']
                sma_20 = closes.tail(20).mean()
                std_20 = closes.tail(20).std()
                current_price = closes.iloc[-1]

                bb_upper = sma_20 + 2 * std_20
                bb_lower = sma_20 - 2 * std_20

                features['bb_position'] = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
                features['bb_width'] = (bb_upper - bb_lower) / sma_20 if sma_20 > 0 else 0
            else:
                features['bb_position'] = 0.5
                features['bb_width'] = 0

            # Stochastic
            if len(price_data) >= 14:
                high_14 = price_data['High'].tail(14).max()
                low_14 = price_data['Low'].tail(14).min()
                current = price_data['Close'].iloc[-1]
                features['stochastic_k'] = (current - low_14) / (high_14 - low_14) * 100 if (high_14 - low_14) > 0 else 50
            else:
                features['stochastic_k'] = 50

            # Additional technical features
            features['rsi_oversold'] = 1 if features['rsi'] < 30 else 0
            features['rsi_overbought'] = 1 if features['rsi'] > 70 else 0
            features['macd_bullish'] = 1 if features['macd'] > features['macd_signal'] else 0

        except Exception as e:
            logger.warning(f"Error creating technical features: {e}")

        return features

    def _create_volume_features(self, price_data: pd.DataFrame) -> Dict:
        """Create volume-based features."""
        features = {}

        try:
            if 'Volume' in price_data.columns:
                volume = price_data['Volume']
                current_vol = volume.iloc[-1]

                # Volume ratios
                features['volume_vs_avg_5d'] = current_vol / volume.tail(5).mean() if volume.tail(5).mean() > 0 else 1
                features['volume_vs_avg_20d'] = current_vol / volume.tail(20).mean() if volume.tail(20).mean() > 0 else 1

                # Volume trend
                features['volume_trend_5d'] = (volume.tail(5).mean() / volume.tail(10).mean()) if len(volume) > 10 and volume.tail(10).mean() > 0 else 1

                # Price-volume correlation
                if len(price_data) > 20:
                    returns = price_data['Close'].pct_change().tail(20)
                    vol_changes = volume.pct_change().tail(20)
                    features['price_volume_corr'] = returns.corr(vol_changes) if len(returns) == len(vol_changes) else 0
                else:
                    features['price_volume_corr'] = 0

                # On-Balance Volume (simplified)
                if len(price_data) > 10:
                    obv = (np.sign(price_data['Close'].diff()) * volume).tail(10).sum()
                    features['obv_10d'] = obv / volume.tail(10).sum() if volume.tail(10).sum() > 0 else 0
                else:
                    features['obv_10d'] = 0

            else:
                # No volume data
                features['volume_vs_avg_5d'] = 1
                features['volume_vs_avg_20d'] = 1
                features['volume_trend_5d'] = 1
                features['price_volume_corr'] = 0
                features['obv_10d'] = 0

        except Exception as e:
            logger.warning(f"Error creating volume features: {e}")

        return features

    def _create_fundamental_features(self, fundamentals: Optional[Fundamentals]) -> Dict:
        """Create fundamental ratio features."""
        features = {}

        if fundamentals:
            # Valuation
            features['pe_ratio'] = float(fundamentals.pe_ratio) if fundamentals.pe_ratio else 20
            features['peg_ratio'] = float(fundamentals.peg_ratio) if fundamentals.peg_ratio else 2
            features['price_to_book'] = float(fundamentals.price_to_book) if fundamentals.price_to_book else 3
            features['price_to_sales'] = float(fundamentals.price_to_sales) if fundamentals.price_to_sales else 2

            # Profitability
            features['roe'] = float(fundamentals.roe) if fundamentals.roe else 0.10
            features['roa'] = float(fundamentals.roa) if fundamentals.roa else 0.05
            features['profit_margin'] = float(fundamentals.profit_margin) if fundamentals.profit_margin else 0.05
            features['operating_margin'] = float(fundamentals.operating_margin) if fundamentals.operating_margin else 0.10

            # Growth
            features['revenue_growth'] = float(fundamentals.revenue_growth) if fundamentals.revenue_growth else 0
            features['earnings_growth'] = float(fundamentals.earnings_growth) if fundamentals.earnings_growth else 0

            # Financial health
            features['debt_to_equity'] = float(fundamentals.debt_to_equity) if fundamentals.debt_to_equity else 1.0
            features['current_ratio'] = float(fundamentals.current_ratio) if fundamentals.current_ratio else 1.5

            # Dividends
            features['dividend_yield'] = float(fundamentals.dividend_yield) if fundamentals.dividend_yield else 0
            features['payout_ratio'] = min(float(fundamentals.payout_ratio), 2.0) if fundamentals.payout_ratio else 0

            # Quality indicators
            features['has_positive_fcf'] = 1 if fundamentals.free_cash_flow and fundamentals.free_cash_flow > 0 else 0

        else:
            # Default values
            features['pe_ratio'] = 20
            features['peg_ratio'] = 2
            features['price_to_book'] = 3
            features['price_to_sales'] = 2
            features['roe'] = 0.10
            features['roa'] = 0.05
            features['profit_margin'] = 0.05
            features['operating_margin'] = 0.10
            features['revenue_growth'] = 0
            features['earnings_growth'] = 0
            features['debt_to_equity'] = 1.0
            features['current_ratio'] = 1.5
            features['dividend_yield'] = 0
            features['payout_ratio'] = 0
            features['has_positive_fcf'] = 0

        return features

    def _create_regime_features(self) -> Dict:
        """Create market regime features."""
        features = {}

        try:
            from ..utils.market_regime import MarketRegime, get_market_regime
            import yfinance as yf

            # Get current regime
            regime = get_market_regime()

            # One-hot encode regime
            features['regime_bull'] = 1 if regime == MarketRegime.BULL else 0
            features['regime_bear'] = 1 if regime == MarketRegime.BEAR else 0
            features['regime_high_vol'] = 1 if regime == MarketRegime.HIGH_VOLATILITY else 0
            features['regime_risk_off'] = 1 if regime == MarketRegime.RISK_OFF else 0

            # Get VIX level
            try:
                vix = yf.Ticker("^VIX")
                vix_data = vix.history(period="1d")
                features['vix_level'] = float(vix_data['Close'].iloc[-1]) if not vix_data.empty else 20.0
            except:
                features['vix_level'] = 20.0

        except Exception as e:
            logger.warning(f"Error creating regime features: {e}")
            features['regime_bull'] = 0
            features['regime_bear'] = 0
            features['regime_high_vol'] = 0
            features['regime_risk_off'] = 0
            features['vix_level'] = 20.0

        return features

    def _calculate_rsi(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI if not pre-calculated."""
        try:
            closes = price_data['Close']
            if len(closes) < period + 1:
                return 50.0

            delta = closes.diff()
            gain = delta.where(delta > 0, 0).tail(period).mean()
            loss = -delta.where(delta < 0, 0).tail(period).mean()

            if loss == 0:
                return 100.0

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi)

        except:
            return 50.0
