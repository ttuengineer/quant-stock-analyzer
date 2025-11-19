"""
Momentum-based scoring strategy.

Identifies stocks with strong price momentum and positive technical signals.
Based on institutional momentum factors used by quantitative hedge funds.
"""

from typing import Optional
from decimal import Decimal
import pandas as pd
import numpy as np

from .base import ScoringStrategy
from ..models.domain import Fundamentals, TechnicalIndicators, StockInfo
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class MomentumStrategy(ScoringStrategy):
    """
    Advanced momentum strategy using multiple timeframes.

    Scoring factors:
    - Price momentum (1M, 3M, 6M, 12M)
    - Relative strength vs market
    - Trend strength (ADX)
    - Moving average alignment
    - Volume confirmation
    - Momentum indicators (RSI, MACD)
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(name="Momentum Strategy", weight=weight)

    def calculate_score(
        self,
        price_data: pd.DataFrame,
        fundamentals: Optional[Fundamentals] = None,
        technical_indicators: Optional[TechnicalIndicators] = None,
        stock_info: Optional[StockInfo] = None,
    ) -> Decimal:
        """Calculate momentum score (0-100)."""
        if price_data.empty or len(price_data) < 60:
            logger.warning(f"{self.name}: Insufficient data")
            return Decimal("0")

        total_score = 0.0
        max_score = 0.0

        try:
            current_price = float(price_data['Close'].iloc[-1])

            # 1. Multi-timeframe momentum (35 points max)
            momentum_score, momentum_max = self._calculate_momentum(price_data, current_price)
            total_score += momentum_score
            max_score += momentum_max

            # 2. Trend quality (25 points max)
            if technical_indicators:
                trend_score, trend_max = self._calculate_trend_quality(
                    price_data, technical_indicators
                )
                total_score += trend_score
                max_score += trend_max

            # 3. Volume confirmation (20 points max)
            volume_score, volume_max = self._calculate_volume_signals(price_data)
            total_score += volume_score
            max_score += volume_max

            # 4. Technical momentum indicators (20 points max)
            if technical_indicators:
                tech_score, tech_max = self._calculate_technical_momentum(
                    technical_indicators
                )
                total_score += tech_score
                max_score += tech_max

            # Normalize to 0-100
            final_score = self.normalize_score(total_score, max_score)
            logger.debug(f"{self.name}: Score = {final_score}")
            return final_score

        except Exception as e:
            logger.error(f"{self.name}: Error calculating score: {e}")
            return Decimal("0")

    def _calculate_momentum(
        self,
        price_data: pd.DataFrame,
        current_price: float
    ) -> tuple[float, float]:
        """
        Calculate multi-timeframe price momentum.

        Returns:
            (score, max_score) tuple
        """
        score = 0.0
        max_score = 35.0

        try:
            # 1-month momentum (10 points)
            if len(price_data) >= 22:
                price_1m = float(price_data['Close'].iloc[-22])
                return_1m = (current_price / price_1m - 1) * 100

                if return_1m > 10:  # Strong positive
                    score += 10
                elif return_1m > 5:  # Moderate positive
                    score += 7
                elif return_1m > 0:  # Weak positive
                    score += 4
                elif return_1m > -5:  # Slight negative
                    score += 2

            # 3-month momentum (10 points)
            if len(price_data) >= 66:
                price_3m = float(price_data['Close'].iloc[-66])
                return_3m = (current_price / price_3m - 1) * 100

                if return_3m > 20:
                    score += 10
                elif return_3m > 10:
                    score += 7
                elif return_3m > 5:
                    score += 4
                elif return_3m > 0:
                    score += 2

            # 6-month momentum (8 points)
            if len(price_data) >= 126:
                price_6m = float(price_data['Close'].iloc[-126])
                return_6m = (current_price / price_6m - 1) * 100

                if return_6m > 30:
                    score += 8
                elif return_6m > 15:
                    score += 6
                elif return_6m > 5:
                    score += 3

            # 12-month momentum (7 points)
            if len(price_data) >= 252:
                price_12m = float(price_data['Close'].iloc[-252])
                return_12m = (current_price / price_12m - 1) * 100

                if return_12m > 50:
                    score += 7
                elif return_12m > 25:
                    score += 5
                elif return_12m > 10:
                    score += 3

        except Exception as e:
            logger.warning(f"Error calculating momentum: {e}")

        return score, max_score

    def _calculate_trend_quality(
        self,
        price_data: pd.DataFrame,
        technical_indicators: TechnicalIndicators
    ) -> tuple[float, float]:
        """
        Calculate trend strength and quality.

        Returns:
            (score, max_score) tuple
        """
        score = 0.0
        max_score = 25.0

        try:
            current_price = float(price_data['Close'].iloc[-1])

            # Moving average alignment (15 points)
            # Price > SMA20 > SMA50 > SMA200 is bullish
            ma_score = 0
            if technical_indicators.sma_20 and current_price > float(technical_indicators.sma_20):
                ma_score += 5
            if technical_indicators.sma_50 and current_price > float(technical_indicators.sma_50):
                ma_score += 5
            if technical_indicators.sma_200 and current_price > float(technical_indicators.sma_200):
                ma_score += 3

            # MA alignment
            if (technical_indicators.sma_20 and technical_indicators.sma_50 and
                    float(technical_indicators.sma_20) > float(technical_indicators.sma_50)):
                ma_score += 2

            score += ma_score

            # ADX trend strength (10 points)
            if technical_indicators.adx:
                adx = float(technical_indicators.adx)
                if adx > 40:  # Very strong trend
                    score += 10
                elif adx > 30:  # Strong trend
                    score += 7
                elif adx > 25:  # Moderate trend
                    score += 4
                elif adx > 20:  # Weak trend
                    score += 2

        except Exception as e:
            logger.warning(f"Error calculating trend quality: {e}")

        return score, max_score

    def _calculate_volume_signals(
        self,
        price_data: pd.DataFrame
    ) -> tuple[float, float]:
        """
        Calculate volume-based momentum signals.

        Returns:
            (score, max_score) tuple
        """
        score = 0.0
        max_score = 20.0

        try:
            if 'Volume' not in price_data.columns:
                return score, max_score

            current_volume = price_data['Volume'].iloc[-1]
            avg_volume_20 = price_data['Volume'].rolling(20).mean().iloc[-1]

            if avg_volume_20 > 0:
                volume_ratio = current_volume / avg_volume_20

                # Volume surge on up days (10 points)
                price_change = price_data['Close'].pct_change().iloc[-1]
                if price_change > 0 and volume_ratio > 1.5:
                    score += 10
                elif price_change > 0 and volume_ratio > 1.2:
                    score += 6
                elif price_change > 0:
                    score += 3

                # Consistent volume (10 points)
                recent_volumes = price_data['Volume'].iloc[-10:]
                volume_stability = 1 - (recent_volumes.std() / recent_volumes.mean())
                if volume_stability > 0.7:
                    score += 10
                elif volume_stability > 0.5:
                    score += 6

        except Exception as e:
            logger.warning(f"Error calculating volume signals: {e}")

        return score, max_score

    def _calculate_technical_momentum(
        self,
        technical_indicators: TechnicalIndicators
    ) -> tuple[float, float]:
        """
        Calculate technical momentum indicator scores.

        Returns:
            (score, max_score) tuple
        """
        score = 0.0
        max_score = 20.0

        try:
            # MACD bullish signal (10 points)
            if (technical_indicators.macd and technical_indicators.macd_signal and
                    technical_indicators.macd_histogram):
                macd = float(technical_indicators.macd)
                signal = float(technical_indicators.macd_signal)
                histogram = float(technical_indicators.macd_histogram)

                # MACD above signal and positive
                if macd > signal and macd > 0:
                    score += 10
                elif macd > signal:
                    score += 6

                # Histogram increasing
                if histogram > 0:
                    score += 0  # Already counted above

            # RSI in bullish zone (10 points)
            if technical_indicators.rsi:
                rsi = float(technical_indicators.rsi)

                # RSI 50-70 is bullish momentum
                if 55 <= rsi <= 70:
                    score += 10
                elif 50 <= rsi < 55:
                    score += 7
                elif 45 <= rsi < 50:
                    score += 4
                elif 70 < rsi <= 80:  # Overbought but still bullish
                    score += 5
                elif rsi < 30:  # Oversold - potential reversal
                    score += 3

        except Exception as e:
            logger.warning(f"Error calculating technical momentum: {e}")

        return score, max_score
