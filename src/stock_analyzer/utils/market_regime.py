"""
Market Regime Detection

Detects current market conditions to adapt strategy weights:
- Bull Market: High momentum, risk-on
- Bear Market: Defensive, quality, value
- High Volatility: Low volatility factor, quality
- Sideways: Mean reversion, range-bound strategies

Uses multiple indicators:
- VIX (volatility)
- S&P 500 trend (SMA 50/200)
- Market breadth
"""

from enum import Enum
from decimal import Decimal
from typing import Dict, Optional
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

from .logger import setup_logger

logger = setup_logger(__name__)


class MarketRegime(str, Enum):
    """Market regime types."""

    BULL = "bull"              # Strong uptrend, low vol
    BEAR = "bear"              # Downtrend
    HIGH_VOLATILITY = "high_volatility"  # Uncertain, choppy
    SIDEWAYS = "sideways"      # Range-bound
    RISK_ON = "risk_on"        # Low VIX, strong momentum
    RISK_OFF = "risk_off"      # High VIX, defensive


class MarketRegimeDetector:
    """
    Detect current market regime using multiple indicators.

    Methodology:
    1. VIX levels (fear gauge)
    2. S&P 500 trend (SMA crossovers)
    3. Market breadth (advance/decline)
    4. Volatility regime
    """

    def __init__(self):
        self._cache = {}
        self._cache_expiry = None

    def detect_regime(self) -> MarketRegime:
        """
        Detect current market regime.

        Returns:
            MarketRegime enum value
        """
        # Check cache (5 minute TTL)
        if self._cache_expiry and datetime.now() < self._cache_expiry:
            return self._cache.get("regime", MarketRegime.SIDEWAYS)

        try:
            vix = self._get_vix()
            spy_trend = self._get_spy_trend()
            volatility_20d = self._get_spy_volatility()

            logger.info(f"Market indicators - VIX: {vix:.2f}, SPY Trend: {spy_trend:.2f}%, Vol(20d): {volatility_20d:.2f}%")

            # Determine regime
            regime = self._classify_regime(vix, spy_trend, volatility_20d)

            # Cache result
            self._cache["regime"] = regime
            self._cache_expiry = datetime.now() + timedelta(minutes=5)

            logger.info(f"Market regime detected: {regime.value}")
            return regime

        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime.SIDEWAYS  # Default to neutral

    def get_strategy_weights(self, regime: Optional[MarketRegime] = None) -> Dict[str, float]:
        """
        Get optimal strategy weights for current regime.

        Args:
            regime: Market regime (auto-detect if None)

        Returns:
            Dictionary of strategy weights
        """
        if regime is None:
            regime = self.detect_regime()

        # Regime-specific weights
        weights = {
            MarketRegime.BULL: {
                "ml_prediction": 0.25,
                "momentum": 0.25,
                "growth": 0.20,
                "news_sentiment": 0.15,
                "value": 0.08,
                "fama_french": 0.07
            },
            MarketRegime.BEAR: {
                "ml_prediction": 0.25,
                "value": 0.25,
                "quality": 0.20,
                "fama_french": 0.15,
                "news_sentiment": 0.10,
                "momentum": 0.05
            },
            MarketRegime.HIGH_VOLATILITY: {
                "ml_prediction": 0.30,
                "quality": 0.25,
                "low_volatility": 0.20,
                "value": 0.15,
                "news_sentiment": 0.10
            },
            MarketRegime.SIDEWAYS: {
                "ml_prediction": 0.25,
                "value": 0.20,
                "momentum": 0.18,
                "growth": 0.15,
                "news_sentiment": 0.12,
                "fama_french": 0.10
            },
            MarketRegime.RISK_ON: {
                "ml_prediction": 0.25,
                "momentum": 0.35,
                "growth": 0.20,
                "news_sentiment": 0.12,
                "value": 0.08
            },
            MarketRegime.RISK_OFF: {
                "ml_prediction": 0.25,
                "quality": 0.35,
                "value": 0.18,
                "low_volatility": 0.15,
                "news_sentiment": 0.07
            }
        }

        return weights.get(regime, weights[MarketRegime.SIDEWAYS])

    def _get_vix(self) -> float:
        """Get current VIX (volatility index)."""
        try:
            vix = yf.Ticker("^VIX")
            data = vix.history(period="1d")

            if data.empty:
                logger.warning("No VIX data available, using default")
                return 20.0  # Historical average

            return float(data['Close'].iloc[-1])

        except Exception as e:
            logger.warning(f"Error fetching VIX: {e}")
            return 20.0

    def _get_spy_trend(self) -> float:
        """
        Get S&P 500 trend strength (SMA 50 vs SMA 200).

        Returns:
            Percentage above/below 200-day SMA
        """
        try:
            spy = yf.Ticker("SPY")
            data = spy.history(period="1y")

            if len(data) < 200:
                logger.warning("Insufficient SPY data")
                return 0.0

            current_price = float(data['Close'].iloc[-1])
            sma_50 = float(data['Close'].rolling(50).mean().iloc[-1])
            sma_200 = float(data['Close'].rolling(200).mean().iloc[-1])

            # Calculate trend strength
            trend_pct = ((current_price / sma_200) - 1) * 100

            logger.debug(f"SPY: ${current_price:.2f}, SMA50: ${sma_50:.2f}, SMA200: ${sma_200:.2f}")

            return trend_pct

        except Exception as e:
            logger.warning(f"Error fetching SPY trend: {e}")
            return 0.0

    def _get_spy_volatility(self) -> float:
        """
        Get S&P 500 realized volatility (20-day).

        Returns:
            Annualized volatility percentage
        """
        try:
            spy = yf.Ticker("SPY")
            data = spy.history(period="3mo")

            if len(data) < 20:
                return 15.0  # Historical average

            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()

            # Annualized volatility (20-day)
            vol_20d = returns.tail(20).std() * (252 ** 0.5) * 100

            return float(vol_20d)

        except Exception as e:
            logger.warning(f"Error calculating volatility: {e}")
            return 15.0

    def _classify_regime(self, vix: float, spy_trend: float, volatility: float) -> MarketRegime:
        """
        Classify market regime based on indicators.

        Args:
            vix: Current VIX level
            spy_trend: % above/below 200-day SMA
            volatility: 20-day realized volatility

        Returns:
            MarketRegime classification
        """
        # High Volatility Regime (VIX > 30 or realized vol > 25%)
        if vix > 30 or volatility > 25:
            return MarketRegime.HIGH_VOLATILITY

        # Bear Market (downtrend + elevated VIX)
        if spy_trend < -5 and vix > 20:
            return MarketRegime.BEAR

        # Bull Market (strong uptrend + low VIX)
        if spy_trend > 5 and vix < 15:
            return MarketRegime.BULL

        # Risk-On (low VIX, positive trend)
        if vix < 15 and spy_trend > 0:
            return MarketRegime.RISK_ON

        # Risk-Off (high VIX, any trend)
        if vix > 25:
            return MarketRegime.RISK_OFF

        # Sideways Market (range-bound)
        if abs(spy_trend) < 5:
            return MarketRegime.SIDEWAYS

        # Default to sideways
        return MarketRegime.SIDEWAYS


# Singleton instance
_regime_detector = None


def get_market_regime() -> MarketRegime:
    """Get current market regime (singleton)."""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = MarketRegimeDetector()
    return _regime_detector.detect_regime()


def get_regime_weights(regime: Optional[MarketRegime] = None) -> Dict[str, float]:
    """
    Get strategy weights for specified or current regime.

    Args:
        regime: Market regime (if None, detects current regime)

    Returns:
        Dictionary of strategy weights
    """
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = MarketRegimeDetector()
    return _regime_detector.get_strategy_weights(regime)
