"""
Low Volatility Factor Strategy

The Low Volatility Anomaly: Lower risk stocks historically outperform
higher risk stocks on a risk-adjusted basis.

Focuses on:
- Low historical volatility
- Low beta (market sensitivity)
- Stable, predictable returns
- Defensive characteristics

Performs best in:
- Bear markets
- High volatility regimes
- Risk-off environments

References:
- Ang, Hodrick, Xing & Zhang (2006): "The Cross-Section of Volatility and Expected Returns"
- Baker, Bradley & Wurgler (2011): "Benchmarks as Limits to Arbitrage"
"""

from decimal import Decimal
from typing import Optional
import pandas as pd
import numpy as np

from .base import ScoringStrategy
from ..models.domain import Fundamentals, TechnicalIndicators, StockInfo
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class LowVolatilityStrategy(ScoringStrategy):
    """
    Low volatility factor scoring strategy.

    Rewards stocks with:
    - Low historical volatility
    - Low beta (< 1.0)
    - Stable price movements
    - Consistent performance
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(name="Low Volatility Factor", weight=weight)

    def calculate_score(
        self,
        price_data: pd.DataFrame,
        fundamentals: Optional[Fundamentals] = None,
        technical_indicators: Optional[TechnicalIndicators] = None,
        stock_info: Optional[StockInfo] = None,
        **kwargs
    ) -> Decimal:
        """
        Calculate low volatility score.

        Returns:
            Score from 0-100 (higher = lower volatility = better)
        """
        if price_data is None or price_data.empty:
            logger.warning("Missing price data for volatility analysis")
            return Decimal("50.0")

        score = 0.0

        # Component 1: Historical Volatility (40 points)
        volatility_score = self._score_volatility(price_data)
        score += volatility_score * 0.40

        # Component 2: Beta (30 points)
        beta_score = self._score_beta(price_data)
        score += beta_score * 0.30

        # Component 3: Price Stability (20 points)
        stability_score = self._score_price_stability(price_data)
        score += stability_score * 0.20

        # Component 4: Drawdown (10 points)
        drawdown_score = self._score_drawdown(price_data)
        score += drawdown_score * 0.10

        logger.debug(
            f"Low-Vol components - Volatility: {volatility_score:.1f}, Beta: {beta_score:.1f}, "
            f"Stability: {stability_score:.1f}, Drawdown: {drawdown_score:.1f}"
        )

        return Decimal(str(min(100.0, max(0.0, score))))

    def _score_volatility(self, price_data: pd.DataFrame) -> float:
        """
        Score based on historical volatility.

        Lower volatility = higher score
        """
        try:
            if len(price_data) < 30:
                return 50.0

            # Calculate returns
            returns = price_data['Close'].pct_change().dropna()

            # Annualized volatility
            vol_annual = returns.std() * np.sqrt(252) * 100  # As percentage

            logger.debug(f"Annualized volatility: {vol_annual:.2f}%")

            # Score based on volatility levels
            if vol_annual < 15:  # Very low volatility
                return 100.0
            elif vol_annual < 20:  # Low volatility
                return 90.0
            elif vol_annual < 25:  # Below average
                return 75.0
            elif vol_annual < 30:  # Average
                return 60.0
            elif vol_annual < 40:  # Above average
                return 40.0
            elif vol_annual < 50:  # High volatility
                return 25.0
            else:  # Very high volatility
                return 10.0

        except Exception as e:
            logger.warning(f"Error calculating volatility score: {e}")
            return 50.0

    def _score_beta(self, price_data: pd.DataFrame) -> float:
        """
        Score based on beta (market sensitivity).

        Lower beta = higher score (less systematic risk)
        """
        # Calculate beta from price data
        beta = self._calculate_beta(price_data)

        if beta is None:
            return 50.0

        logger.debug(f"Beta: {beta:.2f}")

        # Score based on beta levels
        if beta < 0.5:  # Very defensive
            return 100.0
        elif beta < 0.7:  # Defensive
            return 95.0
        elif beta < 0.85:  # Low beta
            return 85.0
        elif beta < 1.0:  # Slightly defensive
            return 75.0
        elif beta < 1.15:  # Market beta
            return 55.0
        elif beta < 1.30:  # Slightly aggressive
            return 40.0
        elif beta < 1.50:  # Aggressive
            return 25.0
        else:  # Very aggressive (> 1.5)
            return 10.0

    def _calculate_beta(self, price_data: pd.DataFrame) -> Optional[float]:
        """
        Calculate beta vs S&P 500.

        Beta = Cov(Stock, Market) / Var(Market)
        """
        try:
            import yfinance as yf

            # Get S&P 500 data for same period
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="1y")

            if len(price_data) < 60 or len(spy_data) < 60:
                return None

            # Align dates
            stock_returns = price_data['Close'].pct_change().dropna()
            spy_returns = spy_data['Close'].pct_change().dropna()

            # Merge on dates
            merged = pd.DataFrame({
                'stock': stock_returns,
                'market': spy_returns
            }).dropna()

            if len(merged) < 60:
                return None

            # Calculate beta
            covariance = merged['stock'].cov(merged['market'])
            market_variance = merged['market'].var()

            beta = covariance / market_variance

            return float(beta)

        except Exception as e:
            logger.warning(f"Error calculating beta: {e}")
            return None

    def _score_price_stability(self, price_data: pd.DataFrame) -> float:
        """
        Score price stability (consistency of returns).

        Lower return variance = higher score
        """
        try:
            if len(price_data) < 30:
                return 50.0

            returns = price_data['Close'].pct_change().dropna()

            # Rolling 20-day volatility
            rolling_vol = returns.rolling(20).std()

            # Coefficient of variation of volatility (stability of volatility)
            vol_of_vol = rolling_vol.std() / rolling_vol.mean() if rolling_vol.mean() != 0 else 1.0

            # Lower vol-of-vol = more stable
            if vol_of_vol < 0.3:
                return 100.0
            elif vol_of_vol < 0.5:
                return 85.0
            elif vol_of_vol < 0.7:
                return 70.0
            elif vol_of_vol < 1.0:
                return 55.0
            elif vol_of_vol < 1.5:
                return 35.0
            else:
                return 20.0

        except Exception as e:
            logger.warning(f"Error calculating stability score: {e}")
            return 50.0

    def _score_drawdown(self, price_data: pd.DataFrame) -> float:
        """
        Score based on maximum drawdown.

        Smaller drawdowns = higher score
        """
        try:
            if len(price_data) < 30:
                return 50.0

            # Calculate running maximum
            prices = price_data['Close']
            running_max = prices.expanding().max()

            # Calculate drawdown
            drawdown = (prices / running_max - 1) * 100  # As percentage

            # Maximum drawdown
            max_dd = abs(drawdown.min())

            logger.debug(f"Maximum drawdown: {max_dd:.2f}%")

            # Score based on max drawdown
            if max_dd < 10:  # < 10% drawdown
                return 100.0
            elif max_dd < 15:
                return 90.0
            elif max_dd < 20:
                return 75.0
            elif max_dd < 30:
                return 55.0
            elif max_dd < 40:
                return 35.0
            elif max_dd < 50:
                return 20.0
            else:  # > 50% drawdown
                return 10.0

        except Exception as e:
            logger.warning(f"Error calculating drawdown score: {e}")
            return 50.0
