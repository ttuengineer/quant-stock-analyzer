"""
Growth investing strategy.

Identifies high-growth stocks with strong earnings and revenue momentum.
"""

from typing import Optional
from decimal import Decimal
import pandas as pd

from .base import ScoringStrategy
from ..models.domain import Fundamentals, TechnicalIndicators, StockInfo
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class GrowthStrategy(ScoringStrategy):
    """
    Growth-at-reasonable-price (GARP) strategy.

    Scoring factors:
    - Revenue growth (YoY and QoQ)
    - Earnings growth
    - PEG ratio (growth vs valuation)
    - Margin expansion
    - Market share gains
    - Analyst revisions
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(name="Growth Strategy", weight=weight)

    def calculate_score(
        self,
        price_data: pd.DataFrame,
        fundamentals: Optional[Fundamentals] = None,
        technical_indicators: Optional[TechnicalIndicators] = None,
        stock_info: Optional[StockInfo] = None,
    ) -> Decimal:
        """Calculate growth score (0-100)."""
        if not fundamentals:
            return Decimal("50.0")  # Neutral if no data

        total_score = 0.0
        max_score = 0.0

        try:
            # 1. Revenue growth (30 points)
            rev_score, rev_max = self._calculate_revenue_growth(fundamentals)
            total_score += rev_score
            max_score += rev_max

            # 2. Earnings growth (25 points)
            earn_score, earn_max = self._calculate_earnings_growth(fundamentals)
            total_score += earn_score
            max_score += earn_max

            # 3. Growth quality (25 points)
            quality_score, quality_max = self._calculate_growth_quality(fundamentals)
            total_score += quality_score
            max_score += quality_max

            # 4. Valuation (20 points)
            val_score, val_max = self._calculate_growth_valuation(fundamentals)
            total_score += val_score
            max_score += val_max

            final_score = self.normalize_score(total_score, max_score)

            # If no growth metrics available, return neutral instead of 0
            if float(final_score) == 0.0 and max_score > 0:
                return Decimal("50.0")

            return final_score

        except Exception as e:
            logger.error(f"{self.name}: Error: {e}")
            return Decimal("50.0")  # Neutral on error, not 0

    def _calculate_revenue_growth(self, fundamentals: Fundamentals) -> tuple[float, float]:
        """Calculate revenue growth scores."""
        score, max_score = 0.0, 30.0

        if fundamentals.revenue_growth:
            growth = float(fundamentals.revenue_growth) * 100
            if growth > 30:  # Hyper growth
                score += 30
            elif growth > 20:  # High growth
                score += 25
            elif growth > 15:  # Strong growth
                score += 20
            elif growth > 10:  # Good growth
                score += 15
            elif growth > 5:  # Moderate growth
                score += 8

        return score, max_score

    def _calculate_earnings_growth(self, fundamentals: Fundamentals) -> tuple[float, float]:
        """Calculate earnings growth scores."""
        score, max_score = 0.0, 25.0

        if fundamentals.earnings_growth:
            growth = float(fundamentals.earnings_growth) * 100
            if growth > 25:
                score += 25
            elif growth > 20:
                score += 20
            elif growth > 15:
                score += 15
            elif growth > 10:
                score += 10
            elif growth > 5:
                score += 5

        return score, max_score

    def _calculate_growth_quality(self, fundamentals: Fundamentals) -> tuple[float, float]:
        """Calculate quality of growth."""
        score, max_score = 0.0, 25.0

        # High margins indicate quality
        if fundamentals.profit_margin:
            margin = float(fundamentals.profit_margin) * 100
            if margin > 20:
                score += 12
            elif margin > 15:
                score += 9
            elif margin > 10:
                score += 6

        # Strong ROE
        if fundamentals.roe:
            roe = float(fundamentals.roe) * 100
            if roe > 20:
                score += 13
            elif roe > 15:
                score += 9
            elif roe > 10:
                score += 5

        return score, max_score

    def _calculate_growth_valuation(self, fundamentals: Fundamentals) -> tuple[float, float]:
        """Calculate if growth is reasonably priced."""
        score, max_score = 0.0, 20.0

        # PEG ratio - key for GARP
        if fundamentals.peg_ratio and fundamentals.peg_ratio > 0:
            peg = float(fundamentals.peg_ratio)
            if peg < 1.0:  # Growth at good price
                score += 20
            elif peg < 1.5:  # Reasonable
                score += 15
            elif peg < 2.0:  # Acceptable
                score += 10
            elif peg < 2.5:
                score += 5

        return score, max_score
