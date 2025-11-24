"""
Quality Factor Strategy

Focuses on high-quality companies with:
- High profitability (ROE, ROA, margins)
- Stable earnings
- Low debt
- Strong cash flow generation
- Consistent dividends

Quality stocks tend to outperform in uncertain markets and downturns.

References:
- Asness, Frazzini & Pedersen (2019): "Quality Minus Junk"
- Novy-Marx (2013): "The Other Side of Value"
"""

from decimal import Decimal
from typing import Optional
import pandas as pd

from .base import ScoringStrategy
from ..models.domain import Fundamentals, TechnicalIndicators, StockInfo
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class QualityStrategy(ScoringStrategy):
    """
    Quality factor scoring strategy.

    Rewards companies with:
    - High and stable profitability
    - Strong balance sheet
    - Consistent cash flow
    - Sustainable dividends
    - Low earnings variability
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(name="Quality Factor", weight=weight)

    def calculate_score(
        self,
        price_data: pd.DataFrame,
        fundamentals: Optional[Fundamentals] = None,
        technical_indicators: Optional[TechnicalIndicators] = None,
        stock_info: Optional[StockInfo] = None,
        **kwargs
    ) -> Decimal:
        """
        Calculate quality score.

        Returns:
            Score from 0-100 based on quality metrics
        """
        if fundamentals is None:
            logger.warning("Missing fundamentals for quality analysis")
            return Decimal("50.0")

        score = 0.0

        # Component 1: Profitability (35 points)
        profitability_score = self._score_profitability(fundamentals)
        score += profitability_score * 0.35

        # Component 2: Financial Strength (30 points)
        financial_strength = self._score_financial_strength(fundamentals)
        score += financial_strength * 0.30

        # Component 3: Cash Flow Quality (20 points)
        cash_flow_score = self._score_cash_flow(fundamentals)
        score += cash_flow_score * 0.20

        # Component 4: Dividend Quality (15 points)
        dividend_score = self._score_dividend_quality(fundamentals)
        score += dividend_score * 0.15

        logger.debug(
            f"Quality components - Profitability: {profitability_score:.1f}, "
            f"Financial: {financial_strength:.1f}, Cash Flow: {cash_flow_score:.1f}, "
            f"Dividend: {dividend_score:.1f}"
        )

        return Decimal(str(min(100.0, max(0.0, score))))

    def _score_profitability(self, fundamentals: Fundamentals) -> float:
        """
        Score profitability metrics.

        High ROE, ROA, margins indicate quality earnings.
        """
        score = 0.0
        count = 0

        # Return on Equity (most important)
        if fundamentals.roe:
            roe = float(fundamentals.roe)
            roe_score = 0.0
            if roe > 0.25:  # 25%+ exceptional
                roe_score = 100.0
            elif roe > 0.20:  # 20%+ excellent
                roe_score = 90.0
            elif roe > 0.15:  # 15%+ very good
                roe_score = 75.0
            elif roe > 0.10:  # 10%+ good
                roe_score = 60.0
            elif roe > 0.05:  # 5%+ acceptable
                roe_score = 40.0
            elif roe > 0:
                roe_score = 20.0
            else:
                roe_score = 5.0  # Negative ROE = poor quality
            # Double weight - add score twice
            score += roe_score
            score += roe_score
            count += 2

        # Return on Assets
        if fundamentals.roa:
            roa = float(fundamentals.roa)
            if roa > 0.15:
                score += 100.0
            elif roa > 0.10:
                score += 85.0
            elif roa > 0.07:
                score += 70.0
            elif roa > 0.05:
                score += 55.0
            elif roa > 0:
                score += 30.0
            else:
                score += 5.0
            count += 1

        # Profit Margin
        if fundamentals.profit_margin:
            margin = float(fundamentals.profit_margin)
            if margin > 0.20:  # 20%+ exceptional
                score += 100.0
            elif margin > 0.15:
                score += 85.0
            elif margin > 0.10:
                score += 70.0
            elif margin > 0.05:
                score += 50.0
            elif margin > 0:
                score += 25.0
            else:
                score += 5.0
            count += 1

        # Operating Margin
        if fundamentals.operating_margin:
            margin = float(fundamentals.operating_margin)
            if margin > 0.25:
                score += 100.0
            elif margin > 0.20:
                score += 85.0
            elif margin > 0.15:
                score += 70.0
            elif margin > 0.10:
                score += 55.0
            elif margin > 0:
                score += 30.0
            else:
                score += 5.0
            count += 1

        # Gross Margin
        if fundamentals.gross_margin:
            margin = float(fundamentals.gross_margin)
            if margin > 0.60:
                score += 100.0
            elif margin > 0.50:
                score += 85.0
            elif margin > 0.40:
                score += 70.0
            elif margin > 0.30:
                score += 55.0
            else:
                score += 30.0
            count += 1

        return score / count if count > 0 else 50.0

    def _score_financial_strength(self, fundamentals: Fundamentals) -> float:
        """
        Score balance sheet strength.

        Low debt, high current ratio indicate financial stability.
        """
        score = 0.0
        count = 0

        # Debt-to-Equity (lower is better)
        if fundamentals.debt_to_equity is not None:
            de = float(fundamentals.debt_to_equity)
            de_score = 0.0
            if de < 0.20:  # Minimal debt
                de_score = 100.0
            elif de < 0.50:  # Low debt
                de_score = 85.0
            elif de < 1.00:  # Moderate debt
                de_score = 65.0
            elif de < 1.50:  # Elevated debt
                de_score = 45.0
            elif de < 2.50:  # High debt
                de_score = 25.0
            else:  # Excessive debt
                de_score = 10.0
            # Double weight - add score twice
            score += de_score
            score += de_score
            count += 2

        # Current Ratio (liquidity)
        if fundamentals.current_ratio:
            ratio = float(fundamentals.current_ratio)
            if ratio > 3.0:  # Very strong
                score += 100.0
            elif ratio > 2.5:
                score += 90.0
            elif ratio > 2.0:
                score += 80.0
            elif ratio > 1.5:
                score += 65.0
            elif ratio > 1.0:
                score += 45.0
            else:  # < 1.0 = liquidity risk
                score += 15.0
            count += 1

        # Quick Ratio (acid test)
        if fundamentals.quick_ratio:
            ratio = float(fundamentals.quick_ratio)
            if ratio > 2.0:
                score += 100.0
            elif ratio > 1.5:
                score += 85.0
            elif ratio > 1.0:
                score += 70.0
            elif ratio > 0.7:
                score += 50.0
            else:
                score += 25.0
            count += 1

        return score / count if count > 0 else 50.0

    def _score_cash_flow(self, fundamentals: Fundamentals) -> float:
        """
        Score cash flow quality.

        Positive, growing free cash flow is essential for quality.
        """
        if fundamentals.free_cash_flow is None:
            return 50.0

        fcf = fundamentals.free_cash_flow

        # Positive FCF is critical
        if fcf > 1_000_000_000:  # $1B+ FCF (large, stable)
            return 100.0
        elif fcf > 500_000_000:  # $500M+
            return 90.0
        elif fcf > 100_000_000:  # $100M+
            return 80.0
        elif fcf > 0:  # Positive
            return 70.0
        elif fcf > -100_000_000:  # Slightly negative
            return 40.0
        else:  # Significantly negative
            return 20.0

    def _score_dividend_quality(self, fundamentals: Fundamentals) -> float:
        """
        Score dividend quality and sustainability.

        Consistent dividends with sustainable payout ratios.
        """
        score = 0.0
        count = 0

        # Dividend yield (moderate is good, too high is risky)
        if fundamentals.dividend_yield:
            div_yield = float(fundamentals.dividend_yield)
            if 0.02 <= div_yield <= 0.04:  # 2-4% sweet spot
                score += 100.0
            elif 0.01 <= div_yield < 0.02:  # 1-2% acceptable
                score += 75.0
            elif 0.04 < div_yield <= 0.06:  # 4-6% good
                score += 85.0
            elif div_yield > 0.06:  # > 6% - potentially unsustainable
                score += 50.0
            else:  # No dividend
                score += 60.0  # Not necessarily bad
            count += 1

        # Payout ratio (sustainable)
        if fundamentals.payout_ratio:
            payout = float(fundamentals.payout_ratio)
            if 0.30 <= payout <= 0.60:  # 30-60% sweet spot
                score += 100.0
            elif 0.20 <= payout < 0.30:  # Conservative
                score += 90.0
            elif 0.60 < payout <= 0.80:  # Moderate
                score += 75.0
            elif 0.80 < payout <= 1.00:  # High but sustainable
                score += 50.0
            elif payout > 1.00:  # Unsustainable
                score += 20.0
            else:  # Very low payout
                score += 70.0
            count += 1

        return score / count if count > 0 else 60.0  # Neutral if no dividend
