"""
Value investing strategy.

Identifies undervalued stocks using fundamental analysis.
Based on principles from Warren Buffett, Benjamin Graham, and modern value factors.
"""

from typing import Optional
from decimal import Decimal
import pandas as pd

from .base import ScoringStrategy
from ..models.domain import Fundamentals, TechnicalIndicators, StockInfo
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ValueStrategy(ScoringStrategy):
    """
    Deep value investing strategy.

    Scoring factors:
    - Valuation multiples (P/E, P/B, P/S, EV/EBITDA)
    - Profitability (margins, ROE, ROA)
    - Financial health (debt, liquidity)
    - Dividend yield
    - Price relative to intrinsic value
    - Graham defensive investor criteria
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(name="Value Strategy", weight=weight)

    def calculate_score(
        self,
        price_data: pd.DataFrame,
        fundamentals: Optional[Fundamentals] = None,
        technical_indicators: Optional[TechnicalIndicators] = None,
        stock_info: Optional[StockInfo] = None,
    ) -> Decimal:
        """Calculate value score (0-100)."""
        if not fundamentals:
            logger.warning(f"{self.name}: No fundamentals provided")
            return Decimal("50.0")  # Neutral if no data

        total_score = 0.0
        max_score = 0.0

        try:
            # 1. Valuation metrics (40 points)
            val_score, val_max = self._calculate_valuation_score(fundamentals)
            total_score += val_score
            max_score += val_max

            # 2. Profitability & quality (30 points)
            prof_score, prof_max = self._calculate_profitability_score(fundamentals)
            total_score += prof_score
            max_score += prof_max

            # 3. Financial health (20 points)
            health_score, health_max = self._calculate_financial_health(fundamentals)
            total_score += health_score
            max_score += health_max

            # 4. Shareholder returns (10 points)
            dividend_score, dividend_max = self._calculate_dividend_score(fundamentals)
            total_score += dividend_score
            max_score += dividend_max

            # Normalize to 0-100
            final_score = self.normalize_score(total_score, max_score)
            logger.debug(f"{self.name}: Score = {final_score}")
            return final_score

        except Exception as e:
            logger.error(f"{self.name}: Error calculating score: {e}")
            return Decimal("50.0")  # Neutral on error

    def _calculate_valuation_score(
        self,
        fundamentals: Fundamentals
    ) -> tuple[float, float]:
        """
        Score based on valuation multiples (lower is better for value).

        Returns:
            (score, max_score) tuple
        """
        score = 0.0
        max_score = 40.0

        try:
            # P/E Ratio (15 points)
            if fundamentals.pe_ratio and fundamentals.pe_ratio > 0:
                pe = float(fundamentals.pe_ratio)

                if pe < 10:  # Very undervalued
                    score += 15
                elif pe < 15:  # Undervalued
                    score += 12
                elif pe < 20:  # Fair value
                    score += 8
                elif pe < 25:  # Slightly overvalued
                    score += 4
                elif pe < 30:  # Overvalued
                    score += 1

            # PEG Ratio (10 points) - considers growth
            if fundamentals.peg_ratio and fundamentals.peg_ratio > 0:
                peg = float(fundamentals.peg_ratio)

                if peg < 0.5:  # Significantly undervalued relative to growth
                    score += 10
                elif peg < 1.0:  # Fairly valued (Peter Lynch's criteria)
                    score += 8
                elif peg < 1.5:  # Acceptable
                    score += 5
                elif peg < 2.0:  # Slightly expensive
                    score += 2

            # Price to Book (8 points)
            if fundamentals.price_to_book and fundamentals.price_to_book > 0:
                pb = float(fundamentals.price_to_book)

                if pb < 1.0:  # Trading below book value (Graham criteria)
                    score += 8
                elif pb < 2.0:  # Reasonable
                    score += 6
                elif pb < 3.0:  # Fair
                    score += 3
                elif pb < 5.0:  # High
                    score += 1

            # Price to Sales (4 points)
            if fundamentals.price_to_sales and fundamentals.price_to_sales > 0:
                ps = float(fundamentals.price_to_sales)

                if ps < 1.0:  # Very cheap
                    score += 4
                elif ps < 2.0:  # Reasonable
                    score += 3
                elif ps < 3.0:  # Fair
                    score += 1

            # EV/EBITDA (3 points)
            if fundamentals.ev_to_ebitda and fundamentals.ev_to_ebitda > 0:
                ev_ebitda = float(fundamentals.ev_to_ebitda)

                if ev_ebitda < 8:  # Undervalued
                    score += 3
                elif ev_ebitda < 12:  # Fair
                    score += 2
                elif ev_ebitda < 15:  # Acceptable
                    score += 1

        except Exception as e:
            logger.warning(f"Error calculating valuation score: {e}")

        return score, max_score

    def _calculate_profitability_score(
        self,
        fundamentals: Fundamentals
    ) -> tuple[float, float]:
        """
        Score based on profitability metrics (higher is better).

        Returns:
            (score, max_score) tuple
        """
        score = 0.0
        max_score = 30.0

        try:
            # Profit Margin (10 points)
            if fundamentals.profit_margin:
                margin = float(fundamentals.profit_margin) * 100  # Convert to %

                if margin > 20:  # Excellent
                    score += 10
                elif margin > 15:  # Very good
                    score += 8
                elif margin > 10:  # Good
                    score += 6
                elif margin > 5:  # Acceptable
                    score += 3
                elif margin > 0:  # Positive
                    score += 1

            # ROE - Return on Equity (10 points)
            if fundamentals.roe:
                roe = float(fundamentals.roe) * 100

                if roe > 20:  # Exceptional (Buffett's criteria: >15%)
                    score += 10
                elif roe > 15:  # Excellent
                    score += 8
                elif roe > 10:  # Good
                    score += 5
                elif roe > 5:  # Acceptable
                    score += 2

            # ROA - Return on Assets (5 points)
            if fundamentals.roa:
                roa = float(fundamentals.roa) * 100

                if roa > 10:  # Excellent
                    score += 5
                elif roa > 5:  # Good
                    score += 3
                elif roa > 2:  # Acceptable
                    score += 1

            # Operating Margin (5 points)
            if fundamentals.operating_margin:
                op_margin = float(fundamentals.operating_margin) * 100

                if op_margin > 20:
                    score += 5
                elif op_margin > 15:
                    score += 4
                elif op_margin > 10:
                    score += 2

        except Exception as e:
            logger.warning(f"Error calculating profitability score: {e}")

        return score, max_score

    def _calculate_financial_health(
        self,
        fundamentals: Fundamentals
    ) -> tuple[float, float]:
        """
        Score based on financial health and safety.

        Returns:
            (score, max_score) tuple
        """
        score = 0.0
        max_score = 20.0

        try:
            # Debt to Equity (10 points) - lower is better
            if fundamentals.debt_to_equity is not None:
                de_ratio = float(fundamentals.debt_to_equity)

                if de_ratio < 0.3:  # Very strong balance sheet
                    score += 10
                elif de_ratio < 0.5:  # Strong
                    score += 8
                elif de_ratio < 1.0:  # Healthy (Graham criteria: <1.0)
                    score += 6
                elif de_ratio < 1.5:  # Acceptable
                    score += 3
                elif de_ratio < 2.0:  # High but manageable
                    score += 1

            # Current Ratio (5 points) - liquidity
            if fundamentals.current_ratio:
                current = float(fundamentals.current_ratio)

                if current > 2.0:  # Excellent liquidity (Graham: >2.0)
                    score += 5
                elif current > 1.5:  # Good
                    score += 4
                elif current > 1.0:  # Acceptable
                    score += 2

            # Free Cash Flow (5 points)
            if fundamentals.free_cash_flow and fundamentals.free_cash_flow > 0:
                score += 5  # Positive FCF is important

        except Exception as e:
            logger.warning(f"Error calculating financial health: {e}")

        return score, max_score

    def _calculate_dividend_score(
        self,
        fundamentals: Fundamentals
    ) -> tuple[float, float]:
        """
        Score based on dividend yield and sustainability.

        Returns:
            (score, max_score) tuple
        """
        score = 0.0
        max_score = 10.0

        try:
            # Dividend Yield (7 points)
            if fundamentals.dividend_yield:
                div_yield = float(fundamentals.dividend_yield) * 100  # Convert to %

                if div_yield > 4:  # High yield
                    score += 7
                elif div_yield > 3:  # Good yield
                    score += 6
                elif div_yield > 2:  # Moderate yield
                    score += 4
                elif div_yield > 1:  # Low yield
                    score += 2

            # Payout Ratio (3 points) - sustainability
            if fundamentals.payout_ratio:
                payout = float(fundamentals.payout_ratio) * 100

                # Ideal payout ratio: 30-60% (sustainable)
                if 30 <= payout <= 60:
                    score += 3
                elif 20 <= payout < 30 or 60 < payout <= 70:
                    score += 2
                elif payout < 80:  # Still sustainable
                    score += 1

        except Exception as e:
            logger.warning(f"Error calculating dividend score: {e}")

        return score, max_score
