"""
Fama-French 5-Factor Model Strategy

Implements the academic Fama-French factor model:
- Market Risk (Beta)
- Size (SMB - Small Minus Big)
- Value (HML - High Minus Low B/M ratio)
- Profitability (RMW - Robust Minus Weak)
- Investment (CMA - Conservative Minus Aggressive)

References:
- Fama & French (2015): "A Five-Factor Asset Pricing Model"
- Fama & French (1992): "The Cross-Section of Expected Stock Returns"
"""

from decimal import Decimal
from typing import Optional
import pandas as pd

from .base import ScoringStrategy
from ..models.domain import Fundamentals, TechnicalIndicators, StockInfo
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class FamaFrenchStrategy(ScoringStrategy):
    """
    Fama-French 5-Factor scoring strategy.

    Combines five proven factors that explain stock returns:
    1. Market (Beta) - Systematic risk
    2. Size (Market Cap) - Small cap premium
    3. Value (Book-to-Market) - Value premium
    4. Profitability (Operating Profit/Equity) - Quality premium
    5. Investment (Asset Growth) - Conservative investment premium
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(name="Fama-French 5-Factor", weight=weight)

    def calculate_score(
        self,
        price_data: pd.DataFrame,
        fundamentals: Optional[Fundamentals] = None,
        technical_indicators: Optional[TechnicalIndicators] = None,
        stock_info: Optional[StockInfo] = None,
        **kwargs
    ) -> Decimal:
        """
        Calculate Fama-French 5-factor score.

        Returns:
            Score from 0-100 based on factor exposure
        """
        if fundamentals is None or stock_info is None:
            logger.warning("Missing fundamentals or stock_info for Fama-French analysis")
            return Decimal("50.0")  # Neutral score

        score = 0.0
        max_score = 100.0

        # Factor 1: Market (Beta) - 15 points
        # Prefer moderate beta (0.8-1.2) - not too risky, not too defensive
        beta_score = self._score_beta(price_data)
        score += beta_score * 0.15

        # Factor 2: Size (SMB - Small Minus Big) - 20 points
        # Small-cap premium (historically outperform)
        size_score = self._score_size(stock_info)
        score += size_score * 0.20

        # Factor 3: Value (HML - High Minus Low B/M) - 25 points
        # Value premium (low P/B, low P/E)
        value_score = self._score_value(fundamentals)
        score += value_score * 0.25

        # Factor 4: Profitability (RMW - Robust Minus Weak) - 25 points
        # Operating profitability / Book Equity
        profitability_score = self._score_profitability(fundamentals)
        score += profitability_score * 0.25

        # Factor 5: Investment (CMA - Conservative Minus Aggressive) - 15 points
        # Conservative investment (low asset growth) premium
        investment_score = self._score_investment(fundamentals)
        score += investment_score * 0.15

        logger.debug(
            f"Fama-French scores - Beta: {beta_score:.1f}, Size: {size_score:.1f}, "
            f"Value: {value_score:.1f}, Profit: {profitability_score:.1f}, "
            f"Investment: {investment_score:.1f}"
        )

        return Decimal(str(min(max_score, max(0.0, score))))

    def _score_beta(self, price_data: pd.DataFrame) -> float:
        """
        Score market risk (beta).

        Optimal: 0.8-1.2 (participates in upside, not excessive risk)
        """
        # Calculate beta from price data
        beta = self._calculate_beta(price_data)

        if beta is None:
            return 50.0  # Neutral

        # Optimal beta range: 0.8-1.2
        if 0.8 <= beta <= 1.2:
            return 100.0
        elif 0.6 <= beta < 0.8 or 1.2 < beta <= 1.5:
            return 75.0
        elif beta < 0.6:
            return 40.0  # Too defensive
        else:  # beta > 1.5
            return 30.0  # Too risky

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

    def _score_size(self, stock_info: Optional[StockInfo]) -> float:
        """
        Score size factor (market cap).

        Small-cap premium: Smaller companies historically outperform
        But not too small (liquidity risk)
        """
        if not stock_info or not stock_info.market_cap:
            return 50.0

        market_cap = stock_info.market_cap

        # Market cap ranges (in billions)
        mega_cap = 200_000_000_000  # $200B+
        large_cap = 10_000_000_000  # $10B-$200B
        mid_cap = 2_000_000_000     # $2B-$10B
        small_cap = 300_000_000     # $300M-$2B

        if market_cap >= mega_cap:
            return 40.0  # Mega-cap (low growth potential)
        elif market_cap >= large_cap:
            return 60.0  # Large-cap (stable)
        elif market_cap >= mid_cap:
            return 85.0  # Mid-cap (sweet spot)
        elif market_cap >= small_cap:
            return 100.0  # Small-cap (highest premium)
        else:
            return 50.0  # Micro-cap (too illiquid)

    def _score_value(self, fundamentals: Fundamentals) -> float:
        """
        Score value factor (book-to-market, P/E).

        Value premium: Low P/B, low P/E historically outperform
        """
        score = 0.0
        count = 0

        # Price-to-Book (lower is better for value)
        if fundamentals.price_to_book:
            pb = float(fundamentals.price_to_book)
            if pb < 1.0:
                score += 100.0  # Deep value
            elif pb < 2.0:
                score += 85.0
            elif pb < 3.0:
                score += 60.0
            elif pb < 5.0:
                score += 40.0
            else:
                score += 20.0  # Expensive
            count += 1

        # Price-to-Earnings (lower is better)
        if fundamentals.pe_ratio:
            pe = float(fundamentals.pe_ratio)
            if pe < 10:
                score += 100.0  # Deep value
            elif pe < 15:
                score += 85.0
            elif pe < 20:
                score += 60.0
            elif pe < 30:
                score += 40.0
            else:
                score += 20.0  # Expensive
            count += 1

        # Price-to-Sales (lower is better)
        if fundamentals.price_to_sales:
            ps = float(fundamentals.price_to_sales)
            if ps < 1.0:
                score += 100.0
            elif ps < 2.0:
                score += 80.0
            elif ps < 3.0:
                score += 60.0
            else:
                score += 30.0
            count += 1

        return score / count if count > 0 else 50.0

    def _score_profitability(self, fundamentals: Fundamentals) -> float:
        """
        Score profitability factor (operating profit / book equity).

        Robust profitability premium: High ROE, high margins
        """
        score = 0.0
        count = 0

        # Return on Equity (higher is better)
        if fundamentals.roe:
            roe = float(fundamentals.roe)
            if roe > 0.30:  # 30%+ ROE
                score += 100.0
            elif roe > 0.20:
                score += 85.0
            elif roe > 0.15:
                score += 70.0
            elif roe > 0.10:
                score += 55.0
            elif roe > 0:
                score += 40.0
            else:  # Negative ROE
                score += 10.0
            count += 1

        # Operating Margin (higher is better)
        if fundamentals.operating_margin:
            margin = float(fundamentals.operating_margin)
            if margin > 0.25:  # 25%+
                score += 100.0
            elif margin > 0.15:
                score += 80.0
            elif margin > 0.10:
                score += 60.0
            elif margin > 0.05:
                score += 40.0
            elif margin > 0:
                score += 20.0
            else:
                score += 10.0
            count += 1

        # Profit Margin
        if fundamentals.profit_margin:
            margin = float(fundamentals.profit_margin)
            if margin > 0.20:
                score += 100.0
            elif margin > 0.15:
                score += 85.0
            elif margin > 0.10:
                score += 70.0
            elif margin > 0.05:
                score += 50.0
            elif margin > 0:
                score += 30.0
            else:
                score += 10.0
            count += 1

        # Return on Assets
        if fundamentals.roa:
            roa = float(fundamentals.roa)
            if roa > 0.15:
                score += 100.0
            elif roa > 0.10:
                score += 80.0
            elif roa > 0.05:
                score += 60.0
            elif roa > 0:
                score += 40.0
            else:
                score += 10.0
            count += 1

        return score / count if count > 0 else 50.0

    def _score_investment(self, fundamentals: Fundamentals) -> float:
        """
        Score investment factor (asset growth).

        Conservative investment premium: Low asset growth outperforms
        (Companies reinvesting aggressively tend to underperform)
        """
        # In absence of asset growth data, use conservative proxies
        score = 0.0
        count = 0

        # Free cash flow (higher is better - means not reinvesting excessively)
        if fundamentals.free_cash_flow and fundamentals.free_cash_flow > 0:
            score += 70.0  # Positive FCF = conservative
            count += 1
        elif fundamentals.free_cash_flow and fundamentals.free_cash_flow < 0:
            score += 30.0  # Negative FCF = aggressive reinvestment
            count += 1

        # Current ratio (higher = conservative)
        if fundamentals.current_ratio:
            ratio = float(fundamentals.current_ratio)
            if ratio > 2.5:
                score += 100.0
            elif ratio > 2.0:
                score += 85.0
            elif ratio > 1.5:
                score += 70.0
            elif ratio > 1.0:
                score += 50.0
            else:
                score += 20.0
            count += 1

        # Debt-to-equity (lower = conservative)
        if fundamentals.debt_to_equity:
            de = float(fundamentals.debt_to_equity)
            if de < 0.3:
                score += 100.0
            elif de < 0.5:
                score += 85.0
            elif de < 1.0:
                score += 60.0
            elif de < 2.0:
                score += 35.0
            else:
                score += 15.0
            count += 1

        return score / count if count > 0 else 50.0
