"""
Base strategy interface for stock scoring.

Following Strategy Pattern for pluggable scoring algorithms.
"""

from abc import ABC, abstractmethod
from typing import Optional
from decimal import Decimal
import pandas as pd

from ..models.domain import (
    Fundamentals,
    TechnicalIndicators,
    FactorScores,
    StockInfo,
)


class ScoringStrategy(ABC):
    """
    Abstract base class for stock scoring strategies.

    Each strategy implements a specific investment philosophy
    (value, growth, momentum, etc.) and returns a score 0-100.
    """

    def __init__(self, name: str, weight: float = 1.0):
        """
        Initialize strategy.

        Args:
            name: Strategy name for identification
            weight: Weight in composite scoring (default: 1.0)
        """
        self.name = name
        self.weight = weight

    @abstractmethod
    def calculate_score(
        self,
        price_data: pd.DataFrame,
        fundamentals: Optional[Fundamentals] = None,
        technical_indicators: Optional[TechnicalIndicators] = None,
        stock_info: Optional[StockInfo] = None,
    ) -> Decimal:
        """
        Calculate opportunity score for a stock.

        Args:
            price_data: Historical OHLCV data
            fundamentals: Fundamental metrics (optional)
            technical_indicators: Technical indicators (optional)
            stock_info: Stock metadata (optional)

        Returns:
            Score from 0 to 100 (higher is better)

        Note:
            Implementations should handle missing data gracefully
            and return 0 if insufficient data available.
        """
        pass

    def normalize_score(self, raw_score: float, max_score: float) -> Decimal:
        """
        Normalize raw score to 0-100 range.

        Args:
            raw_score: Raw score value
            max_score: Maximum possible score

        Returns:
            Normalized score 0-100
        """
        if max_score <= 0:
            return Decimal("0")

        normalized = (raw_score / max_score) * 100
        return Decimal(str(min(100, max(0, normalized))))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.weight})"
