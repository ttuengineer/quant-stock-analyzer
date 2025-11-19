"""
Base abstract interfaces for data providers.

Following Dependency Inversion Principle, we define abstractions
that concrete providers must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd

from ...models.domain import Quote, StockInfo, Fundamentals
from ...models.enums import Exchange


class IQuoteProvider(ABC):
    """Interface for real-time quote fetching."""

    @abstractmethod
    async def fetch_quote(self, ticker: str) -> Quote:
        """
        Fetch current quote for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Quote object with current market data

        Raises:
            DataNotFoundError: If ticker not found
            RateLimitError: If rate limit exceeded
            DataProviderError: For other provider errors
        """
        pass

    @abstractmethod
    async def fetch_quotes_batch(self, tickers: List[str]) -> Dict[str, Quote]:
        """
        Fetch quotes for multiple tickers (batch operation).

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to Quote

        Note:
            Implementation should handle rate limiting and batching internally
        """
        pass


class IHistoricalDataProvider(ABC):
    """Interface for historical price data."""

    @abstractmethod
    async def fetch_historical_data(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (default: 1 year ago)
            end_date: End date (default: today)
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Date

        Raises:
            DataNotFoundError: If no data available
            ValidationError: If date range invalid
        """
        pass


class IFundamentalsProvider(ABC):
    """Interface for fundamental data."""

    @abstractmethod
    async def fetch_fundamentals(self, ticker: str) -> Fundamentals:
        """
        Fetch fundamental metrics for a stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Fundamentals object with valuation, profitability, growth metrics

        Raises:
            DataNotFoundError: If fundamentals not available
        """
        pass


class IStockInfoProvider(ABC):
    """Interface for stock metadata."""

    @abstractmethod
    async def fetch_stock_info(self, ticker: str) -> StockInfo:
        """
        Fetch company information and metadata.

        Args:
            ticker: Stock ticker symbol

        Returns:
            StockInfo with company name, sector, industry, etc.

        Raises:
            DataNotFoundError: If stock info not found
        """
        pass


class IUniverseProvider(ABC):
    """
    Interface for fetching stock universes (CRITICAL for no hardcoded tickers).

    This is the key interface for dynamic stock discovery.
    """

    @abstractmethod
    async def fetch_exchange_tickers(
        self,
        exchange: Exchange,
        min_market_cap: Optional[int] = None,
        min_volume: Optional[int] = None
    ) -> List[str]:
        """
        Fetch all tickers from an exchange with optional filters.

        Args:
            exchange: Exchange to query (NASDAQ, NYSE, AMEX)
            min_market_cap: Minimum market cap filter
            min_volume: Minimum average volume filter

        Returns:
            List of ticker symbols

        Example:
            >>> provider = YahooFinanceProvider()
            >>> nasdaq_stocks = await provider.fetch_exchange_tickers(
            ...     Exchange.NASDAQ,
            ...     min_market_cap=1_000_000_000  # $1B+
            ... )
            >>> print(f"Found {len(nasdaq_stocks)} NASDAQ stocks")
        """
        pass

    @abstractmethod
    async def fetch_sp500_tickers(self) -> List[str]:
        """
        Fetch current S&P 500 constituents (live data).

        Returns:
            List of S&P 500 ticker symbols

        Note:
            This should fetch from Wikipedia or official source, not hardcoded list
        """
        pass

    @abstractmethod
    async def fetch_russell_2000_tickers(self) -> List[str]:
        """
        Fetch Russell 2000 constituents.

        Returns:
            List of Russell 2000 ticker symbols
        """
        pass

    @abstractmethod
    async def search_stocks(
        self,
        query: str,
        limit: int = 20
    ) -> List[StockInfo]:
        """
        Search for stocks by name or ticker.

        Args:
            query: Search term (company name or partial ticker)
            limit: Maximum results to return

        Returns:
            List of StockInfo for matching stocks

        Example:
            >>> results = await provider.search_stocks("Apple")
            >>> print(results[0].ticker)  # AAPL
        """
        pass


class DataProvider(
    IQuoteProvider,
    IHistoricalDataProvider,
    IFundamentalsProvider,
    IStockInfoProvider,
    IUniverseProvider,
    ABC
):
    """
    Complete data provider interface (combines all interfaces).

    Concrete providers should inherit from this base class and implement
    all methods. This follows Interface Segregation Principle while providing
    a unified interface for dependency injection.
    """

    def __init__(self, name: str):
        """
        Initialize provider.

        Args:
            name: Provider name for logging and identification
        """
        self.name = name
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize provider (e.g., create HTTP session, validate API keys).

        This follows the async initialization pattern for resources.
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup provider resources (close sessions, etc.).

        Should be called when provider is no longer needed.
        """
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized
