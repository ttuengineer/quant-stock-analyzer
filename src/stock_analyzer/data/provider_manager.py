"""
Multi-provider manager with automatic fallback.

Implements resilience patterns: circuit breaker, fallback, retry.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio

from .providers.base import DataProvider
from .providers.yahoo import YahooFinanceProvider
from ..models.domain import Quote, StockInfo, Fundamentals
from ..models.enums import Exchange
from ..utils.logger import setup_logger
from ..utils.exceptions import ProviderUnavailableError, DataProviderError
from ..config import get_settings

logger = setup_logger(__name__)


class ProviderManager:
    """
    Manages multiple data providers with automatic fallback.

    Follows the circuit breaker pattern and automatically switches to
    backup providers when primary fails.
    """

    def __init__(self, providers: Optional[List[DataProvider]] = None):
        """
        Initialize provider manager.

        Args:
            providers: List of providers in priority order
                      If None, uses default providers based on config
        """
        self.settings = get_settings()
        self._providers: List[DataProvider] = providers or self._init_default_providers()
        self._failure_counts: Dict[str, int] = {p.name: 0 for p in self._providers}
        self._circuit_open: Dict[str, bool] = {p.name: False for p in self._providers}

    def _init_default_providers(self) -> List[DataProvider]:
        """Initialize default providers based on configuration."""
        providers = []

        # Always include Yahoo Finance (no API key required)
        providers.append(YahooFinanceProvider())

        # Add other providers if API keys are configured
        # TODO: Add Alpha Vantage, Polygon when API keys present

        logger.info(f"Initialized {len(providers)} data providers")
        return providers

    async def initialize(self) -> None:
        """Initialize all providers."""
        logger.info("Initializing all providers")
        await asyncio.gather(*[p.initialize() for p in self._providers])

    async def cleanup(self) -> None:
        """Cleanup all providers."""
        logger.info("Cleaning up all providers")
        await asyncio.gather(*[p.cleanup() for p in self._providers])

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    def _mark_failure(self, provider_name: str) -> None:
        """
        Mark a provider failure and potentially open circuit.

        Args:
            provider_name: Name of failed provider
        """
        self._failure_counts[provider_name] += 1

        if self._failure_counts[provider_name] >= 5:
            logger.warning(f"Opening circuit for {provider_name} after 5 failures")
            self._circuit_open[provider_name] = True

    def _mark_success(self, provider_name: str) -> None:
        """
        Mark a provider success and reset failure count.

        Args:
            provider_name: Name of successful provider
        """
        self._failure_counts[provider_name] = 0
        if self._circuit_open[provider_name]:
            logger.info(f"Closing circuit for {provider_name}")
            self._circuit_open[provider_name] = False

    async def _try_with_fallback(
        self,
        operation: str,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Try operation with automatic fallback to backup providers.

        Args:
            operation: Description of operation for logging
            method_name: Name of provider method to call
            *args: Arguments to pass to method
            **kwargs: Keyword arguments to pass to method

        Returns:
            Result from first successful provider

        Raises:
            ProviderUnavailableError: If all providers fail
        """
        last_error = None

        for provider in self._providers:
            # Skip if circuit is open
            if self._circuit_open[provider.name]:
                logger.debug(f"Skipping {provider.name} (circuit open)")
                continue

            try:
                logger.debug(f"Trying {operation} with {provider.name}")

                # Get method from provider
                method = getattr(provider, method_name)
                result = await method(*args, **kwargs)

                # Success!
                self._mark_success(provider.name)
                logger.debug(f"{operation} succeeded with {provider.name}")
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"{operation} failed with {provider.name}: {e}")
                self._mark_failure(provider.name)
                continue

        # All providers failed
        logger.error(f"{operation} failed with all providers")
        raise ProviderUnavailableError(
            f"All providers failed for {operation}. Last error: {last_error}"
        )

    async def fetch_quote(self, ticker: str) -> Quote:
        """
        Fetch quote with automatic fallback.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Quote from first available provider
        """
        return await self._try_with_fallback(
            f"fetch_quote({ticker})",
            "fetch_quote",
            ticker
        )

    async def fetch_quotes_batch(self, tickers: List[str]) -> Dict[str, Quote]:
        """
        Fetch multiple quotes with fallback.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to Quote
        """
        return await self._try_with_fallback(
            f"fetch_quotes_batch({len(tickers)} tickers)",
            "fetch_quotes_batch",
            tickers
        )

    async def fetch_historical_data(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ):
        """Fetch historical data with fallback."""
        return await self._try_with_fallback(
            f"fetch_historical_data({ticker})",
            "fetch_historical_data",
            ticker,
            start_date,
            end_date,
            interval
        )

    async def fetch_fundamentals(self, ticker: str) -> Fundamentals:
        """Fetch fundamentals with fallback."""
        return await self._try_with_fallback(
            f"fetch_fundamentals({ticker})",
            "fetch_fundamentals",
            ticker
        )

    async def fetch_stock_info(self, ticker: str) -> StockInfo:
        """Fetch stock info with fallback."""
        return await self._try_with_fallback(
            f"fetch_stock_info({ticker})",
            "fetch_stock_info",
            ticker
        )

    async def fetch_sp500_tickers(self) -> List[str]:
        """Fetch S&P 500 tickers with fallback."""
        return await self._try_with_fallback(
            "fetch_sp500_tickers",
            "fetch_sp500_tickers"
        )

    async def fetch_exchange_tickers(
        self,
        exchange: Exchange,
        min_market_cap: Optional[int] = None,
        min_volume: Optional[int] = None
    ) -> List[str]:
        """Fetch exchange tickers with fallback."""
        return await self._try_with_fallback(
            f"fetch_exchange_tickers({exchange.value})",
            "fetch_exchange_tickers",
            exchange,
            min_market_cap,
            min_volume
        )

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all providers.

        Returns:
            Dictionary with provider status information
        """
        status = {}
        for provider in self._providers:
            status[provider.name] = {
                "initialized": provider.is_initialized(),
                "failures": self._failure_counts[provider.name],
                "circuit_open": self._circuit_open[provider.name],
            }
        return status
