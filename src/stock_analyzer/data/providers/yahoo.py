"""
Yahoo Finance data provider implementation.

Supports both RapidAPI Yahoo Finance (paid, faster) and free yfinance library.
Uses RapidAPI when API key is available, falls back to yfinance otherwise.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import pandas as pd
from decimal import Decimal
import yfinance as yf
import aiohttp
import requests
from bs4 import BeautifulSoup

from .base import DataProvider
from ...models.domain import Quote, StockInfo, Fundamentals
from ...models.enums import Exchange, Sector, AssetClass
from ...utils.logger import setup_logger
from ...utils.decorators import async_retry, validate_ticker, timing
from ...utils.exceptions import (
    DataNotFoundError,
    DataProviderError,
    RateLimitError,
)
from ...config.settings import get_settings

logger = setup_logger(__name__)


class YahooFinanceProvider(DataProvider):
    """
    Yahoo Finance data provider via RapidAPI.

    Uses RapidAPI's yahoo-finance15 API as primary source.
    Falls back to free yfinance library if RapidAPI fails.
    """

    def __init__(self):
        super().__init__(name="Yahoo Finance")
        self._session: Optional[aiohttp.ClientSession] = None
        self._settings = get_settings()

        # Get RapidAPI credentials
        self._rapidapi_key = None
        self._rapidapi_host = "yahoo-finance15.p.rapidapi.com"

        if self._settings.rapidapi_key:
            self._rapidapi_key = self._settings.rapidapi_key.get_secret_value()

        if self._settings.rapidapi_host:
            self._rapidapi_host = self._settings.rapidapi_host

    async def initialize(self) -> None:
        """Initialize HTTP session."""
        if not self._initialized:
            self._session = aiohttp.ClientSession()
            self._initialized = True
            if self._rapidapi_key:
                logger.info(f"Yahoo Finance provider initialized via RapidAPI ({self._rapidapi_host})")
            else:
                logger.warning("No RapidAPI key configured - falling back to free yfinance (may be unreliable)")

    async def cleanup(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._initialized = False
            logger.info("Yahoo Finance provider cleaned up")

    async def _rapidapi_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a request to RapidAPI Yahoo Finance using synchronous requests.

        Uses synchronous requests library to avoid event loop issues in Streamlit.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            JSON response data
        """
        if not self._rapidapi_key:
            raise DataProviderError("RapidAPI key not configured")

        def _sync_request():
            headers = {
                "x-rapidapi-key": self._rapidapi_key,
                "x-rapidapi-host": self._rapidapi_host
            }

            url = f"https://{self._rapidapi_host}{endpoint}"

            response = requests.get(url, headers=headers, params=params, timeout=10)

            if response.status_code == 429:
                raise RateLimitError("RapidAPI rate limit exceeded", retry_after=60)
            elif response.status_code == 403:
                raise DataProviderError(f"RapidAPI authentication failed - check API key and subscription")
            elif response.status_code != 200:
                raise DataProviderError(f"RapidAPI returned status {response.status_code}: {response.text}")

            return response.json()

        # Run synchronous request in executor to keep async interface
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_request)

    @timing
    @validate_ticker
    @async_retry(max_attempts=3, delay=1.0)
    async def fetch_quote(self, ticker: str) -> Quote:
        """
        Fetch current quote from Yahoo Finance via RapidAPI.

        Falls back to free yfinance if RapidAPI unavailable.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Quote object with current market data
        """
        quote = None

        # Try RapidAPI first
        if self._rapidapi_key:
            try:
                # yahoo-finance15 uses /api/v1/markets/quote endpoint (correct format)
                data = await self._rapidapi_request(
                    "/api/v1/markets/quote",
                    {"ticker": ticker, "type": "STOCKS"}
                )

                # Parse RapidAPI response
                if data and "body" in data:
                    body = data["body"]
                    if isinstance(body, list) and len(body) > 0:
                        stock_data = body[0]
                    elif isinstance(body, dict):
                        stock_data = body
                    else:
                        stock_data = data

                    quote = Quote(
                        ticker=ticker,
                        price=Decimal(str(stock_data.get('regularMarketPrice', 0))),
                        open=Decimal(str(stock_data.get('regularMarketOpen', 0))) if stock_data.get('regularMarketOpen') else None,
                        high=Decimal(str(stock_data.get('regularMarketDayHigh', 0))) if stock_data.get('regularMarketDayHigh') else None,
                        low=Decimal(str(stock_data.get('regularMarketDayLow', 0))) if stock_data.get('regularMarketDayLow') else None,
                        volume=int(stock_data.get('regularMarketVolume', 0)) if stock_data.get('regularMarketVolume') else 0,
                        previous_close=Decimal(str(stock_data.get('regularMarketPreviousClose', 0))) if stock_data.get('regularMarketPreviousClose') else None,
                        timestamp=datetime.utcnow()
                    )
                    logger.debug(f"Fetched quote for {ticker} via RapidAPI: ${quote.price}")

            except Exception as e:
                logger.warning(f"RapidAPI failed for {ticker}, falling back to yfinance: {e}")
                quote = None

        # Fallback to yfinance if RapidAPI not available or failed
        if quote is None:
            try:
                loop = asyncio.get_event_loop()
                stock = await loop.run_in_executor(None, yf.Ticker, ticker)
                info = await loop.run_in_executor(None, lambda: stock.info)

                if not info or 'regularMarketPrice' not in info:
                    raise DataNotFoundError(f"No quote data found for {ticker}")

                quote = Quote(
                    ticker=ticker,
                    price=Decimal(str(info.get('regularMarketPrice', 0))),
                    open=Decimal(str(info.get('regularMarketOpen', 0))) if info.get('regularMarketOpen') else None,
                    high=Decimal(str(info.get('dayHigh', 0))) if info.get('dayHigh') else None,
                    low=Decimal(str(info.get('dayLow', 0))) if info.get('dayLow') else None,
                    volume=int(info.get('regularMarketVolume', 0)),
                    previous_close=Decimal(str(info.get('previousClose', 0))) if info.get('previousClose') else None,
                    timestamp=datetime.utcnow()
                )
                logger.debug(f"Fetched quote for {ticker} via yfinance: ${quote.price}")

            except Exception as e:
                logger.error(f"Error fetching quote for {ticker}: {e}")
                raise DataProviderError(f"Failed to fetch quote for {ticker}: {e}")

        # Calculate change
        if quote and quote.previous_close and quote.previous_close > 0:
            quote.change = quote.price - quote.previous_close
            quote.change_percent = (quote.change / quote.previous_close) * 100

        return quote

    @timing
    async def fetch_quotes_batch(self, tickers: List[str]) -> Dict[str, Quote]:
        """
        Fetch quotes for multiple tickers in parallel.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to Quote
        """
        logger.info(f"Fetching quotes for {len(tickers)} tickers")

        # Create tasks for parallel execution
        tasks = [self.fetch_quote(ticker) for ticker in tickers]

        # Execute with gather (returns exceptions for failed tickers)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dictionary (exclude errors)
        quotes = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Quote):
                quotes[ticker] = result
            else:
                logger.warning(f"Failed to fetch quote for {ticker}: {result}")

        logger.info(f"Successfully fetched {len(quotes)}/{len(tickers)} quotes")
        return quotes

    @timing
    @validate_ticker
    @async_retry(max_attempts=2)
    async def fetch_historical_data(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (default: 1 year ago)
            end_date: End date (default: today)
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        try:
            loop = asyncio.get_event_loop()
            stock = yf.Ticker(ticker)

            df = await loop.run_in_executor(
                None,
                lambda: stock.history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )
            )

            if df.empty:
                raise DataNotFoundError(f"No historical data for {ticker}")

            logger.debug(f"Fetched {len(df)} bars for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            raise DataProviderError(f"Failed to fetch historical data: {e}")

    @timing
    @validate_ticker
    @async_retry(max_attempts=2)
    async def fetch_fundamentals(self, ticker: str) -> Fundamentals:
        """
        Fetch fundamental metrics from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Fundamentals object
        """
        try:
            loop = asyncio.get_event_loop()
            stock = yf.Ticker(ticker)
            info = await loop.run_in_executor(None, lambda: stock.info)

            if not info:
                raise DataNotFoundError(f"No fundamentals for {ticker}")

            fundamentals = Fundamentals(
                pe_ratio=Decimal(str(info['trailingPE'])) if info.get('trailingPE') else None,
                forward_pe=Decimal(str(info['forwardPE'])) if info.get('forwardPE') else None,
                peg_ratio=Decimal(str(info['pegRatio'])) if info.get('pegRatio') else None,
                price_to_book=Decimal(str(info['priceToBook'])) if info.get('priceToBook') else None,
                price_to_sales=Decimal(str(info['priceToSalesTrailing12Months'])) if info.get('priceToSalesTrailing12Months') else None,
                ev_to_ebitda=Decimal(str(info['enterpriseToEbitda'])) if info.get('enterpriseToEbitda') else None,
                profit_margin=Decimal(str(info['profitMargins'])) if info.get('profitMargins') else None,
                operating_margin=Decimal(str(info['operatingMargins'])) if info.get('operatingMargins') else None,
                gross_margin=Decimal(str(info['grossMargins'])) if info.get('grossMargins') else None,
                roe=Decimal(str(info['returnOnEquity'])) if info.get('returnOnEquity') else None,
                roa=Decimal(str(info['returnOnAssets'])) if info.get('returnOnAssets') else None,
                revenue_growth=Decimal(str(info['revenueGrowth'])) if info.get('revenueGrowth') else None,
                earnings_growth=Decimal(str(info['earningsGrowth'])) if info.get('earningsGrowth') else None,
                debt_to_equity=Decimal(str(info['debtToEquity'])) if info.get('debtToEquity') else None,
                current_ratio=Decimal(str(info['currentRatio'])) if info.get('currentRatio') else None,
                quick_ratio=Decimal(str(info['quickRatio'])) if info.get('quickRatio') else None,
                free_cash_flow=int(info['freeCashflow']) if info.get('freeCashflow') else None,
                dividend_yield=Decimal(str(info['dividendYield'])) if info.get('dividendYield') else None,
                payout_ratio=Decimal(str(info['payoutRatio'])) if info.get('payoutRatio') else None,
                analyst_rating=Decimal(str(info['recommendationMean'])) if info.get('recommendationMean') else None,
                analyst_target_price=Decimal(str(info['targetMeanPrice'])) if info.get('targetMeanPrice') else None,
                beta=Decimal(str(info['beta'])) if info.get('beta') else None,
            )

            logger.debug(f"Fetched fundamentals for {ticker}")
            return fundamentals

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            raise DataProviderError(f"Failed to fetch fundamentals: {e}")

    @timing
    @validate_ticker
    @async_retry(max_attempts=2)
    async def fetch_stock_info(self, ticker: str) -> StockInfo:
        """
        Fetch stock metadata from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            StockInfo object
        """
        try:
            loop = asyncio.get_event_loop()
            stock = yf.Ticker(ticker)
            info = await loop.run_in_executor(None, lambda: stock.info)

            if not info:
                raise DataNotFoundError(f"No info for {ticker}")

            # Map sector string to enum
            sector_str = info.get('sector')
            sector = None
            if sector_str:
                try:
                    sector = Sector(sector_str)
                except ValueError:
                    logger.warning(f"Unknown sector: {sector_str}")

            # Map exchange
            exchange_str = info.get('exchange', '').upper()
            exchange = None
            if 'NASDAQ' in exchange_str or 'NMS' in exchange_str:
                exchange = Exchange.NASDAQ
            elif 'NYSE' in exchange_str or 'NYQ' in exchange_str:
                exchange = Exchange.NYSE
            elif 'AMEX' in exchange_str or 'ASE' in exchange_str:
                exchange = Exchange.AMEX
            else:
                exchange = Exchange.OTHER

            stock_info = StockInfo(
                ticker=ticker,
                company_name=info.get('longName') or info.get('shortName'),
                exchange=exchange,
                sector=sector,
                industry=info.get('industry'),
                asset_class=AssetClass.STOCK,
                market_cap=int(info['marketCap']) if info.get('marketCap') else None,
                shares_outstanding=int(info['sharesOutstanding']) if info.get('sharesOutstanding') else None,
                description=info.get('longBusinessSummary'),
                website=info.get('website'),
            )

            logger.debug(f"Fetched info for {ticker}: {stock_info.company_name}")
            return stock_info

        except Exception as e:
            logger.error(f"Error fetching stock info for {ticker}: {e}")
            raise DataProviderError(f"Failed to fetch stock info: {e}")

    @timing
    @async_retry(max_attempts=3, delay=2.0)
    async def fetch_sp500_tickers(self) -> List[str]:
        """
        Fetch S&P 500 constituents from Wikipedia (LIVE data, not hardcoded).

        Returns:
            List of S&P 500 ticker symbols
        """
        logger.info("Fetching live S&P 500 constituents from Wikipedia")

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        try:
            if not self._session:
                await self.initialize()

            # Add headers to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            async with self._session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise DataProviderError(f"Wikipedia returned status {response.status}")

                html = await response.text()

            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table', {'id': 'constituents'})

            if not table:
                raise DataProviderError("Could not find S&P 500 table on Wikipedia")

            tickers = []
            rows = table.find_all('tr')[1:]  # Skip header

            for row in rows:
                cols = row.find_all('td')
                if cols:
                    ticker = cols[0].text.strip()
                    # Clean up ticker (remove newlines, spaces)
                    ticker = ticker.replace('\n', '').strip()
                    tickers.append(ticker)

            logger.info(f"Fetched {len(tickers)} S&P 500 tickers from Wikipedia")
            return tickers

        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers: {e}")
            raise DataProviderError(f"Failed to fetch S&P 500 tickers: {e}")

    @timing
    async def fetch_russell_2000_tickers(self) -> List[str]:
        """
        Fetch Russell 2000 constituents.

        Note: Russell 2000 list requires data provider subscription.
        This is a placeholder - implement when provider available.
        """
        logger.warning("Russell 2000 fetching not yet implemented")
        return []

    @timing
    async def fetch_exchange_tickers(
        self,
        exchange: Exchange,
        min_market_cap: Optional[int] = None,
        min_volume: Optional[int] = None
    ) -> List[str]:
        """
        Fetch all tickers from an exchange.

        Note: This requires scraping or a specialized API.
        For now, we'll use S&P 500 as a proxy for large-cap stocks.

        Args:
            exchange: Exchange to query
            min_market_cap: Minimum market cap filter
            min_volume: Minimum volume filter

        Returns:
            List of ticker symbols
        """
        logger.info(f"Fetching tickers for {exchange.value}")

        # For now, use S&P 500 as the primary universe
        # In production, integrate with a screener API
        tickers = await self.fetch_sp500_tickers()

        # Filter by exchange if specified
        if exchange != Exchange.OTHER:
            filtered_tickers = []
            for ticker in tickers:
                try:
                    info = await self.fetch_stock_info(ticker)
                    if info.exchange == exchange:
                        filtered_tickers.append(ticker)
                except Exception:
                    continue

            tickers = filtered_tickers

        logger.info(f"Found {len(tickers)} tickers for {exchange.value}")
        return tickers

    @timing
    async def search_stocks(self, query: str, limit: int = 20) -> List[StockInfo]:
        """
        Search for stocks by name or ticker.

        Args:
            query: Search term
            limit: Maximum results

        Returns:
            List of StockInfo for matching stocks
        """
        logger.info(f"Searching for stocks matching '{query}'")

        # yfinance doesn't have direct search, so we'll use a simple approach
        # In production, integrate with a proper search API

        # Try exact ticker match first
        try:
            info = await self.fetch_stock_info(query.upper())
            return [info]
        except Exception:
            pass

        logger.warning("Stock search requires additional API integration")
        return []
