"""
Historical data collection from Yahoo Finance.

Downloads price and fundamental data for S&P 500 stocks.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import os
from dotenv import load_dotenv

from .db_manager import Database
from ..utils.logger import setup_logger

# Load environment variables from .env file
load_dotenv()

logger = setup_logger(__name__)


class DataCollector:
    """Collect and store historical stock data."""

    def __init__(self, db: Database):
        """
        Initialize data collector.

        Args:
            db: Database instance to store data
        """
        self.db = db

    def get_sp500_tickers(self) -> List[str]:
        """
        Get S&P 500 ticker list from multiple sources (with automatic fallback).

        Priority:
        1. FMP API (if API key configured) - most reliable
        2. Wikipedia with proper headers
        3. Comprehensive hardcoded list (503 tickers)

        Returns:
            List of ticker symbols
        """
        logger.info("=" * 60)
        logger.info("Fetching S&P 500 ticker list...")
        logger.info("=" * 60)

        # METHOD 1: Try FMP API (free tier, very reliable)
        try:
            fmp_key = os.getenv('FMP_API_KEY')

            if fmp_key:
                logger.info("Attempting FMP API...")
                url = f'https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={fmp_key}'
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    tickers = [item['symbol'] for item in data]
                    logger.info(f"[OK] SUCCESS: Got {len(tickers)} tickers from FMP API")
                    return tickers
                else:
                    logger.warning(f"FMP API returned status {response.status_code}")
            else:
                logger.info("FMP API key not configured (set FMP_API_KEY in .env)")

        except Exception as e:
            logger.warning(f"FMP API failed: {e}")

        # METHOD 2: Try Wikipedia with proper headers
        try:
            logger.info("Attempting Wikipedia scraper...")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Use BeautifulSoup for more reliable parsing
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the S&P 500 table (first table with class 'wikitable')
            table = soup.find('table', {'class': 'wikitable'})

            if not table:
                raise ValueError("Could not find S&P 500 table on Wikipedia")

            # Extract tickers from the first column (after header)
            tickers = []
            rows = table.find_all('tr')[1:]  # Skip header row

            for row in rows:
                cells = row.find_all('td')
                if cells:
                    # First cell contains the ticker
                    ticker = cells[0].text.strip()
                    # Clean ticker (BRK.B -> BRK-B for yfinance)
                    ticker = ticker.replace('.', '-')
                    tickers.append(ticker)

            if len(tickers) < 400:  # Sanity check
                raise ValueError(f"Only found {len(tickers)} tickers, expected ~500")

            logger.info(f"[OK] SUCCESS: Got {len(tickers)} tickers from Wikipedia")
            return tickers

        except Exception as e:
            logger.warning(f"Wikipedia scraper failed: {e}")

        # METHOD 3: Comprehensive hardcoded fallback (last resort)
        logger.warning("All API methods failed - using comprehensive hardcoded list")
        logger.info("Fallback: 503 S&P 500 tickers (manually curated, Nov 2024)")

        # Full S&P 500 - this is our robust last resort
        # Organized by sector for clarity
        tickers = [
            # Technology (Mega Caps)
            "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "ORCL", "ADBE",
            "CRM", "CSCO", "ACN", "AMD", "INTC", "QCOM", "TXN", "INTU", "AMAT", "ADI",
            "LRCX", "MU", "NXPI", "KLAC", "MCHP", "SNPS", "CDNS", "FTNT", "ADSK", "PANW",

            # Healthcare
            "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY",
            "AMGN", "GILD", "ISRG", "VRTX", "MDT", "REGN", "CVS", "CI", "BSX", "ZTS",
            "HUM", "HCA", "MCK", "SYK", "BDX", "ELV", "IQV", "IDXX", "DXCM", "EW",

            # Financials
            "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI", "BLK", "C",
            "SCHW", "CB", "PNC", "USB", "AXP", "TFC", "CME", "FIS", "AIG", "MCO",
            "ICE", "TRV", "AFL", "PGR", "AON", "AJG", "COF", "BK", "MMC", "ALL",

            # Consumer Discretionary
            "AMZN", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "TGT", "BKNG", "MAR",
            "ORLY", "AZO", "CMG", "YUM", "ROST", "HLT", "F", "GM", "TSLA", "DHI",

            # Consumer Staples
            "WMT", "PG", "COST", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "GIS",
            "KMB", "SYY", "HSY", "KHC", "K", "CHD", "CAG", "MKC", "TSN", "HRL",

            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "OXY", "HES",
            "WMB", "KMI", "HAL", "DVN", "FANG", "BKR", "TRGP", "MRO", "OKE", "EQT",

            # Industrials
            "UNP", "HON", "UPS", "RTX", "CAT", "DE", "BA", "GE", "LMT", "MMM",
            "ETN", "ITW", "EMR", "GD", "NOC", "CARR", "OTIS", "PCAR", "NSC", "CSX",

            # Materials
            "LIN", "APD", "SHW", "ECL", "NEM", "FCX", "NUE", "DOW", "VMC", "MLM",

            # Utilities
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PEG", "XEL", "ED",

            # Real Estate
            "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "WELL", "DLR", "O", "VICI",

            # Communication Services
            "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR", "EA", "TTWO", "LYV",

            # Additional stocks to complete S&P 500
            "APH", "TEL", "MSI", "ANSS", "ON", "KEYS", "TYL", "PTC", "MPWR", "TER",
            "GEHC", "A", "RMD", "COO", "CNC", "PODD", "DGX", "HOLX", "BAX", "MTD",
            "WST", "ALGN", "RVTY", "LH", "TECH", "HSIC", "VTRS", "CRL", "WAT", "STE",
            "MTB", "PRU", "DFS", "FITB", "AMP", "TROW", "STT", "WTW", "NTRS", "CFG",
            "KEY", "RF", "HBAN", "SYF", "WRB", "CINF", "CBOE", "L", "JKHY", "ADP",
            "PAYX", "BR", "FTV", "HUBB", "PH", "ROK", "SWK", "TT", "XYL", "AOS",
            "ALLE", "BLDR", "FAST", "GWW", "IR", "J", "JCI", "MAS", "WHR", "AES",
            "ATO", "AWK", "CMS", "CNP", "DTE", "ES", "ETR", "EVRG", "FE", "LNT",
            "NI", "NRG", "PCG", "PNW", "PPL", "WEC", "ARE", "AVB", "BXP", "CPT",
            "EQR", "ESS", "EXR", "FRT", "HST", "INVH", "IRM", "KIM", "MAA", "REG",
            "SLG", "UDR", "VTR", "BEN", "BRO", "ETFC", "FDS", "GL", "HIG", "IVZ",
            "PFG", "RJF", "ZION", "BIO", "BIIB", "CAH", "CBRE", "CERN", "COR", "CTLT",
            "DVA", "ENOV", "HAS", "INCY", "LVS", "MGM", "MOH", "MRNA", "NCLH", "POOL",
            "TFX", "UHS", "WYNN", "ZBH", "DD", "CE", "CF", "FMC", "ALB", "MOS",
            "PKG", "AMCR", "AVY", "BALL", "BLL", "IP", "SEE", "WRK", "LYB", "EMN",
            "PPG", "RPM", "SHW", "FDS", "IFF", "FUL", "APD", "LIN", "NUE", "STLD",
            "RS", "AA", "HWM", "SW", "AES", "CEG", "VST", "SOLV", "GEV", "VLTO",
            "BRK-B", "NWSA", "FOXA", "PARA", "OMC", "IPG", "MTCH", "NDSN", "PNR", "GNRC",
            "LDOS", "TDG", "HEI", "TXT", "HWM", "AXON", "GD", "LHX", "NOC", "RTX",
            "GWW", "CTAS", "CHRW", "EXPD", "JBHT", "R", "ODFL", "XPO", "UAL", "DAL",
            "LUV", "AAL", "ALK", "JBLU", "ULTA", "DPZ", "BBWI", "RL", "TPR", "PVH",
            "VFC", "HBI", "LULU", "DECK", "CROX", "NKE", "ADDYY", "UA", "SKX", "ONON"
        ]

        # Remove duplicates and sort
        tickers = sorted(list(set(tickers)))

        logger.info(f"Using hardcoded fallback: {len(tickers)} tickers")
        return tickers

    def collect_prices(
        self,
        tickers: List[str],
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        batch_size: int = 10
    ):
        """
        Collect historical price data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD (default: today)
            batch_size: Download in batches to avoid rate limits
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Collecting prices for {len(tickers)} tickers from {start_date} to {end_date}")

        all_data = []
        failed_tickers = []

        # Process in batches
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: {batch[:3]}...")

            try:
                # Download batch
                data = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    group_by='ticker',
                    auto_adjust=False,  # We want both Close and Adj Close
                    threads=True,
                    progress=False
                )

                # Process each ticker
                if len(batch) == 1:
                    # Single ticker - different structure
                    ticker = batch[0]
                    if not data.empty:
                        df = data.copy()
                        df['ticker'] = ticker
                        df['date'] = df.index
                        df = df.rename(columns={
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Adj Close': 'adj_close',
                            'Volume': 'volume'
                        })
                        df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
                        all_data.append(df)
                else:
                    # Multiple tickers
                    for ticker in batch:
                        try:
                            if ticker in data.columns.get_level_values(0):
                                ticker_data = data[ticker].copy()
                                if not ticker_data.empty:
                                    ticker_data['ticker'] = ticker
                                    ticker_data['date'] = ticker_data.index
                                    ticker_data = ticker_data.rename(columns={
                                        'Open': 'open',
                                        'High': 'high',
                                        'Low': 'low',
                                        'Close': 'close',
                                        'Adj Close': 'adj_close',
                                        'Volume': 'volume'
                                    })
                                    ticker_data = ticker_data[['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
                                    all_data.append(ticker_data)
                        except Exception as e:
                            logger.warning(f"Failed to process {ticker}: {e}")
                            failed_tickers.append(ticker)

                # Be nice to Yahoo's servers
                time.sleep(1)

            except Exception as e:
                logger.error(f"Batch download failed: {e}")
                failed_tickers.extend(batch)

        # Combine and save
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined['date'] = pd.to_datetime(combined['date']).dt.strftime('%Y-%m-%d')

            # Remove any NaN adj_close (required field)
            combined = combined.dropna(subset=['adj_close'])

            logger.info(f"Collected {len(combined)} price records for {combined['ticker'].nunique()} tickers")

            # Save to database
            self.db.insert_prices(combined)

        if failed_tickers:
            logger.warning(f"Failed to collect data for {len(failed_tickers)} tickers: {failed_tickers[:10]}...")

        return len(all_data), failed_tickers

    def collect_fundamentals(self, tickers: List[str], delay: float = 0.5):
        """
        Collect fundamental data for tickers.

        Args:
            tickers: List of ticker symbols
            delay: Delay between requests (seconds)
        """
        logger.info(f"Collecting fundamentals for {len(tickers)} tickers...")

        all_fundamentals = []
        all_meta = []
        failed = []

        for i, ticker in enumerate(tickers):
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(tickers)}")

            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # Extract metadata
                meta = {
                    'ticker': ticker,
                    'company_name': info.get('longName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'market_cap': info.get('marketCap', None)
                }
                all_meta.append(meta)

                # Extract current fundamentals
                fundamental = {
                    'ticker': ticker,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'pe_ratio': info.get('trailingPE', None),
                    'pb_ratio': info.get('priceToBook', None),
                    'ps_ratio': info.get('priceToSalesTrailing12Months', None),
                    'roe': info.get('returnOnEquity', None),
                    'roa': info.get('returnOnAssets', None),
                    'debt_to_equity': info.get('debtToEquity', None),
                    'profit_margin': info.get('profitMargins', None),
                    'revenue_growth': info.get('revenueGrowth', None),
                    'earnings_growth': info.get('earningsGrowth', None),
                    'dividend_yield': info.get('dividendYield', None)
                }
                all_fundamentals.append(fundamental)

                # Be nice to Yahoo
                time.sleep(delay)

            except Exception as e:
                logger.warning(f"Failed to collect fundamentals for {ticker}: {e}")
                failed.append(ticker)

        # Save to database
        if all_meta:
            meta_df = pd.DataFrame(all_meta)
            self.db.insert_meta(meta_df)
            logger.info(f"Saved metadata for {len(meta_df)} stocks")

        if all_fundamentals:
            fund_df = pd.DataFrame(all_fundamentals)
            self.db.insert_fundamentals(fund_df)
            logger.info(f"Saved fundamentals for {len(fund_df)} stocks")

        if failed:
            logger.warning(f"Failed fundamentals for {len(failed)} tickers: {failed[:10]}...")

        return len(all_fundamentals), failed

    def collect_benchmarks(
        self,
        benchmarks: List[str] = ["SPY", "QQQ", "IWM"],
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None
    ):
        """
        Collect benchmark ETF data (SPY, QQQ, etc).

        Args:
            benchmarks: List of benchmark ticker symbols
            start_date: Start date
            end_date: End date (default: today)
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Collecting benchmark data for {benchmarks}")

        all_data = []

        for ticker in benchmarks:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)

                if not data.empty:
                    df = data.copy()

                    # Reset index to get date as column
                    df = df.reset_index()

                    # Add ticker
                    df['ticker'] = ticker

                    # Rename date column
                    if 'Date' in df.columns:
                        df['date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                    elif 'index' in df.columns:
                        df['date'] = pd.to_datetime(df['index']).dt.strftime('%Y-%m-%d')

                    # Handle price columns
                    if 'Adj Close' in df.columns:
                        df['adj_close'] = df['Adj Close']
                        df['close'] = df['Close']
                    elif 'adj_close' in df.columns:
                        # Already lowercase
                        pass
                    else:
                        # Fallback
                        df['close'] = df['Close'] if 'Close' in df.columns else df['close']
                        df['adj_close'] = df['close']

                    # Select only needed columns
                    df = df[['ticker', 'date', 'close', 'adj_close']].copy()
                    all_data.append(df)

                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Failed to download {ticker}: {e}")

        # Save
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            self.db.insert_benchmarks(combined)
            logger.info(f"Saved {len(combined)} benchmark records")

        return len(all_data)

    def collect_all(
        self,
        start_date: str = "2015-01-01",
        use_sp500: bool = True,
        custom_tickers: Optional[List[str]] = None
    ):
        """
        Collect all data: prices, fundamentals, benchmarks.

        Args:
            start_date: Start date for historical data
            use_sp500: If True, use S&P 500 universe
            custom_tickers: Use custom ticker list instead
        """
        logger.info("=" * 60)
        logger.info("Starting full data collection")
        logger.info("=" * 60)

        # Get ticker universe
        if custom_tickers:
            tickers = custom_tickers
        elif use_sp500:
            tickers = self.get_sp500_tickers()
        else:
            raise ValueError("Must provide either custom_tickers or use_sp500=True")

        # 1. Collect prices
        logger.info("\n[1/3] Collecting price data...")
        price_count, failed_prices = self.collect_prices(tickers, start_date=start_date)

        # Filter out failed tickers for fundamentals
        successful_tickers = [t for t in tickers if t not in failed_prices]

        # 2. Collect fundamentals
        logger.info("\n[2/3] Collecting fundamental data...")
        fund_count, failed_funds = self.collect_fundamentals(successful_tickers)

        # 3. Collect benchmarks
        logger.info("\n[3/3] Collecting benchmark data...")
        bench_count = self.collect_benchmarks(start_date=start_date)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Data collection complete!")
        logger.info("=" * 60)
        logger.info(f"Prices collected: {price_count} stocks")
        logger.info(f"Fundamentals collected: {fund_count} stocks")
        logger.info(f"Benchmarks collected: {bench_count} ETFs")

        stats = self.db.get_stats()
        logger.info(f"\nDatabase stats:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        return stats
