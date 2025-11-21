"""
Database abstraction layer for historical stock data.

Supports SQLite (local, fast) and Supabase (cloud, deployed).
Switch between them with a single parameter.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class Database:
    """
    Database abstraction supporting SQLite and Supabase.

    For development/backtesting: use_supabase=False (fast, local)
    For production/deployment: use_supabase=True (cloud, accessible)
    """

    def __init__(self, db_path: str = "data/stocks.db", use_supabase: bool = False):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file (if not using Supabase)
            use_supabase: If True, use Supabase; if False, use SQLite
        """
        self.use_supabase = use_supabase
        self.db_path = Path(db_path)

        if use_supabase:
            # TODO: Add Supabase support when deploying
            raise NotImplementedError("Supabase support coming when we deploy")
        else:
            # SQLite setup
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._init_schema()
            logger.info(f"SQLite database initialized at {self.db_path}")

    def _init_schema(self):
        """Create database tables if they don't exist."""
        schema = """
        -- Daily price data
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL NOT NULL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        );

        -- Fundamental data (quarterly snapshots)
        CREATE TABLE IF NOT EXISTS fundamentals (
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            pe_ratio REAL,
            pb_ratio REAL,
            ps_ratio REAL,
            roe REAL,
            roa REAL,
            debt_to_equity REAL,
            profit_margin REAL,
            revenue_growth REAL,
            earnings_growth REAL,
            dividend_yield REAL,
            PRIMARY KEY (ticker, date)
        );

        -- Stock metadata (sector, industry, market cap)
        CREATE TABLE IF NOT EXISTS meta (
            ticker TEXT PRIMARY KEY,
            company_name TEXT,
            sector TEXT,
            industry TEXT,
            market_cap REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Benchmark data (SPY, sector ETFs)
        CREATE TABLE IF NOT EXISTS benchmarks (
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            close REAL NOT NULL,
            adj_close REAL NOT NULL,
            PRIMARY KEY (ticker, date)
        );

        -- Engineered features (for ML)
        CREATE TABLE IF NOT EXISTS features (
            ticker TEXT NOT NULL,
            date DATE NOT NULL,
            -- Momentum features (absolute)
            return_1m REAL,
            return_3m REAL,
            return_6m REAL,
            return_12m REAL,
            volatility_20d REAL,
            volatility_60d REAL,
            dist_from_sma_50 REAL,
            dist_from_sma_200 REAL,
            -- 52-week high/low
            dist_from_52w_high REAL,
            dist_from_52w_low REAL,
            -- Cross-sectional ranking features (0-1 percentile vs peers)
            return_1m_rank REAL,
            return_3m_rank REAL,
            return_6m_rank REAL,
            return_12m_rank REAL,
            volatility_20d_rank REAL,
            volatility_60d_rank REAL,
            dist_from_sma_50_rank REAL,
            dist_from_sma_200_rank REAL,
            dist_from_52w_high_rank REAL,
            dist_from_52w_low_rank REAL,
            -- Market regime
            market_volatility REAL,
            market_trend REAL,
            -- Target (label)
            target_1m_excess REAL,
            target_binary INTEGER,
            PRIMARY KEY (ticker, date)
        );

        -- Model predictions (track performance)
        CREATE TABLE IF NOT EXISTS predictions (
            ticker TEXT NOT NULL,
            prediction_date DATE NOT NULL,
            target_date DATE NOT NULL,
            probability REAL NOT NULL,
            predicted_return REAL,
            actual_return REAL,
            was_correct INTEGER,
            model_version TEXT,
            PRIMARY KEY (ticker, prediction_date)
        );

        -- Create indexes for fast queries
        CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);
        CREATE INDEX IF NOT EXISTS idx_prices_ticker ON prices(ticker);
        CREATE INDEX IF NOT EXISTS idx_features_date ON features(date);
        CREATE INDEX IF NOT EXISTS idx_benchmarks_date ON benchmarks(date);
        """

        self.conn.executescript(schema)
        self.conn.commit()
        logger.info("Database schema initialized")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        if self.use_supabase:
            # TODO: Return Supabase client
            raise NotImplementedError()
        else:
            yield self.conn

    # ==================== WRITE OPERATIONS ====================

    def insert_prices(self, df: pd.DataFrame):
        """
        Insert price data (skips duplicates).

        Args:
            df: DataFrame with columns [ticker, date, open, high, low, close, adj_close, volume]
        """
        # Insert using INSERT OR IGNORE to skip duplicates
        cursor = self.conn.cursor()
        sql = """
            INSERT OR IGNORE INTO prices (ticker, date, open, high, low, close, adj_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        records = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']].values.tolist()
        cursor.executemany(sql, records)
        self.conn.commit()

        logger.info(f"Inserted {len(df)} price records (duplicates skipped)")

    def insert_fundamentals(self, df: pd.DataFrame):
        """Insert fundamental data (skips duplicates)."""
        cursor = self.conn.cursor()
        sql = """
            INSERT OR REPLACE INTO fundamentals
            (ticker, date, pe_ratio, pb_ratio, ps_ratio, roe, roa, debt_to_equity,
             profit_margin, revenue_growth, earnings_growth, dividend_yield)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        records = df[['ticker', 'date', 'pe_ratio', 'pb_ratio', 'ps_ratio', 'roe', 'roa',
                      'debt_to_equity', 'profit_margin', 'revenue_growth', 'earnings_growth',
                      'dividend_yield']].values.tolist()
        cursor.executemany(sql, records)
        self.conn.commit()
        logger.info(f"Inserted {len(df)} fundamental records")

    def insert_meta(self, df: pd.DataFrame):
        """Insert stock metadata (replaces existing)."""
        df.to_sql('meta', self.conn, if_exists='replace', index=False)
        logger.info(f"Inserted {len(df)} meta records")

    def insert_benchmarks(self, df: pd.DataFrame):
        """Insert benchmark data (skips duplicates)."""
        cursor = self.conn.cursor()
        sql = """
            INSERT OR IGNORE INTO benchmarks (ticker, date, close, adj_close)
            VALUES (?, ?, ?, ?)
        """
        records = df[['ticker', 'date', 'close', 'adj_close']].values.tolist()
        cursor.executemany(sql, records)
        self.conn.commit()
        logger.info(f"Inserted {len(df)} benchmark records")

    def insert_features(self, df: pd.DataFrame):
        """Insert engineered features (replaces existing table)."""
        # Use 'replace' to auto-create table with correct schema from DataFrame
        df.to_sql('features', self.conn, if_exists='replace', index=False)
        logger.info(f"Inserted {len(df)} feature records")

    def insert_predictions(self, df: pd.DataFrame):
        """Log model predictions for tracking."""
        df.to_sql('predictions', self.conn, if_exists='append', index=False)
        logger.info(f"Inserted {len(df)} prediction records")

    # ==================== READ OPERATIONS ====================

    def get_prices(self, ticker: str = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get price data.

        Args:
            ticker: Filter by ticker (optional)
            start_date: Start date YYYY-MM-DD (optional)
            end_date: End date YYYY-MM-DD (optional)
        """
        query = "SELECT * FROM prices WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY ticker, date"

        return pd.read_sql(query, self.conn, params=params)

    def get_fundamentals(self, ticker: str = None, start_date: str = None) -> pd.DataFrame:
        """Get fundamental data."""
        query = "SELECT * FROM fundamentals WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        query += " ORDER BY ticker, date"

        return pd.read_sql(query, self.conn, params=params)

    def get_meta(self, ticker: str = None, sector: str = None) -> pd.DataFrame:
        """Get stock metadata."""
        query = "SELECT * FROM meta WHERE 1=1"
        params = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if sector:
            query += " AND sector = ?"
            params.append(sector)

        return pd.read_sql(query, self.conn, params=params)

    def get_benchmarks(self, ticker: str = "SPY", start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get benchmark data (SPY by default)."""
        query = "SELECT * FROM benchmarks WHERE ticker = ?"
        params = [ticker]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        return pd.read_sql(query, self.conn, params=params)

    def get_features(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get all engineered features for training."""
        query = "SELECT * FROM features WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date, ticker"

        return pd.read_sql(query, self.conn, params=params)

    def get_universe(self, date: str = None) -> List[str]:
        """
        Get list of tickers in universe.

        Args:
            date: Filter stocks that have price data on this date
        """
        if date:
            query = "SELECT DISTINCT ticker FROM prices WHERE date = ? ORDER BY ticker"
            params = [date]
        else:
            query = "SELECT DISTINCT ticker FROM meta ORDER BY ticker"
            params = []

        df = pd.read_sql(query, self.conn, params=params)
        return df['ticker'].tolist()

    # ==================== UTILITY ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}

        with self.get_connection() as conn:
            # Count records in each table
            for table in ['prices', 'fundamentals', 'meta', 'benchmarks', 'features']:
                count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn).iloc[0]['count']
                stats[f'{table}_count'] = count

            # Date ranges
            prices_range = pd.read_sql(
                "SELECT MIN(date) as min_date, MAX(date) as max_date FROM prices",
                conn
            ).iloc[0]
            stats['price_data_start'] = prices_range['min_date']
            stats['price_data_end'] = prices_range['max_date']

            # Unique tickers
            stats['unique_tickers'] = pd.read_sql(
                "SELECT COUNT(DISTINCT ticker) as count FROM prices",
                conn
            ).iloc[0]['count']

        return stats

    def clear_all(self):
        """DANGER: Clear all data from database."""
        tables = ['prices', 'fundamentals', 'meta', 'benchmarks', 'features', 'predictions']
        for table in tables:
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.commit()
        logger.warning("All data cleared from database")

    def close(self):
        """Close database connection."""
        if not self.use_supabase:
            self.conn.close()
            logger.info("Database connection closed")
