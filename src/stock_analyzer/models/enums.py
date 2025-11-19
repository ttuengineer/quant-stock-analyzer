"""Enumerations for domain models."""

from enum import Enum


class AssetClass(str, Enum):
    """Asset class types."""

    STOCK = "stock"
    ETF = "etf"
    INDEX = "index"
    CRYPTO = "crypto"


class Exchange(str, Enum):
    """Stock exchanges."""

    NASDAQ = "NASDAQ"
    NYSE = "NYSE"
    AMEX = "AMEX"
    OTHER = "OTHER"


class Sector(str, Enum):
    """Market sectors (GICS Level 1 + Yahoo Finance variants)."""

    TECHNOLOGY = "Technology"
    HEALTHCARE = "Healthcare"
    FINANCIALS = "Financials"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    CONSUMER_CYCLICAL = "Consumer Cyclical"  # Yahoo Finance variant
    CONSUMER_DEFENSIVE = "Consumer Defensive"  # Yahoo Finance variant
    INDUSTRIALS = "Industrials"
    ENERGY = "Energy"
    UTILITIES = "Utilities"
    REAL_ESTATE = "Real Estate"
    MATERIALS = "Materials"
    COMMUNICATION_SERVICES = "Communication Services"
    BASIC_MATERIALS = "Basic Materials"  # Yahoo Finance variant
    FINANCIAL_SERVICES = "Financial Services"  # Yahoo Finance variant


class SignalType(str, Enum):
    """Trading signal types."""

    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class TrendDirection(str, Enum):
    """Price trend direction."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"


class AnalysisTimeframe(str, Enum):
    """Analysis timeframe."""

    INTRADAY = "intraday"
    SHORT_TERM = "short_term"  # Days to weeks
    MEDIUM_TERM = "medium_term"  # Weeks to months
    LONG_TERM = "long_term"  # Months to years
