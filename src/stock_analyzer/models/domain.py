"""
Domain models using Pydantic for type-safe data validation.

These models represent core business entities with proper validation,
following domain-driven design principles.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from decimal import Decimal

from .enums import (
    AssetClass,
    Exchange,
    Sector,
    SignalType,
    TrendDirection,
    AnalysisTimeframe,
)


class Quote(BaseModel):
    """Real-time or delayed quote for a security."""

    ticker: str = Field(..., description="Stock ticker symbol")
    price: Decimal = Field(..., description="Current price", gt=0)
    open: Optional[Decimal] = Field(None, description="Opening price")
    high: Optional[Decimal] = Field(None, description="Daily high")
    low: Optional[Decimal] = Field(None, description="Daily low")
    volume: int = Field(..., description="Trading volume", ge=0)
    previous_close: Optional[Decimal] = None
    change: Optional[Decimal] = None
    change_percent: Optional[Decimal] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        """Ensure ticker is uppercase and valid."""
        if not v:
            raise ValueError("Ticker cannot be empty")
        return v.upper().strip()

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat(),
        }
    )


class StockInfo(BaseModel):
    """Detailed stock information and metadata."""

    ticker: str
    company_name: Optional[str] = None
    exchange: Optional[Exchange] = None
    sector: Optional[Sector] = None
    industry: Optional[str] = None
    asset_class: AssetClass = AssetClass.STOCK
    market_cap: Optional[int] = None
    shares_outstanding: Optional[int] = None
    description: Optional[str] = None
    website: Optional[str] = None

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        """Ensure ticker is uppercase."""
        return v.upper().strip() if v else None


class Fundamentals(BaseModel):
    """Fundamental metrics for stock valuation."""

    # Valuation Metrics
    pe_ratio: Optional[Decimal] = Field(None, description="Price to Earnings")
    forward_pe: Optional[Decimal] = Field(None, description="Forward P/E")
    peg_ratio: Optional[Decimal] = Field(None, description="P/E to Growth")
    price_to_book: Optional[Decimal] = Field(None, description="Price to Book")
    price_to_sales: Optional[Decimal] = Field(None, description="Price to Sales")
    ev_to_ebitda: Optional[Decimal] = Field(None, description="EV to EBITDA")

    # Profitability Metrics (can be negative when companies are losing money)
    profit_margin: Optional[Decimal] = Field(None, description="Net profit margin")
    operating_margin: Optional[Decimal] = Field(None, description="Operating margin")
    gross_margin: Optional[Decimal] = Field(None, ge=0, le=1, description="Gross margin (rarely negative)")
    roe: Optional[Decimal] = Field(None, description="Return on Equity")
    roa: Optional[Decimal] = Field(None, description="Return on Assets")
    roic: Optional[Decimal] = Field(None, description="Return on Invested Capital")

    # Growth Metrics
    revenue_growth: Optional[Decimal] = None
    earnings_growth: Optional[Decimal] = None
    revenue_growth_yoy: Optional[Decimal] = None
    earnings_growth_yoy: Optional[Decimal] = None

    # Financial Health
    debt_to_equity: Optional[Decimal] = None
    current_ratio: Optional[Decimal] = None
    quick_ratio: Optional[Decimal] = None
    free_cash_flow: Optional[int] = None

    # Dividend Information
    dividend_yield: Optional[Decimal] = Field(None, ge=0)
    payout_ratio: Optional[Decimal] = Field(None, ge=0)  # Can be > 1.0 (unsustainable dividend)

    # Analyst Data
    analyst_rating: Optional[Decimal] = Field(None, ge=1, le=5)
    analyst_target_price: Optional[Decimal] = None
    analyst_count: Optional[int] = None

    # Risk Metrics
    beta: Optional[Decimal] = None


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators."""

    # Moving Averages
    sma_20: Optional[Decimal] = Field(None, description="20-day SMA")
    sma_50: Optional[Decimal] = Field(None, description="50-day SMA")
    sma_200: Optional[Decimal] = Field(None, description="200-day SMA")
    ema_12: Optional[Decimal] = Field(None, description="12-day EMA")
    ema_26: Optional[Decimal] = Field(None, description="26-day EMA")

    # Momentum Indicators
    rsi: Optional[Decimal] = Field(None, ge=0, le=100, description="RSI (14)")
    macd: Optional[Decimal] = Field(None, description="MACD line")
    macd_signal: Optional[Decimal] = Field(None, description="MACD signal line")
    macd_histogram: Optional[Decimal] = Field(None, description="MACD histogram")
    stochastic_k: Optional[Decimal] = Field(None, ge=0, le=100)
    stochastic_d: Optional[Decimal] = Field(None, ge=0, le=100)

    # Volatility Indicators
    atr: Optional[Decimal] = Field(None, description="Average True Range")
    bollinger_upper: Optional[Decimal] = None
    bollinger_middle: Optional[Decimal] = None
    bollinger_lower: Optional[Decimal] = None
    bollinger_width: Optional[Decimal] = None

    # Volume Indicators
    volume_sma_20: Optional[int] = None
    obv: Optional[int] = Field(None, description="On-Balance Volume")
    volume_ratio: Optional[Decimal] = Field(None, description="Current vs Avg Volume")

    # Trend Indicators
    adx: Optional[Decimal] = Field(None, ge=0, le=100, description="ADX trend strength")
    ichimoku_conversion: Optional[Decimal] = None
    ichimoku_base: Optional[Decimal] = None

    # Price Levels
    pivot_point: Optional[Decimal] = None
    resistance_1: Optional[Decimal] = None
    support_1: Optional[Decimal] = None
    week_52_high: Optional[Decimal] = None
    week_52_low: Optional[Decimal] = None
    distance_from_52w_high: Optional[Decimal] = Field(None, description="% from 52W high")
    distance_from_52w_low: Optional[Decimal] = Field(None, description="% from 52W low")


class MLPrediction(BaseModel):
    """Machine learning model predictions."""

    predicted_return_1d: Optional[Decimal] = Field(None, description="1-day return forecast")
    predicted_return_5d: Optional[Decimal] = Field(None, description="5-day return forecast")
    predicted_return_20d: Optional[Decimal] = Field(None, description="20-day return forecast")

    probability_up: Optional[Decimal] = Field(None, ge=0, le=1, description="Probability of upward movement")
    probability_down: Optional[Decimal] = Field(None, ge=0, le=1)

    confidence_score: Optional[Decimal] = Field(None, ge=0, le=1, description="Model confidence")
    feature_importance: Optional[Dict[str, float]] = Field(default_factory=dict)

    model_version: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FactorScores(BaseModel):
    """Multi-factor model scores (Fama-French style)."""

    value_score: Optional[Decimal] = Field(None, ge=0, le=100)
    growth_score: Optional[Decimal] = Field(None, ge=0, le=100)
    momentum_score: Optional[Decimal] = Field(None, ge=0, le=100)
    quality_score: Optional[Decimal] = Field(None, ge=0, le=100)
    size_score: Optional[Decimal] = Field(None, ge=0, le=100)
    volatility_score: Optional[Decimal] = Field(None, ge=0, le=100)

    composite_score: Decimal = Field(..., ge=0, le=100, description="Weighted composite")


class RiskMetrics(BaseModel):
    """Risk assessment metrics."""

    volatility_annual: Optional[Decimal] = Field(None, description="Annualized volatility")
    sharpe_ratio: Optional[Decimal] = Field(None, description="Risk-adjusted return")
    sortino_ratio: Optional[Decimal] = Field(None, description="Downside risk-adjusted")
    max_drawdown: Optional[Decimal] = Field(None, description="Maximum drawdown %")
    var_95: Optional[Decimal] = Field(None, description="Value at Risk (95%)")
    cvar_95: Optional[Decimal] = Field(None, description="Conditional VaR (95%)")

    beta_to_market: Optional[Decimal] = Field(None, description="Beta vs S&P 500")
    correlation_to_market: Optional[Decimal] = Field(None, ge=-1, le=1)


class Analysis(BaseModel):
    """
    Comprehensive stock analysis result.

    This is the main output model containing all analysis components.
    """

    # Identification
    ticker: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    timeframe: AnalysisTimeframe = AnalysisTimeframe.MEDIUM_TERM

    # Current State
    quote: Optional[Quote] = None
    stock_info: Optional[StockInfo] = None

    # Analysis Components
    fundamentals: Optional[Fundamentals] = None
    technical_indicators: Optional[TechnicalIndicators] = None
    ml_prediction: Optional[MLPrediction] = None
    factor_scores: Optional[FactorScores] = None
    risk_metrics: Optional[RiskMetrics] = None

    # Scoring & Signals
    composite_score: Decimal = Field(..., ge=0, le=100, description="Overall opportunity score")
    signal: SignalType = Field(..., description="Trading signal")
    trend: TrendDirection = Field(..., description="Overall trend")

    confidence: Decimal = Field(..., ge=0, le=1, description="Analysis confidence")

    # Insights
    key_strengths: List[str] = Field(default_factory=list)
    key_risks: List[str] = Field(default_factory=list)
    summary: Optional[str] = None

    # Metadata
    data_quality_score: Optional[Decimal] = Field(None, ge=0, le=1)
    providers_used: List[str] = Field(default_factory=list)

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        """Ensure ticker is uppercase."""
        return v.upper().strip()

    def model_dump(self, **kwargs):
        """Convert to dictionary for serialization."""
        return super().model_dump(exclude_none=True, **kwargs)

    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={
            Decimal: lambda v: float(v),
            datetime: lambda v: v.isoformat(),
        }
    )


class ScreenerResult(BaseModel):
    """Results from stock screening."""

    total_analyzed: int
    results: List[Analysis]
    filters_applied: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    execution_time_seconds: Optional[float] = None

    def top_n(self, n: int = 10) -> List[Analysis]:
        """Get top N stocks by composite score."""
        return sorted(
            self.results,
            key=lambda x: x.composite_score,
            reverse=True
        )[:n]

    def filter_by_signal(self, signal: SignalType) -> List[Analysis]:
        """Filter results by signal type."""
        return [r for r in self.results if r.signal == signal]


class MarketOverview(BaseModel):
    """Overall market statistics and sentiment."""

    total_stocks_analyzed: int
    avg_score: Decimal
    avg_rsi: Optional[Decimal] = None
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0

    top_sectors: List[str] = Field(default_factory=list)
    market_sentiment: str = "neutral"

    timestamp: datetime = Field(default_factory=datetime.utcnow)
