"""
Core stock analysis service.

Orchestrates data fetching, indicator calculation, and scoring.
"""

import asyncio
from typing import List, Optional
from decimal import Decimal
from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta
import numpy as np

from ..data.provider_manager import ProviderManager
from ..models.domain import (
    Analysis,
    Quote,
    Fundamentals,
    TechnicalIndicators,
    StockInfo,
    FactorScores,
    RiskMetrics,
)
from ..models.enums import SignalType, TrendDirection, AnalysisTimeframe
from ..strategies import ScoringStrategy, MomentumStrategy, ValueStrategy, GrowthStrategy
from ..utils.logger import setup_logger
from ..utils.decorators import timing
from ..config import get_settings

logger = setup_logger(__name__)


class StockAnalyzer:
    """
    Comprehensive stock analysis service.

    Integrates data providers, technical analysis, fundamental analysis,
    and multiple scoring strategies to produce actionable insights.
    """

    def __init__(
        self,
        provider_manager: Optional[ProviderManager] = None,
        strategies: Optional[List[ScoringStrategy]] = None
    ):
        """
        Initialize analyzer.

        Args:
            provider_manager: Data provider manager (creates default if None)
            strategies: List of scoring strategies (creates default if None)
        """
        self.settings = get_settings()
        self.provider_manager = provider_manager or ProviderManager()
        self.strategies = strategies or self._create_default_strategies()

        logger.info(f"Initialized analyzer with {len(self.strategies)} strategies")

    def _create_default_strategies(self) -> List[ScoringStrategy]:
        """Create default scoring strategies."""
        return [
            MomentumStrategy(weight=1.0),
            ValueStrategy(weight=1.0),
            GrowthStrategy(weight=1.0),
        ]

    @timing
    async def analyze(
        self,
        ticker: str,
        timeframe: AnalysisTimeframe = AnalysisTimeframe.MEDIUM_TERM
    ) -> Analysis:
        """
        Perform comprehensive analysis of a stock.

        Args:
            ticker: Stock ticker symbol
            timeframe: Analysis timeframe

        Returns:
            Complete Analysis object with all metrics

        Example:
            >>> analyzer = StockAnalyzer()
            >>> analysis = await analyzer.analyze("AAPL")
            >>> print(f"Score: {analysis.composite_score}/100")
        """
        logger.info(f"Analyzing {ticker}")

        try:
            # Fetch all data concurrently
            quote, stock_info, fundamentals, price_data = await asyncio.gather(
                self.provider_manager.fetch_quote(ticker),
                self.provider_manager.fetch_stock_info(ticker),
                self.provider_manager.fetch_fundamentals(ticker),
                self.provider_manager.fetch_historical_data(ticker),
                return_exceptions=True
            )

            # Handle fetch failures gracefully
            if isinstance(quote, Exception):
                logger.error(f"Failed to fetch quote for {ticker}: {quote}")
                quote = None
            if isinstance(stock_info, Exception):
                stock_info = None
            if isinstance(fundamentals, Exception):
                fundamentals = None
            if isinstance(price_data, Exception) or price_data.empty:
                raise ValueError(f"No price data available for {ticker}")

            # Calculate technical indicators
            technical_indicators = self._calculate_technical_indicators(price_data)

            # Calculate factor scores
            factor_scores = self._calculate_factor_scores(
                price_data, fundamentals, technical_indicators, stock_info
            )

            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(price_data)

            # Generate insights
            strengths, risks = self._generate_insights(
                fundamentals, technical_indicators, factor_scores
            )

            # Determine signal and trend
            signal = self._determine_signal(factor_scores.composite_score)
            trend = self._determine_trend(technical_indicators, price_data)

            # Calculate confidence
            confidence = self._calculate_confidence(
                quote, fundamentals, technical_indicators, price_data
            )

            # Create analysis object
            analysis = Analysis(
                ticker=ticker,
                timestamp=datetime.utcnow(),
                timeframe=timeframe,
                quote=quote,
                stock_info=stock_info,
                fundamentals=fundamentals,
                technical_indicators=technical_indicators,
                factor_scores=factor_scores,
                risk_metrics=risk_metrics,
                composite_score=factor_scores.composite_score,
                signal=signal,
                trend=trend,
                confidence=confidence,
                key_strengths=strengths,
                key_risks=risks,
                providers_used=[self.provider_manager._providers[0].name],
            )

            logger.info(
                f"Analysis complete for {ticker}: "
                f"Score={analysis.composite_score:.1f}, Signal={signal.value}"
            )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            raise

    @timing
    async def analyze_batch(
        self,
        tickers: List[str],
        max_concurrent: int = 50
    ) -> List[Analysis]:
        """
        Analyze multiple stocks concurrently.

        Args:
            tickers: List of ticker symbols
            max_concurrent: Maximum concurrent analyses

        Returns:
            List of Analysis objects
        """
        logger.info(f"Batch analyzing {len(tickers)} stocks")

        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_limit(ticker: str):
            async with semaphore:
                try:
                    return await self.analyze(ticker)
                except Exception as e:
                    logger.warning(f"Failed to analyze {ticker}: {e}")
                    return None

        results = await asyncio.gather(
            *[analyze_with_limit(t) for t in tickers]
        )

        # Filter out None results
        analyses = [r for r in results if r is not None]

        logger.info(f"Successfully analyzed {len(analyses)}/{len(tickers)} stocks")
        return analyses

    def _calculate_technical_indicators(
        self,
        price_data: pd.DataFrame
    ) -> TechnicalIndicators:
        """
        Calculate all technical indicators.

        Args:
            price_data: Historical OHLCV data

        Returns:
            TechnicalIndicators object
        """
        try:
            close = price_data['Close']

            # Moving averages
            sma_20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
            sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
            sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

            # RSI
            rsi = ta.rsi(close, length=14).iloc[-1] if len(close) >= 14 else None

            # MACD
            macd_data = ta.macd(close)
            macd = None
            macd_signal = None
            macd_histogram = None
            if macd_data is not None and not macd_data.empty:
                macd = macd_data['MACD_12_26_9'].iloc[-1] if 'MACD_12_26_9' in macd_data else None
                macd_signal = macd_data['MACDs_12_26_9'].iloc[-1] if 'MACDs_12_26_9' in macd_data else None
                macd_histogram = macd_data['MACDh_12_26_9'].iloc[-1] if 'MACDh_12_26_9' in macd_data else None

            # ATR
            atr = ta.atr(price_data['High'], price_data['Low'], close).iloc[-1] if len(close) >= 14 else None

            # ADX
            adx_data = ta.adx(price_data['High'], price_data['Low'], close)
            adx = None
            if adx_data is not None and not adx_data.empty:
                adx = adx_data[f'ADX_14'].iloc[-1] if f'ADX_14' in adx_data else None

            # 52-week high/low
            week_52_high = price_data['High'].rolling(252).max().iloc[-1] if len(close) >= 252 else price_data['High'].max()
            week_52_low = price_data['Low'].rolling(252).min().iloc[-1] if len(close) >= 252 else price_data['Low'].min()
            current_price = close.iloc[-1]

            distance_from_52w_high = ((current_price - week_52_high) / week_52_high * 100) if week_52_high else None
            distance_from_52w_low = ((current_price - week_52_low) / week_52_low * 100) if week_52_low else None

            # Volume
            volume_sma_20 = int(price_data['Volume'].rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
            volume_ratio = price_data['Volume'].iloc[-1] / volume_sma_20 if volume_sma_20 and volume_sma_20 > 0 else None

            return TechnicalIndicators(
                sma_20=Decimal(str(sma_20)) if sma_20 and not pd.isna(sma_20) else None,
                sma_50=Decimal(str(sma_50)) if sma_50 and not pd.isna(sma_50) else None,
                sma_200=Decimal(str(sma_200)) if sma_200 and not pd.isna(sma_200) else None,
                rsi=Decimal(str(rsi)) if rsi and not pd.isna(rsi) else None,
                macd=Decimal(str(macd)) if macd and not pd.isna(macd) else None,
                macd_signal=Decimal(str(macd_signal)) if macd_signal and not pd.isna(macd_signal) else None,
                macd_histogram=Decimal(str(macd_histogram)) if macd_histogram and not pd.isna(macd_histogram) else None,
                atr=Decimal(str(atr)) if atr and not pd.isna(atr) else None,
                adx=Decimal(str(adx)) if adx and not pd.isna(adx) else None,
                volume_sma_20=volume_sma_20,
                volume_ratio=Decimal(str(volume_ratio)) if volume_ratio and not pd.isna(volume_ratio) else None,
                week_52_high=Decimal(str(week_52_high)) if week_52_high and not pd.isna(week_52_high) else None,
                week_52_low=Decimal(str(week_52_low)) if week_52_low and not pd.isna(week_52_low) else None,
                distance_from_52w_high=Decimal(str(distance_from_52w_high)) if distance_from_52w_high and not pd.isna(distance_from_52w_high) else None,
                distance_from_52w_low=Decimal(str(distance_from_52w_low)) if distance_from_52w_low and not pd.isna(distance_from_52w_low) else None,
            )

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return TechnicalIndicators()

    def _calculate_factor_scores(
        self,
        price_data: pd.DataFrame,
        fundamentals: Optional[Fundamentals],
        technical_indicators: Optional[TechnicalIndicators],
        stock_info: Optional[StockInfo],
    ) -> FactorScores:
        """
        Calculate multi-factor scores using all strategies.

        Returns:
            FactorScores with individual and composite scores
        """
        try:
            scores = {}

            # Calculate score from each strategy
            for strategy in self.strategies:
                score = strategy.calculate_score(
                    price_data, fundamentals, technical_indicators, stock_info
                )
                scores[strategy.name] = float(score) * strategy.weight

            # Calculate weighted composite
            total_weight = sum(s.weight for s in self.strategies)
            composite = sum(scores.values()) / total_weight if total_weight > 0 else 0

            # Map strategies to factor scores
            momentum_score = scores.get("Momentum Strategy", 0)
            value_score = scores.get("Value Strategy", 0)
            growth_score = scores.get("Growth Strategy", 0)

            return FactorScores(
                momentum_score=Decimal(str(momentum_score)),
                value_score=Decimal(str(value_score)),
                growth_score=Decimal(str(growth_score)),
                quality_score=Decimal("50"),  # Placeholder
                size_score=Decimal("50"),  # Placeholder
                volatility_score=Decimal("50"),  # Placeholder
                composite_score=Decimal(str(composite)),
            )

        except Exception as e:
            logger.error(f"Error calculating factor scores: {e}")
            return FactorScores(composite_score=Decimal("0"))

    def _calculate_risk_metrics(self, price_data: pd.DataFrame) -> Optional[RiskMetrics]:
        """Calculate risk metrics from price data."""
        try:
            returns = price_data['Close'].pct_change().dropna()

            if len(returns) < 30:
                return None

            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252) * 100

            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            excess_returns = returns - (risk_free_rate / 252)
            sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std()) if excess_returns.std() > 0 else 0

            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100

            return RiskMetrics(
                volatility_annual=Decimal(str(volatility)),
                sharpe_ratio=Decimal(str(sharpe)),
                max_drawdown=Decimal(str(max_drawdown)),
            )

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return None

    def _generate_insights(
        self,
        fundamentals: Optional[Fundamentals],
        technical_indicators: Optional[TechnicalIndicators],
        factor_scores: FactorScores,
    ) -> tuple[List[str], List[str]]:
        """Generate key strengths and risks."""
        strengths = []
        risks = []

        # Fundamentals-based insights
        if fundamentals:
            if fundamentals.roe and float(fundamentals.roe) > 0.15:
                strengths.append(f"Strong ROE of {float(fundamentals.roe)*100:.1f}%")
            if fundamentals.pe_ratio and float(fundamentals.pe_ratio) < 15:
                strengths.append(f"Attractive P/E of {float(fundamentals.pe_ratio):.1f}")
            if fundamentals.debt_to_equity and float(fundamentals.debt_to_equity) > 1.5:
                risks.append(f"High debt/equity ratio: {float(fundamentals.debt_to_equity):.2f}")

        # Technical-based insights
        if technical_indicators:
            if technical_indicators.rsi and float(technical_indicators.rsi) < 30:
                strengths.append("Oversold condition (RSI < 30)")
            elif technical_indicators.rsi and float(technical_indicators.rsi) > 70:
                risks.append("Overbought condition (RSI > 70)")

        # Factor-based insights
        if float(factor_scores.momentum_score) > 70:
            strengths.append("Strong price momentum")
        if float(factor_scores.value_score) > 70:
            strengths.append("Attractive valuation")

        return strengths[:5], risks[:5]  # Limit to 5 each

    def _determine_signal(self, composite_score: Decimal) -> SignalType:
        """Determine trading signal from composite score."""
        score = float(composite_score)

        if score >= 80:
            return SignalType.STRONG_BUY
        elif score >= 60:
            return SignalType.BUY
        elif score >= 40:
            return SignalType.HOLD
        elif score >= 20:
            return SignalType.SELL
        else:
            return SignalType.STRONG_SELL

    def _determine_trend(
        self,
        technical_indicators: Optional[TechnicalIndicators],
        price_data: pd.DataFrame
    ) -> TrendDirection:
        """Determine overall trend."""
        if not technical_indicators:
            return TrendDirection.NEUTRAL

        current_price = float(price_data['Close'].iloc[-1])

        # Check if above key moving averages
        bullish_count = 0
        total_count = 0

        if technical_indicators.sma_20:
            total_count += 1
            if current_price > float(technical_indicators.sma_20):
                bullish_count += 1

        if technical_indicators.sma_50:
            total_count += 1
            if current_price > float(technical_indicators.sma_50):
                bullish_count += 1

        if total_count == 0:
            return TrendDirection.NEUTRAL

        bullish_pct = bullish_count / total_count

        if bullish_pct >= 0.66:
            return TrendDirection.BULLISH
        elif bullish_pct <= 0.33:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL

    def _calculate_confidence(
        self,
        quote: Optional[Quote],
        fundamentals: Optional[Fundamentals],
        technical_indicators: Optional[TechnicalIndicators],
        price_data: pd.DataFrame,
    ) -> Decimal:
        """Calculate confidence score based on data quality."""
        confidence = 0.0
        max_confidence = 4.0

        if quote:
            confidence += 1.0
        if fundamentals:
            confidence += 1.0
        if technical_indicators:
            confidence += 1.0
        if len(price_data) >= 60:
            confidence += 1.0

        return Decimal(str(confidence / max_confidence))
