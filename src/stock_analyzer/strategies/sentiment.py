"""
News Sentiment Strategy

Uses FinBERT-based news sentiment analysis to score stocks.

Sentiment is a proven alpha factor - positive news tends to drive
short-term price momentum while negative news precedes underperformance.

Components:
- Overall sentiment score (-1 to +1)
- Positive/negative article ratios
- Sentiment trend (improving/deteriorating)
- Model confidence weighting
- Article volume (more articles = more reliable signal)

Academic References:
- Tetlock (2007): "Giving Content to Investor Sentiment"
- Garcia (2013): "Sentiment during Recessions"
- Heston & Sinha (2017): "News vs. Sentiment: Predicting Stock Returns"
"""

from decimal import Decimal
from typing import Optional
import pandas as pd

from .base import ScoringStrategy
from ..models.domain import Fundamentals, TechnicalIndicators, StockInfo
from ..sentiment.news_analyzer import NewsAnalyzer, NewsSentiment
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class SentimentStrategy(ScoringStrategy):
    """
    Sentiment-based scoring strategy.

    Rewards stocks with:
    - Positive news sentiment
    - Improving sentiment trend
    - High positive-to-negative ratio
    - Strong conviction (high confidence, many articles)
    """

    def __init__(self, weight: float = 1.0, days_back: int = 7):
        """
        Initialize sentiment strategy.

        Args:
            weight: Strategy weight in composite score
            days_back: Number of days of news to analyze
        """
        super().__init__(name="News Sentiment", weight=weight)
        self.days_back = days_back
        self.analyzer = NewsAnalyzer()

        logger.info("Initialized sentiment strategy")

    def calculate_score(
        self,
        price_data: pd.DataFrame,
        fundamentals: Optional[Fundamentals] = None,
        technical_indicators: Optional[TechnicalIndicators] = None,
        stock_info: Optional[StockInfo] = None,
        **kwargs
    ) -> Decimal:
        """
        Calculate sentiment-based score.

        Returns:
            Score from 0-100 based on news sentiment
        """
        # Get ticker from kwargs or stock_info
        ticker = kwargs.get('ticker')
        if not ticker and stock_info:
            ticker = stock_info.ticker

        if not ticker:
            logger.warning("No ticker provided for sentiment analysis")
            return Decimal("50.0")

        # Get company name for better news matching
        company_name = stock_info.company_name if stock_info else None

        # Analyze sentiment
        sentiment = self.analyzer.analyze_sentiment(
            ticker=ticker,
            company_name=company_name,
            days_back=self.days_back,
            use_cache=True
        )

        if not sentiment or sentiment.article_count == 0:
            logger.debug(f"No sentiment data for {ticker}")
            return Decimal("50.0")  # Neutral score

        try:
            score = 0.0

            # Component 1: Overall Sentiment Score (40 points)
            sentiment_score = self._score_sentiment(sentiment)
            score += sentiment_score * 0.40

            # Component 2: Positive/Negative Ratio (30 points)
            ratio_score = self._score_ratio(sentiment)
            score += ratio_score * 0.30

            # Component 3: Sentiment Trend (20 points)
            trend_score = self._score_trend(sentiment)
            score += trend_score * 0.20

            # Component 4: Conviction (10 points)
            conviction_score = self._score_conviction(sentiment)
            score += conviction_score * 0.10

            logger.debug(
                f"Sentiment components for {ticker} - "
                f"Score: {sentiment_score:.1f}, Ratio: {ratio_score:.1f}, "
                f"Trend: {trend_score:.1f}, Conviction: {conviction_score:.1f}"
            )

            return Decimal(str(min(100.0, max(0.0, score))))

        except Exception as e:
            logger.error(f"Error calculating sentiment score: {e}")
            return Decimal("50.0")

    def _score_sentiment(self, sentiment: NewsSentiment) -> float:
        """
        Score based on overall sentiment (-1 to +1).

        Mapping:
        +1.0: 100 points (extremely positive)
        +0.5: 75 points (positive)
        0.0: 50 points (neutral)
        -0.5: 25 points (negative)
        -1.0: 0 points (extremely negative)
        """
        # Linear mapping from [-1, 1] to [0, 100]
        score = (sentiment.sentiment_score + 1.0) * 50.0
        return max(0.0, min(100.0, score))

    def _score_ratio(self, sentiment: NewsSentiment) -> float:
        """
        Score based on positive/negative ratio.

        Higher positive ratio = higher score
        """
        # If no negative articles, check if we have positive
        if sentiment.negative_ratio == 0:
            if sentiment.positive_ratio > 0.5:
                return 100.0
            elif sentiment.positive_ratio > 0.3:
                return 80.0
            else:
                return 60.0

        # Calculate ratio
        ratio = sentiment.positive_ratio / (sentiment.negative_ratio + 0.001)

        # Score based on ratio
        if ratio >= 3.0:  # 3:1 positive
            return 100.0
        elif ratio >= 2.0:  # 2:1 positive
            return 90.0
        elif ratio >= 1.5:
            return 75.0
        elif ratio >= 1.0:
            return 60.0
        elif ratio >= 0.67:  # 2:3 (more negative)
            return 40.0
        elif ratio >= 0.5:
            return 25.0
        else:  # More than 2:1 negative
            return 10.0

    def _score_trend(self, sentiment: NewsSentiment) -> float:
        """
        Score based on sentiment trend.

        Improving sentiment = higher score
        """
        trend = sentiment.sentiment_trend.lower()

        if trend == "improving":
            return 100.0
        elif trend == "stable":
            # If stable and positive, that's good
            if sentiment.sentiment_score > 0.2:
                return 75.0
            # If stable and negative, that's bad
            elif sentiment.sentiment_score < -0.2:
                return 25.0
            # If stable and neutral
            else:
                return 50.0
        else:  # deteriorating
            return 20.0

    def _score_conviction(self, sentiment: NewsSentiment) -> float:
        """
        Score based on conviction (confidence × article count).

        More articles with high confidence = more reliable signal
        """
        # Article count contribution (0-50 points)
        if sentiment.article_count >= 20:
            article_score = 50.0
        elif sentiment.article_count >= 10:
            article_score = 40.0
        elif sentiment.article_count >= 5:
            article_score = 30.0
        else:
            article_score = sentiment.article_count * 5.0  # 5 points per article

        # Confidence contribution (0-50 points)
        confidence_score = sentiment.confidence * 50.0

        # Combined
        return article_score + confidence_score

    def get_sentiment_details(self, ticker: str) -> Optional[NewsSentiment]:
        """Get detailed sentiment analysis for a ticker."""
        return self.analyzer.get_cached_sentiment(ticker)
