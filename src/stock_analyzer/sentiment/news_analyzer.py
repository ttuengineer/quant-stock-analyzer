"""
News Sentiment Analyzer

Supports multiple sentiment data providers:
1. Alpha Vantage (Preferred): Pre-calculated sentiment scores with ticker relevance
2. NewsAPI + FinBERT: Fetch news and analyze with FinBERT model

Features:
- Real-time news fetching with sentiment scores
- Ticker-specific sentiment filtering
- Sentiment aggregation with recency weighting
- Historical sentiment tracking
- Automatic fallback between providers

Academic References:
- Araci (2019): "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"
- Devlin et al. (2018): "BERT: Pre-training of Deep Bidirectional Transformers"
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import requests

from ..utils.logger import setup_logger
from ..config import get_settings

# Optional imports for NewsAPI provider (not needed for Alpha Vantage)
try:
    from newsapi import NewsApiClient
    from transformers import pipeline
    import torch
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False

logger = setup_logger(__name__)


@dataclass
class NewsSentiment:
    """News sentiment analysis result."""
    ticker: str
    sentiment_score: float  # -1.0 to +1.0
    positive_ratio: float  # 0.0 to 1.0
    negative_ratio: float  # 0.0 to 1.0
    neutral_ratio: float  # 0.0 to 1.0
    article_count: int
    latest_headline: Optional[str]
    sentiment_trend: str  # "improving", "deteriorating", "stable"
    confidence: float  # 0.0 to 1.0
    timestamp: datetime


class NewsAnalyzer:
    """
    Analyze news sentiment using Alpha Vantage or NewsAPI+FinBERT.

    Prefers Alpha Vantage (pre-calculated scores) if available,
    falls back to NewsAPI+FinBERT if configured.
    """

    def __init__(
        self,
        alpha_vantage_key: Optional[str] = None,
        newsapi_key: Optional[str] = None,
        cache_hours: int = 1
    ):
        """
        Initialize news sentiment analyzer.

        Args:
            alpha_vantage_key: Alpha Vantage API key (preferred)
            newsapi_key: NewsAPI key (fallback)
            cache_hours: Cache sentiment results for N hours
        """
        settings = get_settings()
        self.alpha_vantage_key = alpha_vantage_key or getattr(settings, "alpha_vantage_api_key", None)
        self.newsapi_key = newsapi_key or getattr(settings, "newsapi_key", None)
        self.cache_hours = cache_hours

        # Determine provider
        if self.alpha_vantage_key:
            self.provider = "alpha_vantage"
            logger.info("Using Alpha Vantage for news sentiment (pre-calculated scores)")
        elif self.newsapi_key:
            self.provider = "newsapi"
            logger.info("Using NewsAPI + FinBERT for news sentiment")
            self._init_finbert()
        else:
            self.provider = None
            logger.warning("No sentiment provider configured (Alpha Vantage or NewsAPI key needed)")

        # Sentiment cache
        self.cache: Dict[str, NewsSentiment] = {}

    def _init_finbert(self):
        """Initialize FinBERT sentiment model (only for NewsAPI provider)."""
        if not NEWSAPI_AVAILABLE:
            logger.warning("NewsAPI/FinBERT dependencies not installed")
            logger.warning("Falling back to Alpha Vantage if available")
            self.provider = "alpha_vantage" if self.alpha_vantage_key else None
            self.sentiment_pipeline = None
            return

        try:
            logger.info("Loading FinBERT model for sentiment analysis...")

            # Initialize NewsAPI client
            self.news_client = NewsApiClient(api_key=self.newsapi_key)

            # Check if CUDA is available
            device = 0 if torch.cuda.is_available() else -1

            # Load sentiment pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=device,
                max_length=512,
                truncation=True
            )

            logger.info(f"FinBERT loaded successfully (device: {'GPU' if device == 0 else 'CPU'})")

        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            logger.warning("Falling back to Alpha Vantage if available")
            self.provider = "alpha_vantage" if self.alpha_vantage_key else None
            self.sentiment_pipeline = None

    def analyze_sentiment(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        days_back: int = 7,
        use_cache: bool = True
    ) -> Optional[NewsSentiment]:
        """
        Analyze news sentiment for a stock.

        Args:
            ticker: Stock ticker symbol
            company_name: Company name for better news matching
            days_back: Number of days of news to analyze
            use_cache: Use cached results if available

        Returns:
            NewsSentiment object or None if analysis fails
        """
        if not self.provider:
            logger.warning("News sentiment not available (no API key configured)")
            return None

        # Check cache
        if use_cache and ticker in self.cache:
            cached = self.cache[ticker]
            age = (datetime.now() - cached.timestamp).total_seconds() / 3600
            if age < self.cache_hours:
                logger.debug(f"Using cached sentiment for {ticker} ({age:.1f}h old)")
                return cached

        try:
            # Route to appropriate provider
            if self.provider == "alpha_vantage":
                result = self._analyze_alpha_vantage(ticker)
            elif self.provider == "newsapi":
                result = self._analyze_newsapi(ticker, company_name, days_back)
            else:
                return self._default_sentiment(ticker)

            if result:
                # Cache result
                self.cache[ticker] = result

                logger.info(
                    f"Sentiment for {ticker}: {result.sentiment_score:.2f} "
                    f"({result.article_count} articles, {result.sentiment_trend})"
                )

            return result

        except Exception as e:
            logger.error(f"Error analyzing sentiment for {ticker}: {e}")
            return self._default_sentiment(ticker)

    def _analyze_alpha_vantage(self, ticker: str) -> Optional[NewsSentiment]:
        """Analyze sentiment using Alpha Vantage NEWS_SENTIMENT API."""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "limit": 50,
                "apikey": self.alpha_vantage_key
            }

            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if "feed" not in data or not data["feed"]:
                logger.warning(f"No news found for {ticker} from Alpha Vantage")
                return self._default_sentiment(ticker)

            # Extract sentiment data
            sentiments = []
            for article in data["feed"]:
                # Get ticker-specific sentiment
                ticker_sentiment = None
                if "ticker_sentiment" in article:
                    for ts in article["ticker_sentiment"]:
                        if ts.get("ticker") == ticker:
                            ticker_sentiment = ts
                            break

                if ticker_sentiment:
                    sentiment_score = float(ticker_sentiment.get("ticker_sentiment_score", 0))
                    relevance_score = float(ticker_sentiment.get("relevance_score", 0))

                    # Map sentiment score (-1 to +1) to label
                    if sentiment_score > 0.15:
                        label = "positive"
                    elif sentiment_score < -0.15:
                        label = "negative"
                    else:
                        label = "neutral"

                    sentiments.append({
                        'sentiment': sentiment_score,
                        'label': label,
                        'confidence': abs(sentiment_score),  # Use magnitude as confidence
                        'relevance': relevance_score,
                        'published_at': article.get("time_published"),
                        'title': article.get("title", "")
                    })

            if not sentiments:
                logger.warning(f"No ticker-specific sentiment for {ticker}")
                return self._default_sentiment(ticker)

            # Aggregate sentiments with relevance weighting
            result = self._aggregate_alpha_vantage_sentiments(ticker, sentiments)
            return result

        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage sentiment: {e}")
            return None

    def _analyze_newsapi(self, ticker: str, company_name: Optional[str], days_back: int) -> Optional[NewsSentiment]:
        """Analyze sentiment using NewsAPI + FinBERT."""
        try:
            # Fetch news articles
            articles = self._fetch_news(ticker, company_name, days_back)

            if not articles:
                logger.warning(f"No news articles found for {ticker}")
                return self._default_sentiment(ticker)

            # Analyze sentiment for each article
            sentiments = []
            for article in articles:
                sentiment = self._analyze_text(article['title'], article.get('description'))
                if sentiment:
                    sentiments.append({
                        **sentiment,
                        'published_at': article['publishedAt'],
                        'title': article['title']
                    })

            if not sentiments:
                logger.warning(f"No sentiment results for {ticker}")
                return self._default_sentiment(ticker)

            # Aggregate sentiments
            result = self._aggregate_sentiments(ticker, sentiments)
            return result

        except Exception as e:
            logger.error(f"Error analyzing NewsAPI sentiment: {e}")
            return None

    def _fetch_news(
        self,
        ticker: str,
        company_name: Optional[str],
        days_back: int
    ) -> List[Dict]:
        """Fetch news articles from NewsAPI."""
        try:
            # Build search query
            query_terms = [ticker]
            if company_name:
                query_terms.append(company_name)

            query = " OR ".join(query_terms)

            # Fetch from NewsAPI
            from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

            response = self.news_client.get_everything(
                q=query,
                from_param=from_date,
                language='en',
                sort_by='publishedAt',
                page_size=50
            )

            articles = response.get('articles', [])
            logger.debug(f"Fetched {len(articles)} articles for {ticker}")

            return articles

        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []

    def _analyze_text(self, title: str, description: Optional[str] = None) -> Optional[Dict]:
        """Analyze sentiment of news text using FinBERT."""
        try:
            # Combine title and description
            text = title
            if description:
                text = f"{title}. {description}"

            # Truncate to avoid token limit
            text = text[:512]

            # Get sentiment
            result = self.sentiment_pipeline(text)[0]

            # Map FinBERT labels to scores
            label = result['label'].lower()
            confidence = result['score']

            if label == 'positive':
                sentiment_value = confidence
                positive = confidence
                negative = 0.0
                neutral = 1.0 - confidence
            elif label == 'negative':
                sentiment_value = -confidence
                positive = 0.0
                negative = confidence
                neutral = 1.0 - confidence
            else:  # neutral
                sentiment_value = 0.0
                positive = 0.0
                negative = 0.0
                neutral = confidence

            return {
                'sentiment': sentiment_value,
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'confidence': confidence,
                'label': label
            }

        except Exception as e:
            logger.warning(f"Error analyzing text: {e}")
            return None

    def _aggregate_sentiments(self, ticker: str, sentiments: List[Dict]) -> NewsSentiment:
        """
        Aggregate individual article sentiments into overall score.

        Uses recency weighting - newer articles have more influence.
        """
        if not sentiments:
            return self._default_sentiment(ticker)

        # Sort by date (newest first)
        sentiments.sort(key=lambda x: x['published_at'], reverse=True)

        # Calculate recency weights (exponential decay)
        weights = []
        for i in range(len(sentiments)):
            weight = np.exp(-0.1 * i)  # Decay factor
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        # Weighted average sentiment
        sentiment_scores = np.array([s['sentiment'] for s in sentiments])
        weighted_sentiment = np.average(sentiment_scores, weights=weights)

        # Calculate ratios
        positive_count = sum(1 for s in sentiments if s['label'] == 'positive')
        negative_count = sum(1 for s in sentiments if s['label'] == 'negative')
        neutral_count = sum(1 for s in sentiments if s['label'] == 'neutral')

        total = len(sentiments)
        positive_ratio = positive_count / total
        negative_ratio = negative_count / total
        neutral_ratio = neutral_count / total

        # Determine trend (compare recent vs older articles)
        if len(sentiments) >= 4:
            recent_sentiment = np.mean([s['sentiment'] for s in sentiments[:len(sentiments)//2]])
            older_sentiment = np.mean([s['sentiment'] for s in sentiments[len(sentiments)//2:]])

            if recent_sentiment > older_sentiment + 0.1:
                trend = "improving"
            elif recent_sentiment < older_sentiment - 0.1:
                trend = "deteriorating"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Overall confidence (average of individual confidences)
        avg_confidence = np.mean([s['confidence'] for s in sentiments])

        return NewsSentiment(
            ticker=ticker,
            sentiment_score=float(weighted_sentiment),
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            neutral_ratio=neutral_ratio,
            article_count=total,
            latest_headline=sentiments[0]['title'] if sentiments else None,
            sentiment_trend=trend,
            confidence=float(avg_confidence),
            timestamp=datetime.now()
        )

    def _aggregate_alpha_vantage_sentiments(self, ticker: str, sentiments: List[Dict]) -> NewsSentiment:
        """
        Aggregate Alpha Vantage sentiments with relevance weighting.

        Combines recency and relevance scores for optimal weighting.
        """
        if not sentiments:
            return self._default_sentiment(ticker)

        # Calculate combined weights (recency + relevance)
        weights = []
        for i, s in enumerate(sentiments):
            recency_weight = np.exp(-0.1 * i)  # Exponential decay
            relevance_weight = s.get('relevance', 0.5)  # 0-1 from Alpha Vantage
            combined_weight = recency_weight * (1 + relevance_weight)  # Boost by relevance
            weights.append(combined_weight)

        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        # Weighted average sentiment
        sentiment_scores = np.array([s['sentiment'] for s in sentiments])
        weighted_sentiment = np.average(sentiment_scores, weights=weights)

        # Calculate ratios
        positive_count = sum(1 for s in sentiments if s['label'] == 'positive')
        negative_count = sum(1 for s in sentiments if s['label'] == 'negative')
        neutral_count = sum(1 for s in sentiments if s['label'] == 'neutral')

        total = len(sentiments)
        positive_ratio = positive_count / total
        negative_ratio = negative_count / total
        neutral_ratio = neutral_count / total

        # Determine trend
        if len(sentiments) >= 4:
            recent_sentiment = np.mean([s['sentiment'] for s in sentiments[:len(sentiments)//2]])
            older_sentiment = np.mean([s['sentiment'] for s in sentiments[len(sentiments)//2:]])

            if recent_sentiment > older_sentiment + 0.1:
                trend = "improving"
            elif recent_sentiment < older_sentiment - 0.1:
                trend = "deteriorating"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Overall confidence (weighted by relevance)
        avg_confidence = np.average([s['confidence'] for s in sentiments], weights=weights)

        return NewsSentiment(
            ticker=ticker,
            sentiment_score=float(weighted_sentiment),
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            neutral_ratio=neutral_ratio,
            article_count=total,
            latest_headline=sentiments[0]['title'] if sentiments else None,
            sentiment_trend=trend,
            confidence=float(avg_confidence),
            timestamp=datetime.now()
        )

    def _default_sentiment(self, ticker: str) -> NewsSentiment:
        """Return neutral sentiment when no data available."""
        return NewsSentiment(
            ticker=ticker,
            sentiment_score=0.0,
            positive_ratio=0.33,
            negative_ratio=0.33,
            neutral_ratio=0.34,
            article_count=0,
            latest_headline=None,
            sentiment_trend="stable",
            confidence=0.0,
            timestamp=datetime.now()
        )

    def clear_cache(self):
        """Clear sentiment cache."""
        self.cache.clear()
        logger.debug("Sentiment cache cleared")

    def get_cached_sentiment(self, ticker: str) -> Optional[NewsSentiment]:
        """Get cached sentiment if available."""
        return self.cache.get(ticker)
