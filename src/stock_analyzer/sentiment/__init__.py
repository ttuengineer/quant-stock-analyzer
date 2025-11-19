"""
Sentiment Analysis module for news and social media.

Includes:
- News sentiment analysis with FinBERT
- NewsAPI integration for real-time headlines
- Sentiment scoring and aggregation
- Historical sentiment tracking
"""

from .news_analyzer import NewsAnalyzer, NewsSentiment

__all__ = ["NewsAnalyzer", "NewsSentiment"]
