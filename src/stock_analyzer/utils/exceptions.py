"""
Custom exception hierarchy for stock analyzer.

Following principal engineering practices, we define a clear exception
hierarchy for better error handling and debugging.
"""


class StockAnalyzerError(Exception):
    """Base exception for all stock analyzer errors."""

    pass


class ConfigurationError(StockAnalyzerError):
    """Raised when configuration is invalid or missing."""

    pass


class DataProviderError(StockAnalyzerError):
    """Base exception for data provider errors."""

    pass


class RateLimitError(DataProviderError):
    """Raised when API rate limit is exceeded."""

    def __init__(self, provider: str, retry_after: int):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for {provider}. Retry after {retry_after}s"
        )


class DataNotFoundError(DataProviderError):
    """Raised when requested data is not found."""

    pass


class ValidationError(StockAnalyzerError):
    """Raised when data validation fails."""

    pass


class AnalysisError(StockAnalyzerError):
    """Raised when stock analysis fails."""

    pass


class CacheError(StockAnalyzerError):
    """Raised when cache operation fails."""

    pass


class ProviderUnavailableError(DataProviderError):
    """Raised when all data providers are unavailable."""

    pass
