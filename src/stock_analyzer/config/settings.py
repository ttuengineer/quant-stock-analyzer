"""
Application configuration using Pydantic BaseSettings.

Follows 12-factor app principles with environment-based configuration.
All secrets are loaded from environment variables, never hardcoded.
"""

from typing import Optional, List
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings
from enum import Enum


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DataProvider(str, Enum):
    """Available data providers."""

    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"


class Settings(BaseSettings):
    """
    Application settings with validation.

    All sensitive values are SecretStr to prevent accidental logging.
    Load from .env file or environment variables.
    """

    # Application Settings
    app_name: str = "Stock Analyzer"
    environment: Environment = Environment.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    debug: bool = False

    # Data Provider Configuration
    primary_provider: DataProvider = DataProvider.YAHOO
    fallback_providers: List[DataProvider] = [
        DataProvider.ALPHA_VANTAGE,
        DataProvider.POLYGON
    ]

    # API Keys (SecretStr prevents logging)
    alpha_vantage_api_key: Optional[SecretStr] = None
    polygon_api_key: Optional[SecretStr] = None
    finnhub_api_key: Optional[SecretStr] = None
    rapidapi_key: Optional[SecretStr] = None  # RapidAPI Yahoo Finance
    rapidapi_host: Optional[str] = "yahoo-finance15.p.rapidapi.com"  # RapidAPI host

    # Database Configuration
    database_url: Optional[SecretStr] = None
    database_pool_size: int = 20

    # Cache Configuration
    redis_url: Optional[str] = None
    cache_ttl_seconds: int = 300  # 5 minutes default
    enable_cache: bool = True

    # Rate Limiting
    max_requests_per_minute: int = 60
    request_timeout_seconds: int = 30

    # Market Data Settings
    default_lookback_period: str = "1y"
    min_market_cap: int = 1_000_000_000  # $1B minimum
    min_avg_volume: int = 500_000  # 500k shares daily minimum

    # Analysis Settings
    min_data_points: int = 60  # Minimum days of data required
    score_threshold: float = 60.0  # Minimum score for buy signal
    max_concurrent_requests: int = 50  # Max parallel API calls

    # ML Model Settings
    ml_model_path: Optional[str] = None
    ml_retrain_interval_days: int = 7
    ml_feature_lookback_days: int = 90

    # Stock Universe Settings
    # IMPORTANT: No hardcoded tickers - fetch from live sources
    fetch_universe_from_live: bool = True
    stock_exchanges: List[str] = ["NASDAQ", "NYSE", "AMEX"]
    exclude_sectors: List[str] = []  # e.g., ["Utilities", "Real Estate"]

    @field_validator("alpha_vantage_api_key", "polygon_api_key", "finnhub_api_key", "rapidapi_key", mode="before")
    @classmethod
    def validate_api_keys(cls, v, info):
        """Warn if API keys are missing in production."""
        if v is None:
            import warnings
            warnings.warn(
                f"{info.field_name} is not set. Some features may be unavailable.",
                UserWarning
            )
        return v

    @field_validator("min_market_cap")
    @classmethod
    def validate_market_cap(cls, v):
        """Ensure minimum market cap is reasonable."""
        if v < 0:
            raise ValueError("min_market_cap cannot be negative")
        return v

    @field_validator("score_threshold")
    @classmethod
    def validate_score_threshold(cls, v):
        """Ensure score threshold is within valid range."""
        if not 0 <= v <= 100:
            raise ValueError("score_threshold must be between 0 and 100")
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "validate_assignment": True,
    }


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get application settings (singleton pattern).

    Returns:
        Settings instance loaded from environment

    Example:
        >>> settings = get_settings()
        >>> api_key = settings.alpha_vantage_api_key.get_secret_value()
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Force reload settings from environment.

    Useful for testing or when environment changes.
    """
    global _settings
    _settings = Settings()
    return _settings
