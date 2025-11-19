# Stock Analyzer - Institutional-Grade Investment Analysis Platform

> **Principal-Engineer Level**: A professional stock analysis system leveraging advanced financial algorithms, multi-factor models, and real-time data to identify high-potential investment opportunities.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Key Features

### Advanced Analysis
- **Multi-Factor Scoring**: Combines momentum, value, growth, and quality factors
- **Real-Time Data**: Live market data from multiple providers with automatic fallback
- **Technical Analysis**: 20+ indicators including RSI, MACD, ADX, Bollinger Bands
- **Fundamental Analysis**: Comprehensive metrics (P/E, PEG, ROE, debt ratios, margins)
- **Risk Metrics**: Sharpe ratio, max drawdown, volatility, VaR calculations

### Professional Architecture
- **SOLID Principles**: Clean architecture with dependency injection
- **Async-First**: Concurrent processing for analyzing hundreds of stocks
- **Type-Safe**: 100% type hints with Pydantic validation
- **Multi-Provider**: Automatic failover between Yahoo Finance, Alpha Vantage, Polygon.io
- **No Hardcoded Tickers**: Fetches live S&P 500 constituents from Wikipedia

### Investment Strategies
- **Momentum**: Multi-timeframe price momentum with volume confirmation
- **Value**: Deep value investing (Buffett/Graham principles)
- **Growth**: Growth-at-reasonable-price (GARP) with PEG analysis

---

## 🏗️ Architecture

```
stock_analyzer/
├── src/stock_analyzer/
│   ├── config/              # Configuration management (Pydantic)
│   ├── models/              # Domain models (Quote, Analysis, etc.)
│   ├── data/
│   │   ├── providers/       # Multi-provider data layer
│   │   └── provider_manager.py  # Failover & circuit breaker
│   ├── strategies/          # Scoring strategies (pluggable)
│   ├── services/            # Business logic (analyzer, screener)
│   ├── utils/               # Logging, exceptions, decorators
│   └── cli/                 # Command-line interface
├── tests/                   # Comprehensive test suite
├── claude.md                # Engineering principles & standards
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

**Key Design Patterns**:
- Strategy Pattern (scoring algorithms)
- Repository Pattern (data access)
- Factory Pattern (provider creation)
- Circuit Breaker (resilience)
- Async/Await (performance)

---

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to project
cd stock_analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment (optional)
cp config/.env.example .env
# Edit .env with your API keys (optional - works without keys using Yahoo Finance)
```

### Basic Usage

#### Analyze Single Stock
```bash
python -m src.stock_analyzer.cli.main analyze AAPL
```

**Output:**
```
================================================================================
Analyzing 1 stock(s)...
================================================================================

Ticker   Score   Signal       Trend      Price      RSI    P/E    ROE %
--------------------------------------------------------------------------------
AAPL     78.5    buy          bullish    $185.50    58.2   28.5   147.3

Top Pick: AAPL
  Score: 78.5/100
  Signal: BUY
  Strengths:
    - Strong ROE of 147.3%
    - Strong price momentum
    - Attractive valuation
  Risks:
    - High P/E ratio above sector average
```

#### Screen S&P 500 for Top Opportunities
```bash
python -m src.stock_analyzer.cli.main screen --top 20 --min-score 70
```

This will:
1. Fetch live S&P 500 constituents (no hardcoded list!)
2. Analyze all stocks concurrently (~2-3 minutes for 500 stocks)
3. Apply multi-factor scoring
4. Return top 20 highest-scoring opportunities

#### Find Strong Buy Signals
```bash
python -m src.stock_analyzer.cli.main screen --signal strong_buy --top 10
```

#### Analyze Multiple Stocks
```bash
python -m src.stock_analyzer.cli.main analyze AAPL MSFT GOOGL TSLA NVDA --output csv
```

---

## 📊 How It Works

### Multi-Factor Scoring System

Each stock receives a **composite score (0-100)** based on:

#### 1. Momentum Strategy (33.3% weight)
- **Price Momentum**: 1M, 3M, 6M, 12M returns
- **Trend Quality**: ADX, moving average alignment
- **Volume Confirmation**: Volume surges on up days
- **Technical Indicators**: RSI, MACD signals

#### 2. Value Strategy (33.3% weight)
- **Valuation**: P/E < 15, PEG < 1.0, P/B < 2.0
- **Profitability**: Profit margin > 15%, ROE > 20%
- **Financial Health**: Debt/Equity < 1.0, Current Ratio > 2.0
- **Dividends**: Yield > 2%, sustainable payout

#### 3. Growth Strategy (33.3% weight)
- **Revenue Growth**: > 15% YoY
- **Earnings Growth**: > 20% YoY
- **Margin Expansion**: Improving profitability
- **Growth Valuation**: PEG ratio < 1.5 (GARP)

### Signal Generation

| Score Range | Signal | Meaning |
|------------|--------|---------|
| 80-100 | STRONG_BUY | Exceptional opportunity |
| 60-79 | BUY | Attractive opportunity |
| 40-59 | HOLD | Neutral position |
| 20-39 | SELL | Avoid or exit |
| 0-19 | STRONG_SELL | Significant concerns |

---

## 🔧 Advanced Configuration

### Environment Variables

Create `.env` file (see `config/.env.example`):

```ini
# Data Providers
PRIMARY_PROVIDER=yahoo
ALPHA_VANTAGE_API_KEY=your_key_here  # Optional, for enhanced features

# Performance
MAX_CONCURRENT_REQUESTS=50  # Parallel API calls
CACHE_TTL_SECONDS=300  # Cache duration

# Filters
MIN_MARKET_CAP=1000000000  # $1B minimum
MIN_AVG_VOLUME=500000  # 500k shares minimum

# Analysis
SCORE_THRESHOLD=60.0  # Minimum buy score
```

### API Keys (Optional)

While the system works great with just Yahoo Finance (no API key needed), you can add these for enhanced features:

- **Alpha Vantage** (Free): https://www.alphavantage.co/support/#api-key
- **Polygon.io** (Paid): https://polygon.io/
- **Finnhub** (Freemium): https://finnhub.io/

---

## 💻 Programmatic Usage

### Python API

```python
import asyncio
from src.stock_analyzer.services.analyzer import StockAnalyzer
from src.stock_analyzer.data.provider_manager import ProviderManager

async def analyze_stocks():
    # Initialize
    async with ProviderManager() as provider_manager:
        analyzer = StockAnalyzer(provider_manager=provider_manager)

        # Analyze single stock
        analysis = await analyzer.analyze("AAPL")
        print(f"Score: {analysis.composite_score}/100")
        print(f"Signal: {analysis.signal.value}")

        # Batch analysis
        analyses = await analyzer.analyze_batch(["AAPL", "MSFT", "GOOGL"])
        for a in analyses:
            print(f"{a.ticker}: {a.composite_score:.1f}")

# Run
asyncio.run(analyze_stocks())
```

### Custom Strategies

```python
from src.stock_analyzer.strategies.base import ScoringStrategy
from decimal import Decimal

class MyCustomStrategy(ScoringStrategy):
    """Custom scoring strategy."""

    def calculate_score(
        self,
        price_data,
        fundamentals=None,
        technical_indicators=None,
        stock_info=None
    ) -> Decimal:
        # Your custom logic here
        score = 0.0

        if fundamentals and fundamentals.pe_ratio:
            pe = float(fundamentals.pe_ratio)
            if pe < 20:
                score += 50

        if technical_indicators and technical_indicators.rsi:
            rsi = float(technical_indicators.rsi)
            if 40 < rsi < 60:
                score += 50

        return Decimal(str(score))

# Use custom strategy
analyzer = StockAnalyzer(strategies=[MyCustomStrategy(weight=1.0)])
```

---

## 📈 Example Workflows

### Find Undervalued Growth Stocks
```bash
# Screen for stocks with:
# - Score > 70
# - Strong fundamentals
# - Growth potential
python -m src.stock_analyzer.cli.main screen --min-score 70 --top 15
```

### Monitor Watchlist
```bash
# Analyze your watchlist daily
python -m src.stock_analyzer.cli.main analyze AAPL MSFT GOOGL TSLA NVDA AMD AMZN --output csv

# Save to CSV and track changes over time
```

### Market Sentiment Analysis
```bash
# Screen entire S&P 500 to gauge market health
python -m src.stock_analyzer.cli.main screen --top 100
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=src/stock_analyzer tests/

# Run specific test
pytest tests/unit/test_analyzer.py -v
```

---

## 🎓 Engineering Principles

This project follows **principal-engineer level standards**. See `claude.md` for:

- ✅ SOLID Principles (SRP, OCP, LSP, ISP, DIP)
- ✅ KISS, DRY, YAGNI
- ✅ Clean Architecture & Domain-Driven Design
- ✅ Comprehensive error handling
- ✅ Type safety (mypy strict)
- ✅ Async/await patterns
- ✅ Testing strategies
- ✅ Security best practices

---

## 🔐 Security

- ✅ No hardcoded secrets (environment variables only)
- ✅ Input validation (Pydantic)
- ✅ SQL injection prevention (parameterized queries)
- ✅ Rate limiting
- ✅ Secrets management (SecretStr)

Run security audit:
```bash
bandit -r src/stock_analyzer/
```

---

## 📝 Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/
pylint src/

# Type check
mypy src/
```

---

## 🚦 Performance

**Benchmarks** (on standard laptop):
- Single stock analysis: ~2-3 seconds
- S&P 500 screening (500 stocks): ~2-3 minutes
- Concurrent requests: 50 parallel API calls

**Optimizations**:
- Async/await for I/O operations
- Connection pooling
- Response caching (5-min TTL)
- Batch processing with semaphores

---

## 🗺️ Roadmap

### Phase 1 (Current)
- [x] Multi-factor scoring (momentum, value, growth)
- [x] Live S&P 500 fetching
- [x] Multi-provider system with fallback
- [x] Advanced technical indicators
- [x] CLI interface

### Phase 2 (Next)
- [ ] Machine learning price prediction (XGBoost, LightGBM)
- [ ] Sentiment analysis from news/social media
- [ ] Options flow analysis
- [ ] Portfolio optimization (Modern Portfolio Theory)
- [ ] Web dashboard (React + FastAPI)

### Phase 3 (Future)
- [ ] Real-time streaming data (WebSockets)
- [ ] Automated trading integration
- [ ] Backtesting engine with transaction costs
- [ ] Risk management tools (position sizing, stop-loss)
- [ ] Mobile app (React Native)

---

## 🤝 Contributing

We follow principal engineering standards. Before contributing:

1. Read `claude.md` for coding standards
2. Run tests: `pytest tests/`
3. Format code: `black src/`
4. Type check: `mypy src/`
5. Lint: `flake8 src/`

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- Past performance ≠ future results
- Always do your own research (DYOR)
- Consult a licensed financial advisor
- Never invest more than you can afford to lose
- Markets carry inherent risk of loss

---

## 📄 License

MIT License - See LICENSE file for details

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/stock_analyzer/issues)
- **Documentation**: See `claude.md` for detailed engineering docs
- **Email**: support@example.com

---

## 🏆 Acknowledgments

Built with:
- **yfinance** - Yahoo Finance data
- **pandas-ta** - Technical analysis
- **Pydantic** - Data validation
- **aiohttp** - Async HTTP
- **BeautifulSoup** - Web scraping

Inspired by quantitative hedge fund strategies and institutional research.

---

**Happy Investing! 🚀📈**
