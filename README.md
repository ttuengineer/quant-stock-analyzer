# Quant Stock Analyzer - Institutional-Grade Investment Analysis

> **Quantitative Multi-Factor Analysis**: A professional stock screening platform leveraging advanced financial algorithms, multi-factor models, and real-time data to identify high-potential investment opportunities.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://streamlit.io)

---

## 🎯 Key Features

### 📊 Interactive Web Dashboard
- **Real-Time Analysis**: Live stock quotes and comprehensive analysis
- **Stock Screener**: Scan 200+ S&P 500 stocks across multiple sectors
- **Hot Buys**: Discover top investment opportunities with BUY signals
- **Visual Charts**: Interactive price charts, gauges, and performance metrics

### 🧮 Quantitative Analysis
- **Multi-Factor Scoring**: Combines momentum (33%), value (33%), and growth (33%)
- **20+ Technical Indicators**: RSI, MACD, ADX, Bollinger Bands, volume analysis
- **Comprehensive Fundamentals**: P/E, PEG, ROE, ROA, profit margins, debt ratios
- **Risk Metrics**: Sharpe ratio, max drawdown, volatility, VaR calculations

### 🏗️ Professional Architecture
- **SOLID Principles**: Clean architecture with dependency injection
- **Async-First**: Concurrent analysis of 200+ stocks (99% success rate)
- **Type-Safe**: 100% type hints with Pydantic v2 validation
- **Multi-Provider**: RapidAPI Yahoo Finance 15 (primary) with yfinance fallback
- **Live Data**: Fetches S&P 500 constituents from Wikipedia (no hardcoded tickers)

### 💼 Investment Strategies
- **Momentum**: Multi-timeframe price momentum with volume confirmation
- **Value**: Deep value investing (Buffett/Graham principles)
- **Growth**: Growth-at-reasonable-price (GARP) with PEG analysis

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ttuengineer/quant-stock-analyzer.git
cd quant-stock-analyzer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment (optional)
cp .env.example .env
# Edit .env with your API keys (optional - works great without keys using yfinance)
```

### Launch Web Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` with three pages:

1. **📈 Dashboard**: Analyze individual stocks with detailed metrics
2. **🔎 Stock Screener**: Screen S&P 500 by sector (Tech, Healthcare, Financial, Energy)
3. **🔥 Hot Buys**: Discover top investment opportunities

### Quick CLI Usage

```bash
# Analyze single stock
python -m src.stock_analyzer.cli.main analyze AAPL

# Screen for top opportunities
python -m src.stock_analyzer.cli.main screen --top 20 --min-score 60

# Find strong buy signals
python -m src.stock_analyzer.cli.main screen --signal strong_buy
```

---

## 📊 How It Works

### Multi-Factor Scoring System

Each stock receives a **composite score (0-100)** based on three equal-weighted factors:

#### 1. Momentum Strategy (33.3%)
- **Price Momentum**: 1M, 3M, 6M returns with volume confirmation
- **Trend Quality**: ADX > 25, moving average alignment (SMA 20/50/200)
- **Technical Signals**: RSI (30-70 range), MACD crossovers
- **Volume Analysis**: Above-average volume on up days

#### 2. Value Strategy (33.3%)
- **Valuation**: P/E < 15, PEG < 1.0, P/B < 2.0, P/S < 2.0
- **Profitability**: Profit margin > 15%, ROE > 20%, ROA > 10%
- **Financial Health**: Debt/Equity < 1.0, Current Ratio > 2.0
- **Dividends**: Yield > 2%, sustainable payout ratio

#### 3. Growth Strategy (33.3%)
- **Revenue Growth**: > 15% YoY
- **Earnings Growth**: > 20% YoY
- **Margin Expansion**: Improving profitability trends
- **Growth Valuation**: PEG < 1.5 (GARP methodology)

### Signal Generation (Institutional-Grade)

| Score Range | Signal | Meaning | Rarity |
|------------|--------|---------|--------|
| **80-100** | STRONG_BUY | Exceptional opportunity | Very Rare |
| **60-79** | BUY | Attractive opportunity | ~5-10% |
| **40-59** | HOLD | Neutral position | ~60-70% |
| **20-39** | SELL | Avoid or exit | ~15-20% |
| **0-19** | STRONG_SELL | Significant concerns | ~5% |

> **Note**: Unlike consumer apps with inflated scores, this system uses institutional-grade criteria. A score of 60+ represents a genuine buying opportunity, not 90+.

---

## 🏗️ Architecture

```
quant-stock-analyzer/
├── app.py                   # Streamlit web dashboard
├── src/stock_analyzer/
│   ├── config/              # Settings (Pydantic v2)
│   ├── models/              # Domain models (Quote, Analysis, Fundamentals)
│   ├── data/
│   │   ├── providers/       # Yahoo (RapidAPI + yfinance), Alpha Vantage
│   │   └── provider_manager.py  # Multi-provider failover
│   ├── strategies/          # Momentum, Value, Growth (pluggable)
│   ├── services/            # StockAnalyzer, batch processing
│   ├── utils/               # Logging, decorators, exceptions
│   └── cli/                 # Command-line interface
├── tests/                   # Unit & integration tests
├── requirements.txt         # Dependencies
└── README.md               # This file
```

**Key Design Patterns**:
- Strategy Pattern (scoring algorithms)
- Repository Pattern (data providers)
- Circuit Breaker (resilience)
- Async/Await (50 parallel requests)

---

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```ini
# RapidAPI Yahoo Finance 15 (Primary - Optional)
RAPIDAPI_KEY=your_rapidapi_key_here
RAPIDAPI_HOST=yahoo-finance15.p.rapidapi.com

# Fallback Providers (Optional)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key
POLYGON_API_KEY=your_polygon_key

# Application Settings
LOG_LEVEL=INFO
MAX_CONCURRENT_REQUESTS=50
CACHE_TTL_SECONDS=300

# Analysis Filters
MIN_MARKET_CAP=1000000000  # $1B minimum
MIN_AVG_VOLUME=500000      # 500k shares
SCORE_THRESHOLD=60.0       # Buy threshold
```

### API Keys (Optional but Recommended)

The system works great with **free yfinance** (no API key needed), but you can add:

1. **RapidAPI Yahoo Finance 15** (Primary) - [Get Key](https://rapidapi.com/sparior/api/yahoo-finance15)
   - ✅ Real-time quotes
   - ✅ Faster than free alternatives
   - ✅ More reliable for batch analysis
   - **Pro Plan**: $9.99/month for 10,000 requests

2. **Alpha Vantage** (Free) - [Get Key](https://www.alphavantage.co/support/#api-key)
   - 500 requests/day free tier

3. **Finnhub** (Freemium) - [Get Key](https://finnhub.io/)
   - 60 calls/minute free tier

The system automatically uses **RapidAPI → yfinance fallback** for maximum reliability.

---

## 💻 Programmatic Usage

### Python API

```python
import asyncio
from src.stock_analyzer.services.analyzer import StockAnalyzer
from src.stock_analyzer.data.provider_manager import ProviderManager

async def analyze_stocks():
    pm = ProviderManager()
    await pm.initialize()
    analyzer = StockAnalyzer(provider_manager=pm)

    # Single stock
    analysis = await analyzer.analyze("AAPL")
    print(f"{analysis.ticker}: {analysis.composite_score}/100 ({analysis.signal.value})")

    # Batch analysis (concurrent)
    analyses = await analyzer.analyze_batch(["AAPL", "MSFT", "GOOGL", "NVDA"])
    for a in sorted(analyses, key=lambda x: x.composite_score, reverse=True):
        print(f"{a.ticker}: {a.composite_score:.1f} - {a.signal.value}")

    await pm.cleanup()

asyncio.run(analyze_stocks())
```

### Custom Strategies

```python
from src.stock_analyzer.strategies.base import ScoringStrategy
from decimal import Decimal

class MyStrategy(ScoringStrategy):
    """Custom dividend-focused strategy."""

    def calculate_score(self, price_data, fundamentals=None, **kwargs) -> Decimal:
        score = 0.0

        if fundamentals:
            # Reward high dividend yield
            if fundamentals.dividend_yield and fundamentals.dividend_yield > 0.03:
                score += 50

            # Reward low payout ratio (sustainable)
            if fundamentals.payout_ratio and fundamentals.payout_ratio < 0.6:
                score += 50

        return Decimal(str(score))

# Use custom strategy
from src.stock_analyzer.strategies.momentum import MomentumStrategy
from src.stock_analyzer.strategies.value import ValueStrategy

analyzer = StockAnalyzer(strategies=[
    MomentumStrategy(weight=0.25),
    ValueStrategy(weight=0.25),
    MyStrategy(weight=0.50)  # 50% weight on dividends
])
```

---

## 📈 Example Workflows

### Find Top S&P 500 Opportunities

```bash
# Screen entire S&P 500 for BUY signals
python -m src.stock_analyzer.cli.main screen --min-score 60 --top 20
```

**Expected Output:**
```
Analyzing 200 stocks...
Successfully analyzed 198/200 stocks (99% success rate)

Top 20 Investment Opportunities:
Ticker   Score   Signal   Trend      Price      RSI    P/E    ROE %
------------------------------------------------------------------------
LLY      68.0    buy      bullish    $612.45    62.3   45.2   52.1
MRK      62.3    buy      bullish    $98.75     58.1   18.4   28.9
GOOGL    61.7    buy      bullish    $142.30    55.4   25.8   31.2
...
```

### Monitor Daily Watchlist

```bash
# Analyze specific stocks and export to CSV
python -m src.stock_analyzer.cli.main analyze AAPL MSFT GOOGL NVDA AMD --output csv
```

### Sector-Specific Screening

Using the Streamlit dashboard:
1. Navigate to **Stock Screener**
2. Select **"Tech Sector"** (50 stocks) or **"Healthcare"** (50 stocks)
3. Set minimum score: **40**
4. Include signals: **strong_buy, buy, hold**
5. Click **"Run Screen"**

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest --cov=src/stock_analyzer tests/

# Specific test
pytest tests/unit/test_strategies.py -v
```

---

## 🚦 Performance

**Benchmarks** (on standard laptop):
- Single stock: ~2-3 seconds
- Top 100 S&P 500: ~15 seconds
- Full S&P 500 (200 stocks): ~30 seconds
- Success rate: **99%** (198/200 stocks)

**Optimizations**:
- ✅ Async/await with 50 parallel requests
- ✅ Session-based caching (5-min TTL)
- ✅ Connection pooling
- ✅ Graceful degradation (RapidAPI → yfinance)

---

## 🎓 Engineering Principles

**Principal-Engineer Level Standards** (see `claude.md`):
- ✅ SOLID Principles (SRP, OCP, LSP, ISP, DIP)
- ✅ KISS, DRY, YAGNI
- ✅ Clean Architecture
- ✅ Type Safety (Pydantic v2)
- ✅ Comprehensive Error Handling
- ✅ Security Best Practices (no hardcoded secrets)

---

## 🗺️ Roadmap

### ✅ Phase 1 (Complete)
- [x] Multi-factor scoring (momentum, value, growth)
- [x] Live S&P 500 fetching (Wikipedia)
- [x] Multi-provider with RapidAPI + yfinance
- [x] 20+ technical indicators
- [x] Streamlit web dashboard
- [x] Stock screener with sector filtering
- [x] Hot Buys page

### 🚧 Phase 2 (In Progress)
- [ ] Machine learning predictions (XGBoost, LightGBM)
- [ ] Sentiment analysis (news/Twitter)
- [ ] Options flow analysis
- [ ] Portfolio optimization (Markowitz)
- [ ] Backtesting engine

### 🔮 Phase 3 (Future)
- [ ] Real-time WebSocket data
- [ ] Automated trading integration
- [ ] Risk management tools
- [ ] Mobile app (React Native)
- [ ] Paper trading simulator

---

## ⚠️ Disclaimer

**For Educational and Research Purposes Only**

- ❌ Not financial advice
- ❌ Not investment recommendations
- ❌ Past performance ≠ future results
- ✅ Always do your own research (DYOR)
- ✅ Consult a licensed financial advisor
- ✅ Never invest more than you can afford to lose
- ⚠️ All investing carries risk of loss

**The creators of this software are not registered investment advisors and assume no liability for financial losses.**

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🤝 Contributing

Contributions welcome! Please:

1. Read `claude.md` for coding standards
2. Run tests: `pytest tests/`
3. Format: `black src/ && isort src/`
4. Type check: `mypy src/`
5. Submit PR with clear description

---

## 🏆 Acknowledgments

Built with:
- **Streamlit** - Interactive web dashboard
- **yfinance** - Yahoo Finance data (fallback)
- **pandas-ta** - Technical analysis
- **Pydantic v2** - Data validation
- **aiohttp** - Async HTTP
- **requests** - RapidAPI integration
- **plotly** - Interactive charts

Inspired by quantitative hedge fund strategies and institutional research methodologies.

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/ttuengineer/quant-stock-analyzer/issues)
- **Documentation**: See `claude.md` for engineering docs
- **Discussions**: [GitHub Discussions](https://github.com/ttuengineer/quant-stock-analyzer/discussions)

---

**Built with principal-engineer standards. Happy investing! 🚀📈**
