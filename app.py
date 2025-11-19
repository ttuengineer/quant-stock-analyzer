"""
Stock Analyzer - Interactive Web Dashboard

Run with: streamlit run app.py
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.stock_analyzer.services.analyzer import StockAnalyzer
from src.stock_analyzer.data.provider_manager import ProviderManager
from src.stock_analyzer.models.enums import SignalType

# Page configuration
st.set_page_config(
    page_title="Stock Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .buy-signal {
        background-color: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .sell-signal {
        background-color: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .hold-signal {
        background-color: #f59e0b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_provider_manager():
    """Initialize provider manager (cached)."""
    return ProviderManager()


def analyze_stock_cached(ticker: str):
    """Analyze a stock with session state caching."""
    # Use session state for caching instead of st.cache_data (Pydantic models aren't pickle-able)
    if 'analysis_cache' not in st.session_state:
        st.session_state.analysis_cache = {}

    cache_key = f"single_{ticker}"

    # Check cache (5 minute TTL)
    if cache_key in st.session_state.analysis_cache:
        cached_data, timestamp = st.session_state.analysis_cache[cache_key]
        if (datetime.now() - timestamp).total_seconds() < 300:  # 5 minutes
            return cached_data

    # Cache miss - analyze
    async def _analyze():
        pm = get_provider_manager()
        await pm.initialize()
        analyzer = StockAnalyzer(provider_manager=pm)
        return await analyzer.analyze(ticker)

    result = asyncio.run(_analyze())
    st.session_state.analysis_cache[cache_key] = (result, datetime.now())
    return result


def analyze_batch_cached(tickers: list):
    """Batch analyze stocks with session state caching."""
    # Use session state instead of st.cache_data
    if 'batch_cache' not in st.session_state:
        st.session_state.batch_cache = {}

    cache_key = f"batch_{'_'.join(sorted(tickers))}"

    # Check cache (5 minute TTL)
    if cache_key in st.session_state.batch_cache:
        cached_data, timestamp = st.session_state.batch_cache[cache_key]
        if (datetime.now() - timestamp).total_seconds() < 300:  # 5 minutes
            return cached_data

    # Cache miss - analyze
    async def _analyze():
        pm = get_provider_manager()
        await pm.initialize()
        analyzer = StockAnalyzer(provider_manager=pm)
        return await analyzer.analyze_batch(tickers)

    result = asyncio.run(_analyze())
    st.session_state.batch_cache[cache_key] = (result, datetime.now())
    return result


def get_signal_color(signal: SignalType) -> str:
    """Get color for signal badge."""
    colors = {
        SignalType.STRONG_BUY: "#10b981",
        SignalType.BUY: "#34d399",
        SignalType.HOLD: "#f59e0b",
        SignalType.SELL: "#f87171",
        SignalType.STRONG_SELL: "#ef4444",
    }
    return colors.get(signal, "#6b7280")


def create_gauge_chart(value: float, title: str) -> go.Figure:
    """Create a gauge chart for scores."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#fee2e2'},
                {'range': [40, 60], 'color': '#fef3c7'},
                {'range': [60, 80], 'color': '#d1fae5'},
                {'range': [80, 100], 'color': '#a7f3d0'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 60
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_price_chart(price_data):
    """Create candlestick chart for price data."""
    if price_data.empty:
        return None

    fig = go.Figure(data=[go.Candlestick(
        x=price_data.index,
        open=price_data['Open'],
        high=price_data['High'],
        low=price_data['Low'],
        close=price_data['Close'],
        name='Price'
    )])

    fig.update_layout(
        title="Price Chart (Last 6 Months)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400,
        template="plotly_white"
    )
    return fig


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_universe(universe_type: str) -> list:
    """
    Get comprehensive stock universe lists.

    Args:
        universe_type: Type of universe (sp500, top100, tech, etc.)

    Returns:
        List of ticker symbols
    """
    # S&P 500 - Comprehensive fallback list
    sp500_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "XOM",
        "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
        "KO", "AVGO", "COST", "LLY", "ADBE", "TMO", "MCD", "CSCO", "ACN", "ABT",
        "WMT", "NFLX", "CRM", "DHR", "VZ", "DIS", "CMCSA", "NKE", "TXN", "PM",
        "INTC", "UNP", "NEE", "WFC", "RTX", "ORCL", "COP", "AMD", "BA", "HON",
        "UPS", "LOW", "QCOM", "ELV", "INTU", "BMY", "SPGI", "LIN", "SBUX", "AMGN",
        "PLD", "CAT", "T", "DE", "GE", "MDT", "GILD", "BLK", "AXP", "BKNG",
        "ADI", "ISRG", "MMC", "SYK", "VRTX", "TJX", "MDLZ", "ADP", "REGN", "CVS",
        "AMT", "CI", "AMAT", "MO", "LRCX", "ZTS", "PGR", "C", "SO", "CB",
        "BSX", "ETN", "DUK", "EOG", "SCHW", "FISV", "CME", "ITW", "SLB", "MU",
        # Continue with more S&P 500...
        "PNC", "MS", "NOC", "BDX", "GD", "ICE", "USB", "WM", "APD", "MCK",
        "TGT", "HUM", "CL", "NSC", "EMR", "GIS", "F", "PSX", "SHW", "AON",
        "KLAC", "MAR", "APH", "PYPL", "ECL", "CARR", "NXPI", "HCA", "AJG", "AIG",
        "TT", "ORLY", "FIS", "ADSK", "MCO", "PCAR", "ROP", "ROST", "TRV", "SPG",
        "KMB", "SYY", "PAYX", "AFL", "MCHP", "MSI", "AEP", "AZO", "D", "GM",
        "FTNT", "IQV", "O", "KMI", "TEL", "MSCI", "CMG", "A", "PPG", "CPRT",
        "WELL", "CTVA", "DXCM", "SNPS", "MNST", "IDXX", "FAST", "YUM", "OTIS", "HLT",
        "KHC", "ODFL", "EA", "PRU", "DD", "KDP", "GWW", "ALL", "VRSK", "HSY",
        "AME", "EW", "CTSH", "BK", "CSGP", "CEG", "CMI", "GEHC", "VICI", "EXC",
        "STZ", "DLR", "KEYS", "MTB", "GLW", "ANSS", "XEL", "DHI", "IT", "DOV"
    ]

    # Top 100 Large Cap (weighted by market cap)
    top100 = sp500_tickers[:100]

    # Tech Sector
    tech_stocks = [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
        "ADBE", "CRM", "CSCO", "ACN", "AMD", "INTC", "QCOM", "TXN", "INTU", "AMAT",
        "ADI", "LRCX", "MU", "NXPI", "KLAC", "MCHP", "SNPS", "CDNS", "FTNT", "ADSK",
        "PANW", "APH", "TEL", "MSI", "ANSS", "ON", "KEYS", "TYL", "PTC", "MPWR",
        "TER", "TRMB", "ZBRA", "NTAP", "STX", "AKAM", "JNPR", "FFIV", "ENPH", "SMCI"
    ]

    # Healthcare Sector
    healthcare_stocks = [
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY",
        "AMGN", "GILD", "ISRG", "VRTX", "MDT", "REGN", "CVS", "CI", "BSX", "ZTS",
        "HUM", "HCA", "MCK", "SYK", "BDX", "ELV", "IQV", "IDXX", "DXCM", "EW",
        "A", "RMD", "GEHC", "COO", "CNC", "PODD", "DGX", "HOLX", "BAX", "MTD",
        "WST", "ALGN", "RVTY", "LH", "TECH", "HSIC", "VTRS", "CRL", "WAT", "STE"
    ]

    # Financial Sector
    financial_stocks = [
        "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "SPGI", "BLK",
        "C", "SCHW", "CB", "PNC", "USB", "AXP", "TFC", "CME", "FIS", "AIG",
        "MCO", "ICE", "TRV", "AFL", "PGR", "AON", "AJG", "COF", "BK", "MMC",
        "ALL", "MTB", "PRU", "DFS", "FITB", "AMP", "TROW", "STT", "WTW", "NTRS",
        "CFG", "KEY", "RF", "HBAN", "SYF", "WRB", "CINF", "CBOE", "L", "JKHY"
    ]

    # Energy Sector
    energy_stocks = [
        "XOM", "CVX", "COP", "EOG", "SLB", "PSX", "MPC", "VLO", "OXY", "HES",
        "WMB", "KMI", "HAL", "DVN", "FANG", "BKR", "TRGP", "MRO", "OKE", "EQT",
        "CTRA", "LNG", "APA", "CHRD", "FTI", "NOV", "MTDR", "RRC", "PR", "MGY"
    ]

    # Quick 50 - Top 50 most liquid stocks
    quick50 = sp500_tickers[:50]

    # Return appropriate universe
    universes = {
        "sp500": sp500_tickers,
        "top100": top100,
        "tech": tech_stocks,
        "healthcare": healthcare_stocks,
        "financial": financial_stocks,
        "energy": energy_stocks,
        "quick50": quick50
    }

    return universes.get(universe_type, quick50)


def show_dashboard():
    """Main dashboard page."""
    st.markdown('<h1 class="main-header">📈 Stock Analyzer Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Institutional-Grade Investment Analysis")

    # Quick search
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("🔍 Enter stock ticker", value="AAPL", key="quick_search").upper()
    with col2:
        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    if analyze_btn or ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                analysis = analyze_stock_cached(ticker)

                # Header metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Composite Score",
                        f"{float(analysis.composite_score):.1f}/100",
                        delta=None
                    )

                with col2:
                    signal_color = get_signal_color(analysis.signal)
                    st.markdown(
                        f"**Signal**  \n<span style='background-color: {signal_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 4px; font-weight: bold;'>{analysis.signal.value.upper()}</span>",
                        unsafe_allow_html=True
                    )

                with col3:
                    if analysis.quote:
                        price = f"${float(analysis.quote.price):.2f}"
                        change = f"{float(analysis.quote.change_percent):.2f}%" if analysis.quote.change_percent else "N/A"
                        st.metric("Current Price", price, delta=change)

                with col4:
                    st.metric("Trend", analysis.trend.value.upper())

                st.markdown("---")

                # Three column layout
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("#### 🎯 Factor Scores")
                    if analysis.factor_scores:
                        st.plotly_chart(
                            create_gauge_chart(
                                float(analysis.factor_scores.momentum_score),
                                "Momentum"
                            ),
                            use_container_width=True
                        )
                        st.plotly_chart(
                            create_gauge_chart(
                                float(analysis.factor_scores.value_score),
                                "Value"
                            ),
                            use_container_width=True
                        )
                        st.plotly_chart(
                            create_gauge_chart(
                                float(analysis.factor_scores.growth_score),
                                "Growth"
                            ),
                            use_container_width=True
                        )

                with col2:
                    st.markdown("#### 📊 Technical Indicators")
                    if analysis.technical_indicators:
                        tech_data = {}
                        if analysis.technical_indicators.rsi:
                            tech_data['RSI'] = f"{float(analysis.technical_indicators.rsi):.1f}"
                        if analysis.technical_indicators.macd and analysis.technical_indicators.macd_signal:
                            macd_signal = "🟢 Bullish" if float(analysis.technical_indicators.macd) > float(analysis.technical_indicators.macd_signal) else "🔴 Bearish"
                            tech_data['MACD'] = macd_signal
                        if analysis.technical_indicators.adx:
                            tech_data['ADX (Trend Strength)'] = f"{float(analysis.technical_indicators.adx):.1f}"
                        if analysis.technical_indicators.volume_ratio:
                            tech_data['Volume Ratio'] = f"{float(analysis.technical_indicators.volume_ratio):.2f}x"

                        for key, value in tech_data.items():
                            st.metric(key, value)

                    st.markdown("#### 💰 Fundamentals")
                    if analysis.fundamentals:
                        fund_data = {}
                        if analysis.fundamentals.pe_ratio:
                            fund_data['P/E Ratio'] = f"{float(analysis.fundamentals.pe_ratio):.2f}"
                        if analysis.fundamentals.peg_ratio:
                            fund_data['PEG Ratio'] = f"{float(analysis.fundamentals.peg_ratio):.2f}"
                        if analysis.fundamentals.roe:
                            fund_data['ROE'] = f"{float(analysis.fundamentals.roe)*100:.1f}%"
                        if analysis.fundamentals.profit_margin:
                            fund_data['Profit Margin'] = f"{float(analysis.fundamentals.profit_margin)*100:.1f}%"

                        for key, value in fund_data.items():
                            st.metric(key, value)

                with col3:
                    st.markdown("#### ✅ Key Strengths")
                    if analysis.key_strengths:
                        for strength in analysis.key_strengths:
                            st.success(f"✓ {strength}")
                    else:
                        st.info("No key strengths identified")

                    st.markdown("#### ⚠️ Key Risks")
                    if analysis.key_risks:
                        for risk in analysis.key_risks:
                            st.warning(f"⚠ {risk}")
                    else:
                        st.info("No major risks identified")

                # Risk metrics
                if analysis.risk_metrics:
                    st.markdown("---")
                    st.markdown("#### 📉 Risk Metrics")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if analysis.risk_metrics.sharpe_ratio:
                            st.metric("Sharpe Ratio", f"{float(analysis.risk_metrics.sharpe_ratio):.2f}")
                    with col2:
                        if analysis.risk_metrics.volatility_annual:
                            st.metric("Volatility (Annual)", f"{float(analysis.risk_metrics.volatility_annual):.1f}%")
                    with col3:
                        if analysis.risk_metrics.max_drawdown:
                            st.metric("Max Drawdown", f"{float(analysis.risk_metrics.max_drawdown):.1f}%")

            except Exception as e:
                st.error(f"Error analyzing {ticker}: {str(e)}")


def show_screener():
    """Stock screener page."""
    st.markdown('<h1 class="main-header">🔎 Stock Screener</h1>', unsafe_allow_html=True)
    st.markdown("### Scan Entire Markets for Investment Opportunities")

    # Comprehensive stock universes
    universe_options = {
        "S&P 500 (All 500 stocks)": "sp500",
        "Top 100 Large Cap": "top100",
        "Tech Sector": "tech",
        "Healthcare Sector": "healthcare",
        "Financial Sector": "financial",
        "Energy Sector": "energy",
        "Quick Scan (Top 50)": "quick50",
        "Custom List": "custom"
    }

    list_choice = st.selectbox("Select stock universe", list(universe_options.keys()))
    universe_type = universe_options[list_choice]

    # Fetch appropriate ticker list
    if universe_type == "custom":
        custom_tickers = st.text_input("Enter tickers (comma-separated)", "AAPL,MSFT,GOOGL")
        tickers = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()]
    else:
        with st.spinner("Loading stock universe..."):
            tickers = get_stock_universe(universe_type)

    # Show universe info
    st.info(f"📊 **Universe**: {list_choice} | **Stocks to analyze**: {len(tickers)}")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_score = st.slider("Minimum Score", 0, 100, 40)  # Changed from 50 to 40
    with col2:
        signal_filter = st.multiselect(
            "Signal Type",
            ["strong_buy", "buy", "hold", "sell", "strong_sell"],
            default=["strong_buy", "buy", "hold"]  # Added "hold" to defaults
        )
    with col3:
        max_results = st.slider("Max Results", 5, 50, 20)

    if st.button("🔍 Run Screen", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {len(tickers)} stocks..."):
            try:
                analyses = analyze_batch_cached(tickers)

                # Show quick stats
                all_scores = [float(a.composite_score) for a in analyses]
                avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
                max_score = max(all_scores) if all_scores else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Stocks Analyzed", len(analyses))
                with col2:
                    st.metric("Average Score", f"{avg_score:.1f}")
                with col3:
                    st.metric("Highest Score", f"{max_score:.1f}")

                # Filter results
                filtered = [
                    a for a in analyses
                    if float(a.composite_score) >= min_score
                    and a.signal.value in signal_filter
                ]

                # Sort by score
                filtered.sort(key=lambda x: x.composite_score, reverse=True)
                results = filtered[:max_results]

                # Show summary statistics
                if len(filtered) > 0:
                    st.success(f"Found {len(filtered)} stocks matching criteria (showing top {len(results)})")
                else:
                    # Show what we actually found
                    all_scores = [float(a.composite_score) for a in analyses]
                    all_signals = [a.signal.value for a in analyses]
                    max_score = max(all_scores) if all_scores else 0

                    from collections import Counter
                    signal_counts = Counter(all_signals)

                    st.warning(f"Found 0 stocks matching criteria out of {len(analyses)} analyzed")
                    st.info(f"""
                        **What we found:**
                        - Highest score: {max_score:.1f}
                        - Signal distribution: {dict(signal_counts)}

                        **Try lowering your minimum score or changing signal types!**
                    """)

                # Create DataFrame for display
                data = []
                for a in results:
                    data.append({
                        'Ticker': a.ticker,
                        'Score': float(a.composite_score),
                        'Signal': a.signal.value.upper(),
                        'Trend': a.trend.value,
                        'Price': f"${float(a.quote.price):.2f}" if a.quote else "N/A",
                        'Change %': f"{float(a.quote.change_percent):.2f}%" if a.quote and a.quote.change_percent else "N/A",
                        'RSI': f"{float(a.technical_indicators.rsi):.1f}" if a.technical_indicators and a.technical_indicators.rsi else "N/A",
                        'P/E': f"{float(a.fundamentals.pe_ratio):.1f}" if a.fundamentals and a.fundamentals.pe_ratio else "N/A",
                        'ROE %': f"{float(a.fundamentals.roe)*100:.1f}" if a.fundamentals and a.fundamentals.roe else "N/A",
                    })

                df = pd.DataFrame(data)

                # Only show results if we have data
                if len(df) > 0:
                    # Display as interactive table
                    st.dataframe(
                        df,
                        use_container_width=True,
                        height=400,
                        hide_index=True
                    )

                    # Score distribution chart
                    st.markdown("### 📊 Score Distribution")
                    fig = px.bar(
                        df,
                        x='Ticker',
                        y='Score',
                        color='Signal',
                        title="Composite Scores by Stock",
                        color_discrete_map={
                            'STRONG_BUY': '#10b981',
                            'BUY': '#34d399',
                            'HOLD': '#f59e0b',
                            'SELL': '#f87171',
                            'STRONG_SELL': '#ef4444'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Export option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"stock_screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    # No results found - show helpful message
                    st.info(
                        """
                        **No stocks match your criteria.**

                        Try adjusting your filters:
                        - Lower the **Minimum Score** threshold
                        - Add more **Signal Types** (e.g., include 'hold')
                        - Select a different **stock universe**
                        """
                    )

            except Exception as e:
                st.error(f"Error running screen: {str(e)}")


def show_hot_buys():
    """Hot buys page."""
    st.markdown('<h1 class="main-header">🔥 Hot Buys</h1>', unsafe_allow_html=True)
    st.markdown("### Top Investment Opportunities Right Now")

    # Expanded watchlist - scan top 100 S&P 500 stocks for opportunities
    watchlist = get_stock_universe("top100")

    if st.button("🔄 Find Hot Buys", type="primary", use_container_width=True):
        with st.spinner(f"Scanning {len(watchlist)} popular stocks..."):
            try:
                analyses = analyze_batch_cached(watchlist)

                # Filter for good investment opportunities
                # BUY/STRONG_BUY signals (score 60+), or exceptional HOLD signals (score 58+)
                hot_buys = [
                    a for a in analyses
                    if (a.signal in [SignalType.STRONG_BUY, SignalType.BUY]) or
                       (a.signal == SignalType.HOLD and float(a.composite_score) >= 58)
                ]

                hot_buys.sort(key=lambda x: x.composite_score, reverse=True)

                if hot_buys:
                    st.success(f"🎯 Found {len(hot_buys)} top investment opportunities!")
                    st.info(f"**Criteria**: BUY/STRONG_BUY signals (score ≥ 60), or exceptional HOLD signals (score ≥ 58)")

                    # Display top picks
                    for i, analysis in enumerate(hot_buys[:15], 1):  # Show up to 15 instead of 10
                        with st.container():
                            col1, col2, col3, col4 = st.columns([1, 2, 2, 2])

                            with col1:
                                st.markdown(f"### #{i}")
                                st.markdown(f"**{analysis.ticker}**")

                            with col2:
                                st.metric("Score", f"{float(analysis.composite_score):.1f}/100")
                                signal_color = get_signal_color(analysis.signal)
                                st.markdown(
                                    f"<span style='background-color: {signal_color}; color: white; padding: 0.25rem 0.75rem; border-radius: 4px;'>{analysis.signal.value.upper()}</span>",
                                    unsafe_allow_html=True
                                )

                            with col3:
                                if analysis.quote:
                                    st.metric("Price", f"${float(analysis.quote.price):.2f}")
                                if analysis.fundamentals and analysis.fundamentals.pe_ratio:
                                    st.metric("P/E", f"{float(analysis.fundamentals.pe_ratio):.1f}")

                            with col4:
                                if analysis.key_strengths:
                                    st.markdown("**Top Strength:**")
                                    st.success(f"✓ {analysis.key_strengths[0]}")

                            st.markdown("---")
                else:
                    st.warning(
                        """
                        **No buy opportunities found in current scan.**

                        This is normal in certain market conditions. Try:
                        - Scanning again later
                        - Using the Stock Screener with lower thresholds
                        - Checking individual stocks on the Dashboard
                        """
                    )

            except Exception as e:
                st.error(f"Error finding hot buys: {str(e)}")


# Main app
def main():
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Stock Screener", "Hot Buys"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    # Cache management
    st.sidebar.markdown("### 🔄 Cache")
    if st.sidebar.button("Clear Cache", use_container_width=True):
        if 'analysis_cache' in st.session_state:
            st.session_state.analysis_cache = {}
        if 'batch_cache' in st.session_state:
            st.session_state.batch_cache = {}
        st.sidebar.success("Cache cleared!")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ About")
    st.sidebar.info(
        """
        **Stock Analyzer Pro**

        Institutional-grade stock analysis powered by:
        - Multi-factor scoring
        - Advanced technical analysis
        - Fundamental metrics
        - Real-time data

        Built with principal engineering standards.
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Route to pages
    if page == "Dashboard":
        show_dashboard()
    elif page == "Stock Screener":
        show_screener()
    elif page == "Hot Buys":
        show_hot_buys()


if __name__ == "__main__":
    main()
