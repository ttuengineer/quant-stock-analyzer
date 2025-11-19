# 🚀 Stock Analyzer UI - Quick Start Guide

## 📺 Launch the Web Dashboard

### Method 1: Double-Click (Easiest)
1. Navigate to: `C:\Users\dgarz\OneDrive\Desktop\Dev\stock_analyzer`
2. **Double-click** `run_ui.bat`
3. Your browser will automatically open to `http://localhost:8501`

### Method 2: VS Code
1. Open VS Code
2. Open folder: `C:\Users\dgarz\OneDrive\Desktop\Dev\stock_analyzer`
3. Open terminal (`Ctrl+\``)
4. Run:
   ```bash
   .venv\Scripts\activate
   streamlit run app.py
   ```
5. Browser opens automatically

### Method 3: Command Line
```bash
cd C:\Users\dgarz\OneDrive\Desktop\Dev\stock_analyzer
.venv\Scripts\streamlit run app.py
```

---

## 🎯 What You'll See

### 📊 Dashboard (Main Page)
- **Quick Search**: Analyze any stock instantly
- **Real-time Metrics**: Score, Signal, Price, Trend
- **Factor Scores**: Beautiful gauge charts for Momentum/Value/Growth
- **Technical Indicators**: RSI, MACD, ADX, Volume
- **Fundamentals**: P/E, PEG, ROE, Profit Margin
- **Key Strengths & Risks**: AI-generated insights

**Try it:**
- Enter "AAPL" → Click "Analyze"
- See comprehensive analysis with charts!

### 🔎 Stock Screener
- **Screen Multiple Stocks**: Tech Giants, Dow 30, or Custom lists
- **Advanced Filters**:
  - Minimum Score (0-100)
  - Signal Type (Strong Buy, Buy, Hold, etc.)
  - Max Results
- **Interactive Table**: Sortable, filterable results
- **Charts**: Score distribution visualizations
- **Export**: Download results as CSV

**Try it:**
1. Select "Tech Giants"
2. Set Min Score: 50
3. Select Signals: "strong_buy", "buy"
4. Click "Run Screen"
5. See top opportunities ranked!

### 🔥 Hot Buys
- **Auto-scan** 23 popular stocks
- **Find best opportunities** automatically
- **Ranked by score** (highest first)
- **Top 10 picks** with details
- **Quick insights** for each stock

**Try it:**
- Click "Find Hot Buys"
- See instant top picks!

---

## 🎨 Features

✅ **Beautiful UI** - Modern gradient design, responsive layout
✅ **Real-Time Data** - Live stock prices and analysis
✅ **Interactive Charts** - Gauge charts, candlestick charts, bar charts
✅ **Caching** - 5-minute cache for fast performance
✅ **Export Data** - Download CSV reports
✅ **Mobile Responsive** - Works on any device

---

## 💡 Tips

1. **First Time**: Let the first analysis run (takes ~3 seconds)
2. **Cache**: Subsequent analyses of same stock are instant (cached)
3. **Batch Analysis**: Screener analyzes multiple stocks in parallel
4. **Refresh Data**: Wait 5 minutes or restart server for fresh data

---

## 🛑 Stopping the Server

- In terminal: Press `Ctrl+C`
- Or close the terminal window

---

## 📸 Screenshots

### Dashboard
```
┌─────────────────────────────────────────────────┐
│  📈 Stock Analyzer Pro                          │
│  Institutional-Grade Investment Analysis        │
├─────────────────────────────────────────────────┤
│  🔍 Enter stock ticker: [AAPL      ] [Analyze] │
├─────────────────────────────────────────────────┤
│  Score: 50.3/100  │  Signal: HOLD  │  Price: $267.44
├─────────────────────────────────────────────────┤
│  ┌─ Momentum ─┐  ┌─ Technical ─┐  ┌─ Strengths ─┐
│  │ [Gauge 50] │  │ RSI: 54.6   │  │ ✓ Strong ROE
│  │            │  │ MACD: Bear  │  │ ✓ High margin
│  └────────────┘  └─────────────┘  └──────────────┘
└─────────────────────────────────────────────────┘
```

### Screener
```
Found 8 stocks matching criteria
┌────────┬───────┬───────┬─────────┬────────┐
│ Ticker │ Score │ Signal│ Price   │ P/E    │
├────────┼───────┼───────┼─────────┼────────┤
│ GOOGL  │  61.7 │ BUY   │ $284.28 │ 22.1   │
│ AAPL   │  50.3 │ HOLD  │ $267.44 │ 35.9   │
└────────┴───────┴───────┴─────────┴────────┘
[📥 Download Results (CSV)]
```

---

## 🆘 Troubleshooting

**Port already in use?**
```bash
streamlit run app.py --server.port 8502
```

**Cache issues?**
- Click "Clear Cache" in hamburger menu (☰)
- Or restart the server

**Slow first load?**
- Normal! First analysis fetches live data
- Subsequent loads are instant (cached)

---

## 🎓 What's Happening Behind the Scenes

1. **Streamlit** renders the beautiful UI
2. **Async processing** fetches data from Yahoo Finance
3. **3 scoring strategies** evaluate each stock:
   - Momentum (trend analysis)
   - Value (Graham/Buffett principles)
   - Growth (GARP methodology)
4. **20+ indicators** calculated in real-time
5. **Caching** stores results for 5 minutes
6. **Plotly** generates interactive charts

---

**Enjoy your institutional-grade stock analysis dashboard! 🚀📈**
