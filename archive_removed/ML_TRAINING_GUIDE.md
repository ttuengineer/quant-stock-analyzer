# ML Model Training Guide

## Overview

Your stock analyzer uses Machine Learning models (XGBoost, LightGBM, RandomForest) to predict stock price direction and returns. This guide shows you how to train these models with historical data.

---

## Quick Start (3 Steps)

### 1. Run the Training Script

Open your terminal and run:

```bash
cd "C:\Users\dgarz\OneDrive\Desktop\Dev\Github\quant-stock-analyzer"
python scripts/train_models.py
```

**That's it!** The script handles everything automatically.

---

### 2. Wait for Training to Complete

The script will:
- ✓ Download 3 years of historical data for ~70 stocks
- ✓ Calculate 60+ technical and fundamental features
- ✓ Train 6 ML models (3 for direction, 3 for return magnitude)
- ✓ Validate performance on test data
- ✓ Save models to the `models/` folder

**Estimated time: 15-30 minutes** (depending on your internet speed and CPU)

---

### 3. Restart Streamlit

Once training finishes, restart your Streamlit app:

```bash
streamlit run app.py
```

The ML Prediction strategy will now use your trained models!

---

## What Happens During Training?

### Data Collection
The script downloads historical data for **70 diverse stocks** across sectors:
- **Technology**: AAPL, MSFT, GOOGL, META, NVDA, etc.
- **Healthcare**: JNJ, UNH, PFE, ABBV, MRK, LLY, etc.
- **Financials**: JPM, BAC, WFC, GS, MS, etc.
- **Consumer**: AMZN, TSLA, WMT, HD, MCD, etc.
- **Energy, Industrials, Utilities, Communications**

This ensures the models learn patterns from various market conditions and sectors.

---

### Feature Engineering (60+ Features)

For each stock, the script calculates:

**Price Momentum (20 features)**:
- Returns over 1d, 5d, 10d, 20d, 60d periods
- Volatility measures
- Distance from 52-week high/low
- Moving average ratios (SMA 20, 50, 200)

**Technical Indicators (25 features)**:
- RSI, MACD, ADX, ATR
- Bollinger Bands position and width
- Stochastic oscillator
- Trend strength indicators

**Volume Analysis (10 features)**:
- Volume trends and surges
- Volume-price relationships

**Fundamentals (15 features)**:
- P/E, P/B, ROE, ROA ratios
- Debt levels, profit margins
- Growth rates

**Market Regime (5 features)**:
- VIX levels
- Market trend (bull/bear)
- Sector performance

---

### Model Training

**Three models trained for EACH task** (6 total):

#### Direction Prediction (Classification):
1. **XGBoost Classifier** - Gradient boosting with regularization
2. **LightGBM Classifier** - Fast gradient boosting
3. **RandomForest Classifier** - Ensemble of decision trees

#### Return Magnitude Prediction (Regression):
4. **XGBoost Regressor**
5. **LightGBM Regressor**
6. **RandomForest Regressor**

**Ensemble voting**: Predictions are combined using weighted voting:
- XGBoost: 40%
- LightGBM: 35%
- RandomForest: 25%

---

### Performance Metrics

After training, you'll see metrics like:

```
====================================================
TRAINING COMPLETE!
====================================================

Direction Classification Metrics:
  Accuracy:  0.62    (62% correct direction predictions)
  Precision: 0.64    (64% of predicted UPs are correct)
  Recall:    0.68    (catches 68% of actual UPs)
  F1 Score:  0.66    (balanced precision/recall)

Return Prediction Metrics:
  MAE:  0.0234       (average error: 2.34%)
  RMSE: 0.0312       (root mean squared error)
  R²:   0.18         (explains 18% of return variance)

====================================================
```

**What's good performance?**
- **Accuracy > 55%**: Better than random (50%)
- **Precision > 60%**: Most BUY signals are correct
- **MAE < 3%**: Predictions within 3% of actual returns
- **R² > 0.10**: Models capture meaningful patterns

Stock markets are noisy, so 55-65% accuracy is **excellent**!

---

## Customization (Advanced)

### Change Training Period

Edit `scripts/train_models.py` line 488:

```python
# Default: 3 years
await trainer.run_training_pipeline(years=3)

# Train on 5 years instead
await trainer.run_training_pipeline(years=5)
```

### Change Training Stocks

Edit the `get_training_tickers()` method (lines 33-68) to add/remove tickers.

### Adjust Model Parameters

Edit `_train_classifiers()` and `_train_regressors()` in `src/stock_analyzer/ml/predictor.py` to tune hyperparameters:

```python
self.xgb_clf = xgb.XGBClassifier(
    n_estimators=100,      # Number of trees (increase for more accuracy)
    max_depth=5,           # Tree depth (increase for more complexity)
    learning_rate=0.1,     # Step size (decrease for slower, more careful learning)
    ...
)
```

---

## Retraining Schedule

**When to retrain:**
- **Initial setup**: Train once to get started
- **Monthly**: Update with new market data
- **After major market events**: Capture new patterns
- **When performance degrades**: Check accuracy on recent predictions

**Automatic retraining**: The system checks if models are >90 days old and suggests retraining.

---

## Troubleshooting

### Error: "No training data collected"
**Fix**: Check your internet connection. The script needs to download historical data from Yahoo Finance.

### Error: "ModuleNotFoundError: No module named 'xgboost'"
**Fix**: Install ML dependencies:
```bash
pip install xgboost lightgbm scikit-learn joblib
```

### Training is very slow
**Normal**: First training takes 15-30 minutes to download all historical data.
**Speed up**: Reduce the number of training tickers in `get_training_tickers()`.

### Out of memory errors
**Fix**: Reduce training period from 3 years to 2 years, or train on fewer stocks.

---

## How Models Are Used

After training, the **ML Prediction Strategy** uses your models to:

1. **Analyze new stocks**:
   - Calculate same 60+ features
   - Feed to trained models
   - Get ensemble prediction

2. **Generate signals**:
   - **STRONG_BUY**: >70% chance of gain, >3% predicted return
   - **BUY**: >60% chance of gain, >1% predicted return
   - **HOLD**: Neutral prediction
   - **SELL**: <40% chance of gain, <-1% predicted return

3. **Contribute to composite score**:
   - ML score weighted with 7 other strategies
   - Adaptive weighting based on market regime
   - Final score determines overall signal

---

## Model Files

After training, you'll see these files in `models/`:

```
models/
├── stock_predictor_20251119.pkl    (timestamped backup)
└── stock_predictor_latest.pkl      (used by analyzer)
```

**Backup**: The timestamped file is kept for rollback if needed.
**Active**: The "latest" file is automatically loaded by the analyzer.

---

## Expected Results

With trained models, you should see:

**Before training** (ML returns neutral 50.0):
```
Strategy Scores:
- Momentum: 65
- Value: 45
- Quality: 80
- ML Prediction: 50  ← neutral (not trained)
Composite: 60 (BUY)
```

**After training** (ML provides real predictions):
```
Strategy Scores:
- Momentum: 65
- Value: 45
- Quality: 80
- ML Prediction: 72  ← real prediction!
Composite: 65.5 (BUY)
```

The ML strategy now contributes meaningful insights instead of neutral scores.

---

## Next Steps

After successful training:

1. ✓ **Analyze stocks** - ML strategy now active
2. ✓ **Monitor performance** - Track if ML predictions are accurate
3. ✓ **Retrain monthly** - Keep models up-to-date
4. ⚠️ **Backtest strategies** - Validate historical performance (advanced)

---

## Questions?

**Check logs**: `logs/stock_analyzer.log` for detailed training progress

**Model not loading?**
- Verify `models/stock_predictor_latest.pkl` exists
- Check file permissions
- Try retraining

**Low accuracy?**
- Normal for stock prediction (55-65% is good!)
- Markets are inherently noisy
- Combine with other strategies (momentum, value, quality)

---

## Summary

```bash
# 1. Train models (15-30 min, run once)
python scripts/train_models.py

# 2. Restart Streamlit
streamlit run app.py

# 3. Done! ML strategy now active
```

Your analyzer now uses **state-of-the-art machine learning** alongside traditional quant strategies! 🚀
