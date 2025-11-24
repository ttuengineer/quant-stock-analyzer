# ML Training - Quick Start

## Never trained ML models before? Follow these 3 simple steps:

---

## Step 1: Open Terminal

Press `Windows Key + R`, type `cmd`, press Enter

---

## Step 2: Run Training Command

Copy and paste this into the terminal:

```bash
cd "C:\Users\dgarz\OneDrive\Desktop\Dev\Github\quant-stock-analyzer"
python scripts/train_models.py
```

Press Enter and **wait 15-30 minutes**.

You'll see progress like this:

```
======================================================================
ML MODEL TRAINING PIPELINE
======================================================================

This will train XGBoost, LightGBM, and RandomForest models
using 3 years of historical data.

Estimated time: 15-30 minutes
======================================================================

[1/3] Collecting historical data...
Processing AAPL: 100%|████████████████| 70/70 [10:23<00:00,  8.91s/ticker]
Collected 12,450 training samples

[2/3] Training models...
Training complete!

[3/3] Saving models...

======================================================================
✓ TRAINING COMPLETE!
======================================================================

Direction Classification Metrics:
  Accuracy:  62.4%
  Precision: 64.1%
  Recall:    67.8%
  F1 Score:  65.9%

Return Prediction Metrics:
  MAE:  2.34%
  RMSE: 3.12%
  R²:   0.187

======================================================================
Your models are ready to use!
Restart your Streamlit app to use the trained ML strategy.
======================================================================
```

---

## Step 3: Restart Streamlit

After training finishes, restart your app:

```bash
streamlit run app.py
```

---

## That's It!

Your ML models are now trained and active. The system will automatically:
- Load the trained models
- Use them to predict stock returns
- Combine ML predictions with 7 other strategies
- Generate smarter BUY/SELL signals

---

## What Just Happened?

The script:
✓ Downloaded 3 years of data for 70 stocks
✓ Calculated 60+ features per stock
✓ Trained 6 machine learning models
✓ Validated accuracy (should be 55-65%)
✓ Saved models to `models/` folder

---

## When to Retrain?

**Monthly**: Run `python scripts/train_models.py` again to update with new data

---

## Need Help?

See `ML_TRAINING_GUIDE.md` for detailed explanations and troubleshooting.

---

**You're now using institutional-grade ML for stock analysis!** 🚀
