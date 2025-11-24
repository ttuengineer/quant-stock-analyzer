# Quick Start: Optimal Configuration

**Last Updated:** 2025-11-24
**Status:** Production Ready

---

## 🚀 Run Optimal Backtest (One Command)

```bash
cd "/path/to/quant-stock-analyzer"

.venv/Scripts/python.exe scripts/walk_forward_validation.py \
    --ensemble 3 \
    --mega-cap-overlay \
    --min-mega-cap-allocation 0.60 \
    --mega-cap-force-top-k 5
```

**Expected Results:**
- Portfolio: +269.2% over 8 years
- SPY: +125.5%
- **Excess: +143.7%**
- **Sharpe: 0.52** (beats SPY's 0.40)
- Win 6/8 years

---

## 📊 What This Does

- Trains 3 XGBoost models with different seeds
- Runs walk-forward validation (2018-2025)
- Forces 60% allocation to top 5 mega-caps (AAPL, MSFT, NVDA, GOOGL, AMZN)
- Allocates 40% to ML-selected stocks
- Returns year-by-year performance vs SPY

---

## 🎯 Key Metrics

| Metric | Value |
|--------|-------|
| Total Return | +269.2% |
| SPY Return | +125.5% |
| **Excess Return** | **+143.7%** |
| Annualized Return | +20.3% |
| **Sharpe Ratio** | **0.52** |
| Max Drawdown | -34.6% |
| Beta | 1.57 |
| **Alpha** | **+4.0%** |
| Win Rate | 75% (6/8 years) |

---

## ⚙️ Configuration Explained

### `--ensemble 3`
Trains 3 models with different random seeds and averages predictions. More models = more stable but slower.

### `--mega-cap-overlay`
Enables the mega-cap overlay module that forces inclusion of top SPY holdings.

### `--min-mega-cap-allocation 0.60`
Ensures at least 60% of portfolio is allocated to mega-caps. This is the optimal balance.

### `--mega-cap-force-top-k 5`
Forces inclusion of top 5 SPY mega-caps (AAPL, MSFT, NVDA, GOOGL, AMZN/META) regardless of ML score.

---

## 📁 Output Files

After running, check:
- `models/walk_forward_results.csv` - Monthly returns
- `models/walk_forward_summary.csv` - Performance summary
- Console output - Year-by-year breakdown

---

## 🔧 Alternative Configurations

### Conservative (40% mega-caps)
```bash
.venv/Scripts/python.exe scripts/walk_forward_validation.py \
    --ensemble 3 \
    --mega-cap-overlay \
    --min-mega-cap-allocation 0.40
```
**Result:** +63.1% excess (less concentrated)

### Aggressive (Force 8 mega-caps)
```bash
.venv/Scripts/python.exe scripts/walk_forward_validation.py \
    --ensemble 3 \
    --mega-cap-overlay \
    --min-mega-cap-allocation 0.30 \
    --mega-cap-force-top-k 8
```
**Result:** +29.4% excess (more diversified but worse 2022)

---

## 📖 More Information

- Full results: `MEGA_CAP_OVERLAY_FINAL_RESULTS.md`
- Root cause analysis: `ROOT_CAUSE_ANALYSIS.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`

---

**Bottom Line:** Run the first command. Get +143.7% excess return vs SPY. Done.
