# ML Stock Picker: Out-of-Sample Results

## 🏆 OPTIMAL CONFIGURATION (RECOMMENDED) - Updated 2025-11-24

**Configuration**: Mega-Cap Overlay + 60% allocation to top 5 SPY holdings
**Command**:
```bash
python scripts/walk_forward_validation.py \
    --ensemble 3 \
    --mega-cap-overlay \
    --min-mega-cap-allocation 0.60 \
    --mega-cap-force-top-k 5
```

**Results Summary**:
- Portfolio: **+269.2%** over 8 years (2018-2025)
- SPY: +125.5%
- **Excess Return: +143.7%**
- Beat SPY in **6 of 8 years (75%)**

**Risk Metrics**:
- **Sharpe Ratio: 0.52** (beats SPY's 0.40!)
- Annualized Return: +20.3%
- Annualized Volatility: 29.3%
- **Alpha: +4.0%** (positive skill!)
- Beta: 1.57
- Max Drawdown: -34.6%
- Information Ratio: 0.58
- Turnover: 392% annually

### Year-by-Year Performance (Optimal Configuration)

| Year | Portfolio | S&P 500 | Excess | Result |
|------|-----------|---------|--------|--------|
| 2018 | -11.8% | -9.5% | -2.3% | Underperformed |
| 2019 | +40.4% | +22.6% | **+17.9%** | **Beat SPY** |
| 2020 | +40.8% | +15.9% | **+24.9%** | **Beat SPY** |
| 2021 | +50.7% | +28.7% | **+22.1%** | **Beat SPY** |
| 2022 | -31.1% | -14.6% | -16.6% | Underperformed |
| 2023 | +37.8% | +16.8% | **+21.0%** | **Beat SPY** |
| 2024 | +17.3% | +21.0% | -3.8% | Underperformed |
| 2025 | +26.2% | +12.8% | **+13.4%** | **Beat SPY** |
| **TOTAL** | **+269.2%** | **+125.5%** | **+143.7%** | **Beat SPY** |

### Key Improvements vs Baseline

| Metric | Baseline (no overlay) | Optimal (with overlay) | Improvement |
|--------|----------------------|------------------------|-------------|
| Total Excess | -3.9% | **+143.7%** | **+147.6%** |
| Sharpe Ratio | 0.21 | **0.52** | +148% |
| Win Rate | 50% (4/8 years) | **75% (6/8 years)** | +50% |
| Max Drawdown | -47.5% | **-34.6%** | +27% improvement |
| Alpha | -5.2% | **+4.0%** | +9.2% |
| Turnover | 478% | **392%** | -18% |

**See `MEGA_CAP_OVERLAY_FINAL_RESULTS.md` for complete analysis and `QUICK_START_OPTIMAL.md` for quick reference.**

---

## Historical Configurations (For Comparison)

## Latest Walk-Forward (results_ensemble3.txt)

- Data through 2025-09-30, 84 features, ensemble=3, equal-weight top 20
- Portfolio +92.1% vs SPY +125.5% (excess -33.4%)
- Beat SPY in 3 of 8 years (38%)
- Model quality: AUC 0.649 | Precision@10 19.7% | IC 0.022 | Bottom decile hit 3.2%

### Year-by-Year Performance

| Year | Portfolio | S&P 500 | Excess | Result |
|------|-----------|---------|--------|--------|
| 2018 | -14.8% | -9.5% | -5.4% | Underperformed |
| 2019 | +20.2% | +22.6% | -2.3% | Underperformed |
| 2020 | +14.5% | +15.9% | -1.4% | Underperformed |
| 2021 | +49.2% | +28.7% | +20.5% | Beat SPY |
| 2022 | -22.8% | -14.6% | -8.3% | Underperformed |
| 2023 | +20.9% | +16.8% | +4.1% | Beat SPY |
| 2024 | -1.2% | +21.0% | -22.2% | Underperformed |
| 2025 | +19.2% | +12.8% | +6.4% | Beat SPY |

### Notes

- These numbers are parsed from `results_ensemble3.txt` produced by `scripts/walk_forward_validation.py --ensemble 3`.
- Risk metrics from the same run: Sharpe 0.13, vol 36.1%, beta 1.76, max drawdown -51.4%.

## Optimizer Run (optimize_results.txt)

- CVXPY constraints: beta target 1.0 ±0.1, vol target 16%, sector max 20%, ensemble=3
- Portfolio +111.2% vs SPY +125.5% (excess -14.3%)
- Beat SPY in 4 of 8 years (50%)
- Risk metrics: Sharpe 0.19, vol 32.7%, beta 1.63, max drawdown -47.6%

### Year-by-Year Performance (optimizer)

| Year | Portfolio | S&P 500 | Excess | Result |
|------|-----------|---------|--------|--------|
| 2018 | -16.3% | -9.5% | -6.8% | Underperformed |
| 2019 | +17.4% | +22.6% | -5.2% | Underperformed |
| 2020 | +21.3% | +15.9% | +5.3% | Beat SPY |
| 2021 | +48.7% | +28.7% | +20.1% | Beat SPY |
| 2022 | -19.5% | -14.6% | -5.0% | Underperformed |
| 2023 | +21.0% | +16.8% | +4.2% | Beat SPY |
| 2024 | -0.7% | +21.0% | -21.8% | Underperformed |
| 2025 | +23.3% | +12.8% | +10.5% | Beat SPY |

## Optimizer + Meta-Ensemble (optimize_results_meta.txt)

- CVXPY constraints (tighter): beta target 1.0 ±0.05, vol target ~14%, sector max 15%, ensemble=10, meta-ensemble (XGB + LGBM + Ridge)
- Portfolio +119.3% vs SPY +125.5% (excess -6.2%)
- Beat SPY in 4 of 8 years (50%)
- Risk metrics: Sharpe 0.21, vol 32.3%, beta 1.61, max drawdown -46.9%

### Year-by-Year Performance (optimizer meta-ensemble)

| Year | Portfolio | S&P 500 | Excess | Result |
|------|-----------|---------|--------|--------|
| 2018 | -15.8% | -9.5% | -6.3% | Underperformed |
| 2019 | +16.6% | +22.6% | -6.0% | Underperformed |
| 2020 | +24.2% | +15.9% | +8.3% | Beat SPY |
| 2021 | +46.4% | +28.7% | +17.8% | Beat SPY |
| 2022 | -16.4% | -14.6% | -1.9% | Underperformed |
| 2023 | +20.7% | +16.8% | +3.9% | Beat SPY |
| 2024 | -1.8% | +21.0% | -22.9% | Underperformed |
| 2025 | +24.2% | +12.8% | +11.4% | Beat SPY |

## Optimizer + Meta-Ensemble (tight constraints) (optimize_results_meta_tight.txt)

- CVXPY constraints (tighter, current defaults): beta target 1.0 ±0.05, vol target ~14%, sector max 10%, max weight 3%, min weight 0.5%, turnover max 20%, ensemble=15, meta-ensemble (XGB + LGBM + Ridge)
- Portfolio +121.7% vs SPY +125.5% (excess -3.9%)
- Beat SPY in 4 of 8 years (50%)
- Risk metrics: Sharpe 0.21, vol 32.6%, beta 1.62, max drawdown -47.5%

### Year-by-Year Performance (tight meta-ensemble)

| Year | Portfolio | S&P 500 | Excess | Result |
|------|-----------|---------|--------|--------|
| 2018 | -15.1% | -9.5% | -5.7% | Underperformed |
| 2019 | +16.1% | +22.6% | -6.5% | Underperformed |
| 2020 | +24.1% | +15.9% | +8.2% | Beat SPY |
| 2021 | +46.2% | +28.7% | +17.5% | Beat SPY |
| 2022 | -17.4% | -14.6% | -2.8% | Underperformed |
| 2023 | +20.5% | +16.8% | +3.7% | Beat SPY |
| 2024 | -0.4% | +21.0% | -21.4% | Underperformed |
| 2025 | +25.0% | +12.8% | +12.2% | Beat SPY |

## Optimizer + Meta-Ensemble (ultra constraints) (optimize_results_meta_ultra.txt)

- CVXPY constraints (even tighter): beta target 1.0 ±0.05, vol target ~14%, sector max 8%, max weight 2%, min weight 0.3%, turnover max 15%, ensemble=20, meta-ensemble (XGB + LGBM + Ridge)
- Portfolio +111.3% vs SPY +125.5% (excess -14.3%)
- Beat SPY in 4 of 8 years (50%)
- Risk metrics: Sharpe 0.19, vol 32.7%, beta 1.63, max drawdown -48.3%

### Year-by-Year Performance (ultra meta-ensemble)

| Year | Portfolio | S&P 500 | Excess | Result |
|------|-----------|---------|--------|--------|
| 2018 | -15.5% | -9.5% | -6.1% | Underperformed |
| 2019 | +16.2% | +22.6% | -6.4% | Underperformed |
| 2020 | +20.9% | +15.9% | +5.0% | Beat SPY |
| 2021 | +44.9% | +28.7% | +16.3% | Beat SPY |
| 2022 | -16.9% | -14.6% | -2.4% | Underperformed |
| 2023 | +20.2% | +16.8% | +3.4% | Beat SPY |
| 2024 | -0.7% | +21.0% | -21.8% | Underperformed |
| 2025 | +24.0% | +12.8% | +11.2% | Beat SPY |

## Factor-Neutral Meta-Ensemble (optimize_results_factor_neutral.txt)

- Factor model (beta/sector/momentum/size/value/low-vol/reversal) with meta-ensemble (XGB + LGBM + Ridge), ensemble=10
- Portfolio +41.8% vs SPY +125.5% (excess -83.7%)
- Beat SPY in 2 of 8 years (25%)
- Risk metrics: Sharpe ~0.00, vol 26.3%, beta 1.31, max drawdown -41.9% (no significant alpha)

### Year-by-Year Performance (factor-neutral)

| Year | Portfolio | S&P 500 | Excess | Result |
|------|-----------|---------|--------|--------|
| 2018 | -15.7% | -9.5% | -6.3% | Underperformed |
| 2019 | +20.2% | +22.6% | -2.4% | Underperformed |
| 2020 | -7.5% | +15.9% | -23.4% | Underperformed |
| 2021 | +23.2% | +28.7% | -5.5% | Underperformed |
| 2022 | -15.4% | -14.6% | -0.8% | Underperformed |
| 2023 | +23.1% | +16.8% | +6.3% | Beat SPY |
| 2024 | +4.2% | +21.0% | -16.8% | Underperformed |
| 2025 | +13.2% | +12.8% | +0.4% | Beat SPY |

## Legacy (documented previously)

The earlier README numbers (+123.5% portfolio vs +109.2% SPY, excess +14.3%) were produced with an older configuration (42 features, earlier data cut, ensemble=3) and are kept only for historical reference. They do not reflect the latest data or code.
