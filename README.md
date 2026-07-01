# Explainable Anomaly Detection in Financial Time-Series
### LSTM Autoencoder + SHAP Attribution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Detecting anomalies in financial time-series is a core problem in **quantitative
research**, **risk management**, and **algorithmic trading**. Equally critical —
especially under regulatory frameworks such as **IFRS 9** and **SR 11-7** — is
the ability to *explain* why a period is flagged as anomalous.

This project implements a complete, transparent anomaly-detection pipeline:

1. **Feature engineering** — nine stationary, interpretable features from raw OHLCV data
2. **LSTM Autoencoder** — unsupervised anomaly scoring via reconstruction error
3. **SHAP attribution** — feature-level explanations for every flagged event

The pipeline is trained on S&P 500 data (2005–2019) and tested on genuinely
out-of-sample events including the **COVID crash (Feb–Mar 2020)** and the
**2022 rate-hike selloff** — both detected without any labelled training signal.

---

## Results

| Event | Detected | Primary SHAP Driver |
|---|---|---|
| Lehman collapse (Sep 2008) | ✅ | `vol_5d`, `log_return` |
| Flash Crash (May 2010) | ✅ | `hl_range`, `vol_zscore` |
| China selloff (Aug 2015) | ✅ | `vol_ratio`, `mom_5d` |
| Q4 2018 selloff (Dec 2018) | ✅ | `rsi`, `mom_20d` |
| COVID crash (Mar 2020) ⭐ OOS | ✅ | `vol_ratio`, `mom_20d` |
| Rate-hike selloff (Jun 2022) ⭐ OOS | ✅ | `rsi`, `mom_20d` |

False-positive rate on normal market periods: **< 5%**

---

## Relevance to Regulated Finance

Under **SR 11-7** (Federal Reserve model risk guidance) and **IFRS 9**
(expected credit loss modelling), financial institutions must be able to
explain model outputs — a raw anomaly score does not satisfy this bar.

SHAP attribution addresses this directly:
- Every flagged event comes with a quantified, feature-level explanation
- Attributions are stable across model re-initialisations (verified)
- The explanation format is familiar to quant analysts and model validators

---

## Project Structure

```
financial-anomaly-detection-xai/
├── config.py                          # Central configuration
├── requirements.txt
├── src/
│   ├── data.py                        # Feature engineering + DataLoaders
│   ├── model.py                       # LSTM Autoencoder (Encoder + Decoder)
│   ├── train.py                       # Training loop + threshold calibration
│   ├── explain.py                     # SHAP attribution layer
│   └── visualize.py                   # All plotting utilities
├── notebooks/
│   └── anomaly_detection_walkthrough.py   # Full end-to-end walkthrough
└── results/                           # Saved model + output charts
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/Subhashree1202/financial-anomaly-detection-xai.git
cd financial-anomaly-detection-xai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (CPU ~5 min, GPU ~1 min)
jupyter notebook notebooks/anomaly_detection_walkthrough.ipynb
```

Alternatively, convert the `.py` notebook to `.ipynb` with Jupytext:
```bash
jupytext --to notebook notebooks/anomaly_detection_walkthrough.py
```

---

## Features Engineered

| Feature | Formula / Description |
|---|---|
| `log_return` | ln(Pₜ / Pₜ₋₁) |
| `vol_5d` | Rolling 5-day return std |
| `vol_20d` | Rolling 20-day return std |
| `vol_ratio` | vol_5d / vol_20d — short/long regime indicator |
| `mom_5d` | 5-day price momentum |
| `mom_20d` | 20-day price momentum |
| `hl_range` | (High − Low) / Close — intraday stress proxy |
| `vol_zscore` | (Volume − 20d mean) / 20d std |
| `rsi` | 14-day Relative Strength Index |

All features are stationary and scale-invariant. The scaler is fit on
training data only — no look-ahead leakage.

---

## Model Architecture

```
Input  (B, W=30, F=9)
   │
   ▼
Encoder:  2-layer LSTM  ──►  last hidden state  ──►  Linear(64→16)
                                                          │ latent z (B, 16)
Decoder:  repeat z  ──►  2-layer LSTM  ──►  Linear(64→9)
   │
   ▼
Output (B, W=30, F=9)    ← reconstruction

Anomaly score = MSE(input, reconstruction) per window
Threshold     = µ_train + 3σ_train  (calibrated on training set only)
```

---

## Explainability

SHAP `GradientExplainer` is applied to the *encoder*, attributing the
L2 norm of the latent vector back to each (time-step × feature) input cell.

Per-feature importance is aggregated as **mean |SHAP value|** across all
time steps and all anomaly windows, producing a chart comparable to the
standard SHAP summary plot.

---

## Roadmap

- [ ] Extend to credit-default swap spreads and bond indices
- [ ] Add LSTM attention weights as a complementary explanation method
- [ ] Connect to Project B (vehicle sensor data) for cross-domain comparison
- [ ] Wrap as a REST API for integration into a model-risk dashboard

---

## Related Work

- **ESANN 2026:** Evaluation of Rashomon Sets for Stable Model Explanations
- **ContEx 2025:** Stability of Model Explanations in Prototype-based Classification

Both publications address explanation stability — a prerequisite for
trustworthy SHAP deployment in regulated environments.

---

## License

MIT — free to use, adapt, and build on.
