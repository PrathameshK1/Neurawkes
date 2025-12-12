## Paper outline (draft)

### Title
Neural Hawkes for Macro Shock Propagation in Monthly Rates and Retail Price Series

### Abstract (1 paragraph)
- Convert monthly macro/retail-price series into shock events.\n- Fit a multivariate marked Neural Hawkes model.\n- Evaluate out-of-sample event prediction (log-likelihood, calibration, horizon AUC).\n- Compare to discrete-time baselines.\n
### 1. Introduction
- Motivation: shock clustering, contagion across macro variables.\n- Why point processes vs standard time-series models.\n
### 2. Data
- Dataset description, missingness (`-1.0`).\n- Train/val/test split.\n
### 3. Event construction
- Transform: log-diff / pct-change.\n- Train-only z-score.\n- Thresholding into up/down shocks.\n- Marks.\n
### 4. Models
- Neural Hawkes (continuous-time LSTM).\n- Baselines:\n  - Poisson iid\n  - Hawkes-like AR(1) Poisson\n  - VAR(1) threshold proxy\n  - Multi-label logistic / MLP\n
### 5. Evaluation
- Held-out log-likelihood.\n- Count calibration.\n- Horizon AUC (H=1,3 months).\n
### 6. Results
- Main table (test metrics).\n- Error analysis by event type frequency.\n- Regime discussion (2022–2023 shift).\n
### 7. Discussion / limitations
- Small sample size, monthly frequency.\n- Sensitivity to threshold.\n- Recommendations for extensions (higher-frequency series, longer history).\n
### 8. Reproducibility
- Commands:\n  - `python -m nhp_torch.cli.train`\n  - `python -m nhp_torch.cli.evaluate`\n  - `python -m nhp_torch.cli.report`\n

