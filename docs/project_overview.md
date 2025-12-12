## Project overview

This repo is a research pipeline to evaluate **Neural Hawkes Processes** (continuous-time LSTM point process) on a monthly macro/price dataset:

- input: `full_dataset_by_year_2014_2023_75_percent_float_casted_date_indexed.csv`
- output: event-based predictive metrics + baselines + plots

The project converts monthly time series into a **multivariate marked point process** by thresholding standardized changes into **shock events** and then fits a PyTorch Neural Hawkes model to predict future event intensity.

### Why Hawkes here?

The Hawkes framing models **clustering** and **cross-excitation** of shocks:

- energy shocks ↔ food shocks
- rate regime shocks ↔ price shocks

### Split protocol

- Train: 2014–2020
- Validation: 2021
- Test: 2022–2023

### Key artifacts

- model checkpoint: `artifacts/checkpoints/best.pt`
- evaluation metrics: `artifacts/eval/metrics.json`
- markdown summary: `artifacts/eval/summary.md`
- plot: `artifacts/eval/top_event_types.png`


