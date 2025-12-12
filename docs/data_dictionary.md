## Data dictionary (CSV)

File: `full_dataset_by_year_2014_2023_75_percent_float_casted_date_indexed.csv`

### Index / frequency

- **Frequency**: monthly
- **Date column**: `Date YY-MM` (values are `YYYY-MM`)

### Missing values

- `-1.0` indicates **missing** (treated as NaN in the pipeline).
- If a series is missing at month \(t\) or \(t-1\), its month-to-month change is treated as missing and produces **no event**.

### Column groups

1) **US Treasury average interest rate series** (marketable and non-marketable categories, plus totals)\n\n2) **Retail prices (USD/unit)** for selected items:\n- food staples (eggs, milk, flour, rice, etc.)\n- meats (multiple beef cuts, chicken)\n- energy prices (fuel oil #2, gasoline)\n+
### Pipeline transformations

Configured in `nhp_torch/config/default.toml`:

- **transform**: `logdiff` (default) or `pct`
- **z-score normalization**: mean/std fit on **train split only**


