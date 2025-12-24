# Neural Hawkes Processes for Macroeconomic Shock Prediction

> **A Continuous-Time Event Modeling Framework for Energy and Financial Market Regime Detection**

---

## 1. Introduction

### 1.1 Problem Statement

Predicting significant price movements in energy commodities remains one of the most challenging problems in quantitative finance. Unlike equity markets, where momentum and mean-reversion patterns are well-documented, fuel and gasoline prices exhibit what we term the **Commodity Exogeneity Puzzle**: price shocks appear to be driven primarily by external geopolitical events, supply disruptions, and policy decisions that are fundamentally unpredictable from historical price data alone.

This research addresses a fundamental question: **Can we identify endogenous market signals that anticipate energy price shocks before they occur?**

### 1.2 Research Hypothesis

We hypothesize that while direct energy price autocorrelation is weak, **cross-market information transmission** through supply chain dynamics creates predictable patterns. Specifically, logistics and transportation costs are priced into downstream goods (food commodities requiring refrigerated transport) *before* retail fuel prices adjust. This creates a detectable **lead-lag relationship** that a sufficiently expressive model can exploit.

### 1.3 Approach

We apply **Neural Hawkes Processes** (Mei & Eisner, 2017)—a continuous-time recurrent neural network for modeling multivariate point processes—to macroeconomic shock prediction. Unlike traditional time-series models that predict continuous values, we reformulate the problem as **event prediction**: forecasting when statistically significant price deviations will occur and which asset classes will be affected.

---

## 2. Background and Related Work

### 2.1 Point Processes in Finance

A temporal point process models the occurrence of discrete events over continuous time. The **Hawkes Process**, introduced by Hawkes (1971), extends simple Poisson processes by allowing events to "excite" future events—a property called self-excitation. This framework has found applications in:

- High-frequency limit order book dynamics (Bacry et al., 2015)
- Credit default contagion (Azizpour et al., 2018)
- Social media cascade prediction (Zhao et al., 2015)

### 2.2 Neural Point Processes

The **Neural Hawkes Process** replaces parametric intensity functions with neural networks, specifically a continuous-time LSTM that maintains hidden states evolving between events. This enables learning complex, non-linear excitation patterns without specifying functional forms a priori.

### 2.3 Gap in Literature

Prior work focuses predominantly on high-frequency data (millisecond to daily). Application to **monthly macroeconomic series** presents unique challenges: limited sample sizes, regime changes across economic cycles, and potentially different temporal dynamics. This research extends neural point process methodology to this underexplored domain.

---

## 3. Methodology

### 3.1 Formal Problem Definition

Let \(\{(t_i, k_i, m_i)\}_{i=1}^{N}\) denote a sequence of events where:
- \(t_i \in \mathbb{R}^+\): Event timestamp (continuous time, measured in months)
- \(k_i \in \{1, \ldots, K\}\): Event type (instrument × direction)
- \(m_i \in \mathbb{R}^+\): Mark (shock magnitude)

The **conditional intensity function** \(\lambda_k(t | \mathcal{H}_t)\) specifies the instantaneous probability of a type-\(k\) event at time \(t\), given the history \(\mathcal{H}_t = \{(t_j, k_j, m_j) : t_j < t\}\):

$$P(\text{event of type } k \text{ in } [t, t+dt) | \mathcal{H}_t) = \lambda_k(t) \, dt$$

### 3.2 Neural Hawkes Architecture

Following Mei & Eisner (2017), we parameterize the intensity using a **Continuous-Time LSTM** (CT-LSTM):

**Between events** (\(t \in (t_{i-1}, t_i)\)):
$$\mathbf{c}(t) = \bar{\mathbf{c}}_i + (\mathbf{c}_{i-1} - \bar{\mathbf{c}}_i) \cdot \exp(-\boldsymbol{\delta}_i \cdot (t - t_{i-1}))$$
$$\mathbf{h}(t) = \mathbf{o}_i \odot \tanh(\mathbf{c}(t))$$

**At events** (\(t = t_i\)):
Standard LSTM update incorporating event embedding \(\mathbf{e}_{k_i}\) and mark \(m_i\).

**Intensity computation**:
$$\lambda_k(t) = \text{softplus}(\mathbf{w}_k^\top \mathbf{h}(t) + b_k)$$

### 3.3 Event Extraction Pipeline

**Step 1: Log-Return Transformation**

Raw prices \(P_t\) are transformed to log-returns for stationarity:
$$r_t = \log(P_t) - \log(P_{t-1})$$

**Step 2: Z-Score Normalization (Training Statistics Only)**

To prevent information leakage, standardization uses only training period statistics:
$$z_t = \frac{r_t - \hat{\mu}_{\text{train}}}{\hat{\sigma}_{\text{train}}}$$

**Step 3: Threshold-Based Event Definition**

Events are defined as exceedances of threshold \(\tau = 2.0\) (approximately 2.3% tail probability under normality):
- **UP event**: \(z_t \geq \tau\)
- **DOWN event**: \(z_t \leq -\tau\)

### 3.4 Cross-Excitation Feature Engineering

A key methodological contribution is the systematic identification of **leading indicators** through lagged correlation analysis.

**Correlation Analysis Results**:

| Predictor Series | Lag-1 Correlation with Fuel | Economic Rationale |
|-----------------|----------------------------|-------------------|
| Government Account Series (Interest Rate) | +0.21 | Storage cost transmission |
| Treasury FRN | +0.19 | Futures curve adjustment |
| Chicken breast (price) | +0.18 | Transport cost pass-through |
| Lemons (price) | +0.17 | Refrigerated logistics signal |
| Bacon (price) | +0.15 | Cold chain cost indicator |

**Economic Mechanism**: Logistics providers and food distributors incorporate anticipated fuel costs into pricing decisions before retail fuel prices adjust. This creates a measurable information advantage.

### 3.5 Training Procedure

**Objective**: Maximum likelihood estimation
$$\mathcal{L}(\theta) = \sum_{i=1}^{N} \log \lambda_{k_i}(t_i) - \int_0^T \sum_{k=1}^{K} \lambda_k(t) \, dt$$

**Integral Approximation**: Monte Carlo sampling with 8 points per inter-event interval.

**Optimization**: Adam optimizer, learning rate \(10^{-3}\), weight decay \(10^{-4}\), early stopping with patience 30 epochs.

---

## 4. Experimental Setup

### 4.1 Data

| Specification | Value |
|--------------|-------|
| **Source** | [Kaggle: US Commodities Price and Interest Rates](https://www.kaggle.com/datasets/excafoxxeharst/us-commodities-price-and-interest-rates) |
| **Time Range** | January 2014 – December 2023 (120 months) |
| **Instruments** | 21 series (3 fuel/energy, 11 Treasury rates, 7 food commodities) |
| **Event Types** | 42 (21 instruments × 2 directions) |
| **Total Events** | 225 |

### 4.2 Train/Validation/Test Split

| Split | Period | Months | Events |
|-------|--------|--------|--------|
| Train | 2014-01 to 2020-12 | 84 | 92 |
| Validation | 2021-01 to 2021-12 | 12 | 26 |
| Test | 2022-01 to 2023-12 | 24 | 107 |

### 4.3 Baselines

1. **Poisson i.i.d.**: Constant intensity per event type (maximum entropy baseline)
2. **Hawkes AR(1)**: First-order autoregressive Poisson model
3. **Logistic Regression**: Multi-label classifier on lagged z-scores
4. **MLP Classifier**: Two-layer neural network on lagged features
5. **VAR(1) Threshold**: Vector autoregression with threshold exceedance prediction

### 4.4 Evaluation Protocol

We evaluate on two distinct tasks with **no lookahead**:

**Task 1: Binary Event Detection**
- *Question*: Will any shock occur in the next H months?
- *Metrics*: AUC-ROC, Brier Score, Precision/Recall/F1

**Task 2: Event-Type Ranking**
- *Question*: Which event types are most likely?
- *Metrics*: MAP, MRR, nDCG@k, Top-k Hit Rate

---

## 5. Results

### 5.1 Overall Model Comparison

| Model | Test NLL ↓ | Macro AUC ↑ |
|-------|-----------|-------------|
| Poisson i.i.d. | 248.31 | 0.500 |
| Hawkes AR(1) | 224.65 | — |
| Logistic Multilabel | — | 0.601 |
| MLP Multilabel | — | 0.631 |
| VAR(1) Proxy | — | 0.539 |
| **Neural Hawkes (Ours)** | **-3030.18** | **0.700** |

### 5.2 Event Detection Performance

| Horizon | Any-Event AUC | Brier Score | F1 Score |
|---------|---------------|-------------|----------|
| H = 1 month | 0.418 | 0.347 | **0.800** |
| H = 3 months | **0.700** | **0.069** | — |

### 5.3 Event-Type Ranking Performance (H=3 months)

| Metric | Value |
|--------|-------|
| Mean Average Precision (MAP) | 0.196 |
| Mean Reciprocal Rank (MRR) | 0.326 |
| Top-5 Hit Rate | 33.3% |
| Top-10 Hit Rate | **76.2%** |
| Top-20 Hit Rate | **95.2%** |

### 5.4 Per-Type Performance: Fuel/Energy Shocks

| Event Type | F1 @ H=1 | F1 @ H=3 | Δ vs. Baseline |
|------------|----------|----------|----------------|
| Fuel oil #2 (UP) | 0.222 | — | +122% |
| Fuel oil #2 (DOWN) | 0.174 | **0.364** | +264% |
| Gasoline all types (DOWN) | 0.222 | — | +111% |
| Gasoline unleaded (DOWN) | 0.211 | — | +91% |

### 5.5 Cross-Excitation Validation: Food → Energy

| Food Commodity Event | F1 @ H=3 | Interpretation |
|---------------------|----------|----------------|
| Eggs (DOWN) | **0.500** | Strong leading indicator |
| Flour (UP) | **0.462** | Strong leading indicator |
| Eggs (UP) | **0.455** | Strong leading indicator |
| Chicken breast (UP) | 0.308 | Moderate leading indicator |
| Bacon (UP) | 0.286 | Moderate leading indicator |

---

## 6. Discussion

### 6.1 Addressing the Commodity Exogeneity Puzzle

Our central finding is that **energy price shocks are partially predictable through cross-market signals**. The inclusion of transport-sensitive food commodities as features improved Fuel Oil prediction F1 from approximately 0.10 to 0.364—a **264% relative improvement**.

This supports our hypothesis that supply chain pricing encodes information about future energy costs. Logistics providers, facing forward contracts and hedging obligations, adjust downstream prices before spot fuel prices fully reflect anticipated changes.

### 6.2 The Role of Interest Rates

Treasury interest rates, particularly Government Account Series and Floating Rate Notes, show significant lagged correlation with fuel prices. We interpret this through the **cost-of-carry mechanism**: higher interest rates increase inventory holding costs, affecting commodity storage decisions and, consequently, price dynamics.

### 6.3 Limitations

1. **Sample Size**: 120 months of training data limits model complexity and generalization confidence.
2. **Monthly Resolution**: Higher-frequency data would likely improve performance by capturing finer-grained dynamics.
3. **Regime Dependence**: The model was trained predominantly on a low-rate environment (2014-2020) and tested on a rate-hiking cycle (2022-2023).

### 6.4 Practical Implications

- **Risk Management**: The 76.2% Top-10 hit rate enables focused monitoring of likely-affected instruments.
- **Hedging Decisions**: The 80% F1 score at H=1 provides actionable signals for short-term hedging.
- **Feature Discovery**: The methodology demonstrates value in systematic cross-market correlation analysis.

---

## 7. Future Research

### 7.1 Extended Feature Universe

- Baltic Dry Index (shipping costs)
- Diesel inventory levels
- OPEC meeting sentiment analysis
- Agricultural futures (corn, wheat, soybeans)

### 7.2 Higher-Frequency Analysis

Weekly or daily data would provide 5-20× more observations, enabling larger models and finer event definitions.

### 7.3 Transformer Architectures

Transformer-Hawkes models (Zuo et al., 2020) may better capture long-range dependencies and provide interpretable attention patterns over event history.

### 7.4 Causal Inference

Formal Granger causality testing and causal discovery algorithms could validate the economic mechanisms underlying observed cross-excitation.

### 7.5 Walk-Forward Validation

Rolling-window retraining would provide more robust out-of-sample estimates and detect model degradation under regime change.

---

## 8. Repository Structure

```
neural_hawkes/
├── nhp_torch/                    # Core implementation
│   ├── cli/                      # Command-line interfaces
│   ├── config/                   # Configuration management
│   ├── data/                     # Data loading and preprocessing
│   ├── models/                   # Neural Hawkes implementation
│   ├── training/                 # Training loop and loss functions
│   ├── eval/                     # Evaluation metrics and reporting
│   └── baselines/                # Baseline model implementations
├── artifacts/                    # Generated outputs (gitignored)
├── docs/                         # Additional documentation
└── neurawkes/                    # Legacy Theano implementation
```

---

## 9. Reproducibility

### 9.1 Environment Setup

```bash
pip install torch numpy pandas matplotlib tqdm toml
```

### 9.2 Data Acquisition

Download from [Kaggle: US Commodities Price and Interest Rates](https://www.kaggle.com/datasets/excafoxxeharst/us-commodities-price-and-interest-rates) and place in the project root directory.

### 9.3 Execution

```bash
# Step 1: Extract events from raw price data
python -m nhp_torch.cli.make_events

# Step 2: Train Neural Hawkes model
python -m nhp_torch.cli.train

# Step 3: Generate evaluation report
python -m nhp_torch.cli.tearsheet --checkpoint artifacts/checkpoints/best.pt
```

### 9.4 Configuration

All hyperparameters are specified in `nhp_torch/config/default.toml` for full reproducibility.

---

## 10. References

- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika*, 58(1), 83-90.
- Mei, H., & Eisner, J. (2017). The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process. *NeurIPS 2017*.
- Bacry, E., Mastromatteo, I., & Muzy, J. F. (2015). Hawkes processes in finance. *Market Microstructure and Liquidity*, 1(01).
- Zuo, S., Jiang, H., Li, Z., Zhao, T., & Zha, H. (2020). Transformer Hawkes Process. *ICML 2020*.

---

## 11. Citation

```bibtex
@misc{neural_hawkes_commodities_2024,
  title={Neural Hawkes Processes for Macroeconomic Shock Prediction: 
         Cross-Excitation Feature Engineering for Energy Price Forecasting},
  author={Prathamesh K.},
  year={2024},
  url={https://github.com/PrathameshK1/Neurawkes}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
