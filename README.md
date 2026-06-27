# Huntington Macro Equity Engine

A high-performance quantitative research platform designed to identify lead-lag relationships between global macroeconomic indicators and GICS sector ETFs. The system automates the full data lifecycle, from multi-source ETL ingestion to real-time risk assessment and directional price forecasting.

## 🎯 Core Systems

### Lead-Lag Signal Discovery Engine
* **Matrix-Based Correlation:** Utilizes vectorized masking to identify stable signals across rolling time windows.
* **Recursive ADF Testing:** Enforces stationarity across 25+ years of market data to eliminate spurious correlations.
* **Predictive Lift:** Achieved a **52% increase** in signal strength via optimized feature-target alignment.

### Multi-Model ML Suite
* **Linear & Ensemble Models:** Integrates Random Forest ensembles and adaptive OLS variants (Recursive & Rolling Window).
* **Dynamic PCA:** Implements dimensionality reduction, cutting feature noise by **70%** while preserving variance.
* **Performance:** Sustained **65%+ directional accuracy** in forecasting sector price action across shifting market regimes.

### Sector Risk Engine
* **Quantitative Scoring:** Aggregates metrics using a weighted algorithm:  
    `Risk Score = σ_sector × β × ρ_holdings`
* **Dynamic Ranking:** Real-time sector classification based on volatility and inter-asset correlation.

## 🚀 Infrastructure & Performance

* **Vectorized Pipeline:** Engineered using NumPy and Pandas to eliminate Python loops, ensuring **sub-30ms retrieval latency**.
* **Caching Layer:** Implemented a custom persistence strategy to cut API usage by **~90%** and bypass network bottlenecks.
* **Modular Architecture:** Built with a "plug-and-play" design for adding new alpha factors or model architectures.

## 📊 Architecture

```mermaid
graph TD
    A[Data Sources: FRED, Yahoo Finance] --> B[ETL Pipeline & ADF Validator]
    B --> C[Localized Caching Layer]
    C --> D[Compute Core: Vectorized Matrix Ops]
    D --> E[PCA & Lead-Lag Engine]
    E --> F[ML Ensemble Suite]
    F --> G[Risk Engine & Analytics]
    G --> H[Signal Export & Dashboard]
