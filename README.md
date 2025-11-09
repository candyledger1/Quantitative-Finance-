# Computational Finance — Automated Trading Strategies

This repository contains the work for the **“B404B Computational Finance”** course at the **University of Tübingen**.  
Our project focuses on developing, implementing, and evaluating **algorithmic trading strategies** based on historical stock data using **Python and NumPy**.  
The final submission date for this project was **June 29, 2025**.

---
## Assessment Notebook Overview 

## Table of Contents 
1. [Data Acquisition](#data-acquisition)  
2. [Definition of Signals](#definition-of-signals)  
3. [Computation of Signals and Resulting Positions](#computation-of-signals-and-resulting-positions)  
4. [Statistics of the Strategy](#statistics-of-the-strategy)  
5. [Graphs of Strategy](#graphs-of-strategy)

---

The project uses historical stock price data (closing prices and returns) from **Yahoo Finance**.  
All data is automatically downloaded and cleaned within the notebooks.  
The datasets serve as the foundation for testing and evaluating the trading strategies.

---

## Definition of Trading Signals

We developed and analyzed **three distinct trading signals**, each representing a different market perspective:

### **Signal 1 – Momentum Crossover**
A combination of **short-term and long-term moving averages** to identify momentum shifts in price trends.

### **Signal 2 – Mean Reversion (RSI-Based)**
Uses the **Relative Strength Index (RSI)** to detect overbought and oversold conditions, triggering contrarian trades.

### **Signal 3 – Volatility Breakout**
Implements a **dynamic volatility channel** that generates buy/sell signals when price movements exceed statistical thresholds.

Each signal was parameterized, tested, and calibrated using in-sample data, then validated on out-of-sample periods.

---

## Computation of Signals and Portfolio Positions

- Signals are computed using **pure NumPy** functions (no rolling or built-in smoothing from Pandas).  
- The computed signals define **stock positions** (+1 for long, 0 for neutral, –1 for short).  
- The resulting time series of positions is combined with returns to generate **strategy returns**.  
- All reusable functions (returns, volatility, signal generation, evaluation) are stored in **`module.py`** for cleaner structure and reproducibility.

---

## Statistical Analysis

The performance of each trading strategy is evaluated using several financial metrics, including:

- **Cumulative returns**  
- **Volatility and drawdowns**  
- **Sharpe ratio**  
- **Jensen’s alpha** (for benchmark-adjusted performance)  

These statistics allow a comprehensive comparison between the individual signals and a passive benchmark strategy such as the **S&P 500**.

---

## Graphical Results

The notebooks visualize the strategies through:

- Price and signal overlay plots  
- Portfolio performance curves  
- Rolling Sharpe ratios and return distributions  
- Comparative charts between strategies and benchmarks  

These visualizations make it easy to interpret the effectiveness of each signal and its robustness across different market regimes.

---

## Additional Considerations

In the final discussion, we explore:

- The influence of **market volatility** on signal stability  
- **Transaction costs** and their impact on net returns  
- The **economic intuition** behind each strategy’s performance  
- How combining multiple signals could potentially improve diversification  

---


## Research Notebook Overview

The **`research_notebook.ipynb`** expands on the assessment notebook by conducting a **full empirical analysis** of the implemented trading strategies.  
It evaluates the robustness, parameter sensitivity, and economic interpretation of the signals across multiple assets and time periods.

---

## 📘 Table of Contents
1. [Data Acquisition and Preprocessing](#data-acquisition-and-preprocessing)  
2. [Exploratory Stock Analysis](#exploratory-stock-analysis)  
3. [Signal 1 — Moving Average Crossover with MACD Indicator](#signal-1--moving-average-crossover-with-macd-indicator)  
4. [Signal 2 — RSI with Bollinger Bands](#signal-2--rsi-with-bollinger-bands)  
5. [Signal 3 — Breakout Momentum Strategy](#signal-3--breakout-momentum-strategy)  
6. [Parameter Optimization](#parameter-optimization)  
7. [Backtesting and Evaluation](#backtesting-and-evaluation)  
8. [Performance Visualization](#performance-visualization)  
9. [Economic Interpretation and Discussion](#economic-interpretation-and-discussion)

---

## Data Acquisition and Preprocessing

- Historical **daily stock data** (2011–2025) is downloaded directly from **Yahoo Finance** for:  
  **Tesla (TSLA), Apple (AAPL), AMD, and the S&P 500 (^GSPC)**.  
- The data is split into:
  - **In-sample (2011–2019)** for training and parameter tuning  
  - **Out-of-sample (2020–2025)** for validation and robustness testing  
- Basic price visualizations reveal general market movement patterns and volatility differences between the assets.

---

## Exploratory Stock Analysis

Before implementing signals, the notebook plots normalized price trends for all assets to illustrate:  
- Distinct volatility profiles (e.g., Tesla’s high risk–high reward nature)  
- Market cycles and drawdowns across different time periods  
- Comparison between growth and benchmark indices (S&P 500)

These plots provide a **contextual foundation** for why each trading strategy is designed the way it is.

---

## Signal 1 — Moving Average Crossover with MACD Indicator

- Combines **trend-following (Moving Average)** and **momentum (MACD)** indicators.  
- Both sub-signals are merged using an **AND logic**, so a trade is executed only when both trend and momentum align.  
- A **grid search** systematically optimizes parameters such as short/long MA lengths and MACD windows.  
- Strategy is evaluated across all four assets to identify the best-performing combination.  
- The combined signal demonstrates **superior Sharpe ratio** compared to using MA or MACD alone.

---

## Signal 2 — RSI with Bollinger Bands

- Integrates **mean reversion** (via RSI) with **volatility filtering** (via Bollinger Bands).  
- Detects overbought/oversold conditions and filters noise by volatility levels.  
- Parameter grids are defined for RSI lookback, threshold bounds, and Bollinger window/width.  
- Results show strong performance on AMD, with improved **risk-adjusted returns** compared to benchmarks.  

---

## Signal 3 — Breakout Momentum Strategy

- Captures price breakouts from volatility channels or recent highs.  
- Focuses on momentum continuation patterns following significant market moves.  
- This signal acts as a **trend accelerator**, complementing the smoother mean reversion signals.  

---

## 🔍 Parameter Optimization

- Each signal is optimized via **grid search** using in-sample data.  
- The notebook reports best-performing parameter combinations per asset and strategy.  
- Optimization criteria include **Sharpe ratio**, **maximum drawdown**, and **volatility**.  

---

## Backtesting and Evaluation

- For each signal and ticker, strategy returns are compared to a **Buy-and-Hold benchmark**.  
- Metrics include:
  - Cumulative return  
  - Sharpe ratio  
  - Volatility  
  - Maximum drawdown  
  - Jensen’s alpha  

All statistics are summarized in **formatted tables** for direct comparison across assets.

---

## Performance Visualization

- Plots of buy/sell signals overlaid on price data for each stock.  
- Visual separation between in-sample and out-of-sample periods.  
- Comparative charts display:
  - Strategy vs. benchmark returns  
  - Rolling Sharpe ratios  
  - Signal entry/exit timing  
  - Portfolio growth trajectories  

---

## Economic Interpretation and Discussion

The notebook concludes by interpreting results economically:
- **Trend-following signals** perform better in persistent bull markets (e.g., Tesla, Apple).  
- **Mean-reversion signals** excel in sideways or corrective phases.  
- The combination of momentum and volatility filters provides a balanced, risk-efficient performance.  
- Practical aspects such as **transaction costs** and **market noise filtering** are discussed as real-world considerations.

---

 *This research notebook thus represents the empirical backbone of the project, testing, optimizing, and interpreting multiple algorithmic trading signals using real financial data in Python.*

## References and Acknowledgements

This project was completed as part of the **Computational Finance (B404B)** course at the **University of Tübingen** under the supervision of **Dr. Thomas Schön**.  
We thank our instructors and peers for their valuable input and feedback throughout the project.