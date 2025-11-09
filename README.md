Computational Finance — Automated Trading Strategies

This repository contains the work for the “B404B Computational Finance” course at the University of Tübingen.
Our project focuses on developing, implementing, and evaluating algorithmic trading strategies based on historical stock data using Python and NumPy.
The final submission date for this project was June 29, 2025.

⸻

Table of Contents
	1.	Setup and Data Import￼
	2.	Definition of Trading Signals￼
	3.	Computation of Signals and Portfolio Positions￼
	4.	Statistical Analysis￼
	5.	Graphical Results￼
	6.	Additional Considerations￼

⸻

Setup and Data Import

To run the notebooks, you need a Python environment with the following packages installed:# Computational Finance
pip install numpy pandas matplotlib scipy pandas_datareader

The project uses historical stock price data (closing prices and returns) from Yahoo Finance.
All data is automatically downloaded and cleaned within the notebooks.
The datasets serve as the foundation for testing and evaluating the trading strategies.

⸻

Definition of Trading Signals

We developed and analyzed three distinct trading signals, each representing a different market perspective:

Signal 1 – Momentum Crossover

A combination of short-term and long-term moving averages to identify momentum shifts in price trends.

Signal 2 – Mean Reversion (RSI-Based)

Uses the Relative Strength Index (RSI) to detect overbought and oversold conditions, triggering contrarian trades.

Signal 3 – Volatility Breakout

Implements a dynamic volatility channel that generates buy/sell signals when price movements exceed statistical thresholds.

Each signal was parameterized, tested, and calibrated using in-sample data, then validated on out-of-sample periods.

⸻

Computation of Signals and Portfolio Positions
	•	Signals are computed using pure NumPy functions (no rolling or built-in smoothing from Pandas).
	•	The computed signals define stock positions (+1 for long, 0 for neutral, –1 for short).
	•	The resulting time series of positions is combined with returns to generate strategy returns.
	•	All reusable functions (returns, volatility, signal generation, evaluation) are stored in module.py for cleaner structure and reproducibility.

⸻

Statistical Analysis

The performance of each trading strategy is evaluated using several financial metrics, including:
	•	Cumulative returns
	•	Volatility and drawdowns
	•	Sharpe ratio
	•	Jensen’s alpha (for benchmark-adjusted performance)

These statistics allow a comprehensive comparison between the individual signals and a passive benchmark strategy such as the S&P 500.

⸻

Graphical Results

The notebooks visualize the strategies through:
	•	Price and signal overlay plots
	•	Portfolio performance curves
	•	Rolling Sharpe ratios and return distributions
	•	Comparative charts between strategies and benchmarks

These visualizations make it easy to interpret the effectiveness of each signal and its robustness across different market regimes.

⸻

Additional Considerations

In the final discussion, we explore:
	•	The influence of market volatility on signal stability
	•	Transaction costs and their impact on net returns
	•	The economic intuition behind each strategy’s performance
	•	How combining multiple signals could potentially improve diversification

⸻

References and Acknowledgements

This project was completed as part of the Computational Finance (B404B) course at the University of Tübingen under the supervision of Dr. Thomas Schön.
We thank our instructors and peers for their valuable input and feedback throughout the project.