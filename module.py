### IMPORTS

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pandas_datareader import data



def download_stock_price_data(tickers, start_date, end_date):    
    # download the time series of adj. close price 
    # for each of the tickers from Yahoo finance
    # and dataframe with price changes
    #
    # in case you receive an error update your yfinance to the current version
    # and/or perform bug fixing
    
    df_prices = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    prev = df_prices.to_numpy()
    prev = prev / np.insert(prev[:-1,:], 0, np.ones(prev.shape[1]), 0)
    prev[0] = np.ones(prev.shape[1])
    df_price_changes = df_prices.copy(deep=True)
    df_price_changes[:] = prev

    return df_prices, df_price_changes

######## Signal Helper Function ##########################

#Function combine two subsignals into one combined signal
def combine_two_subsignals(signal1, signal2):
    
    signal1 = np.asarray(signal1)
    signal2 = np.asarray(signal2)
    assert signal1.shape == signal2.shape

    combined = np.zeros_like(signal1)
    holding = 0

    for i in range(len(signal1)):
        if holding == 0 and (signal1[i] == 1 or signal2[i] == 1):  #Open position even if only one signal open
            holding = 1
        elif holding == 1 and signal1[i] == 0 and signal2[i] == 0:
            holding = 0
        combined[i] = holding

    return combined

###########################################################

######## Signal 01 ########################################

def moving_average(prices, window_length):
    window_length = int(window_length)
    return np.convolve(prices, np.ones(window_length)/window_length, mode='same')

#Creates a standard MA Crossover signal
def ma_signal(series, short_window, long_window):
    signals = pd.DataFrame(index=series.index)
    signals['signal'] = 0.0

    prices_array = series.to_numpy()
    signals['short_ma'] = moving_average(prices_array, short_window)
    signals['long_ma'] = moving_average(prices_array, long_window)

    valid_range = max(short_window, long_window)
    signals.iloc[valid_range:, signals.columns.get_loc('signal')] = np.where(
        signals.iloc[valid_range:, signals.columns.get_loc('short_ma')] >
        signals.iloc[valid_range:, signals.columns.get_loc('long_ma')],
        1.0, 0.0)

    signals['position_change'] = signals['signal'].diff()
    signals.loc[series.index[0], 'position_change'] = 0

    return signals


#Helper function to compute the exponential moving average
def exponential_moving_average(prices, MACD_window_length):

    ema = np.empty(len(prices))
    ema[:] = np.nan
    alpha = 2 / (MACD_window_length + 1)

    #Find first index
    first_index = np.where(~np.isnan(prices))[0][0]
    start = first_index + MACD_window_length - 1

    if start < len(prices):
        #Compute initial average manually
        initial_sum = 0
        count = 0
        for i in range(start - MACD_window_length + 1, start + 1):
            if not np.isnan(prices[i]):
                initial_sum += prices[i]
                count += 1
        initial_avg = initial_sum / count
        ema[start] = initial_avg

        #Compute EMA
        for i in range(start + 1, len(prices)):
            if not np.isnan(prices[i]):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

    return ema

#Function that generates signal for the MACD
def signal_macd(prices, short_window, long_window, signal_window):
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = 0.0

    price_array = prices.to_numpy()
    signals['short_ema'] = exponential_moving_average(price_array, short_window)
    signals['long_ema'] = exponential_moving_average(price_array, long_window)

    signals['MACD'] = signals['short_ema'] - signals['long_ema']
    signals['signal_line'] = exponential_moving_average(signals['MACD'].to_numpy(), signal_window)

    #Generate signal where MACD > signal line
    valid_range = max(short_window, long_window)
    signals.loc[signals.index[valid_range:], 'signal'] = np.where(
        signals['MACD'][valid_range:] > signals['signal_line'][valid_range:], 1.0, 0.0
    )

    signals['position_change'] = signals['signal'].diff()
    signals.loc[prices.index[0], 'position_change'] = 0

    return signals

#Full implemntation fo first Signal
def signal01(prices, short_ma, long_ma, short_macd, long_macd, signal_window_macd):

    #MA Signal
    ma_sig = ma_signal(prices, short_ma, long_ma)

    #MACD Signal
    macd_sig = signal_macd(prices, short_macd, long_macd, signal_window_macd)

    #Combine Signals
    combined = combine_two_subsignals(ma_sig['signal'], macd_sig['signal'])

    #Return combined signal dataframe
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = combined
    signals['position_change'] = signals['signal'].diff().fillna(0)
    signals['ma_position_change'] = ma_sig['position_change']
    signals['macd_position_change'] = macd_sig['position_change']

    return signals

#######################################################################

############# Signal 02 ##############################################

#Helper function to compute the Relative Strength Index
def compute_rsi(prices, window_length):
    
    prices = np.asarray(prices).flatten()
    deltas = np.diff(prices)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    #Initilize and don't compute the first window_length days avg gain and loss
    avg_gain = np.empty_like(prices, dtype=float)
    avg_loss = np.empty_like(prices, dtype=float)    
    avg_gain[:window_length] = np.nan
    avg_loss[:window_length] = np.nan
    
    #First average gain and loss
    avg_gain[window_length] = gains[:window_length].mean()
    avg_loss[window_length] = losses[:window_length].mean()
    
    #Implement the Wilder smoothing after the first mean computation
    for i in range(window_length + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (window_length - 1) + gains[i - 1]) / window_length
        avg_loss[i] = (avg_loss[i - 1] * (window_length - 1) + losses[i - 1]) / window_length
    
    #Compute index
    rs =  avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    #Let the first window_length entries be 50, so it doens't generate any signal, 50 being way off any threshold
    rsi[:window_length] = 50
    
    return rsi

#Function that generates signal for the RSI
def signal_rsi(prices, rsi_window, lower_rsi_bound, upper_rsi_bound):   
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = 0.0
    
    rsi = compute_rsi(prices, rsi_window)
    
    buy_signal = rsi < lower_rsi_bound
    sell_signal = rsi > upper_rsi_bound

    #Create positions
    position = np.zeros(len(prices), dtype=int)
    holding = 0
    for i in range(len(prices)):
        if holding == 0 and buy_signal[i]:
            holding = 1
        elif holding == 1 and sell_signal[i]:
            holding = 0
        position[i] = holding

    signals['signal'] = position
    signals['position_change'] = signals['signal'].diff().fillna(0)
    signals.iloc[0, signals.columns.get_loc('position_change')] = 0

    return signals

#Helper function to compute Bollinger Bands
def compute_bollinger_bands(prices, window_length, num_std):
    sma = moving_average(prices, window_length)
    
    #Shift window forward and compute std
    std = np.full_like(prices, np.nan)
    for i in range(window_length - 1, len(prices)):
        window = prices[i - window_length + 1:i + 1]
        std[i] = np.std(window, ddof=0)
    
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    
    return sma, upper_band, lower_band

#Function that generates signal for the Bollinger Bands
def signal_bollinger(prices, bollinger_window_length, num_std):
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = 0.0

    prices_array = prices.to_numpy()
    sma, upper_band, lower_band = compute_bollinger_bands(
        prices_array, window_length=bollinger_window_length, num_std=num_std)

    #Check if prices cross back inside lower band
    outside_lower = prices_array < lower_band
    outside_lower_prev = np.roll(outside_lower, 1)
    outside_lower_prev[0] = False
    buy_signal = (outside_lower_prev == True) & (outside_lower == False)

    #Check if prices cross back inside upper band
    outside_upper = prices_array > upper_band
    outside_upper_prev = np.roll(outside_upper, 1)
    outside_upper_prev[0] = False
    sell_signal = (outside_upper_prev == False) & (outside_upper == True)

    #Create positions wiht sell and buy signal
    position = np.zeros(len(prices), dtype=int)
    holding = 0
    for i in range(len(prices)):
        if holding == 0 and buy_signal[i]:
            holding = 1
        elif holding == 1 and sell_signal[i]:
            holding = 0
        position[i] = holding

    signals['signal'] = position
    signals['position_change'] = signals['signal'].diff().fillna(0)
    signals.loc[prices.index[0], 'position_change'] = 0

    return signals

#Full implemntation of second Signal
def signal02(prices, rsi_window_length, lower_rsi_bound, upper_rsi_bound, bollinger_window_length, bollinger_n_stds):

    #RSI Signal
    rsi_sig= signal_rsi(prices, rsi_window_length, lower_rsi_bound, upper_rsi_bound)

    #Bollinger Signal
    bollinger_sig = signal_bollinger(prices, bollinger_window_length, bollinger_n_stds)

    #Combine Signals
    combined = combine_two_subsignals(rsi_sig['signal'], bollinger_sig['signal'])

    #Return combined signal dataframe
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = combined
    signals['position_change'] = signals['signal'].diff().fillna(0)
    signals['rsi_position_change'] = rsi_sig['position_change']
    signals['bollinger_position_change'] = bollinger_sig['position_change']

    return signals

###################################################################


############### Signal 03 #########################################

#Helper function that creates donchian channel high and lows
def donchian_channel(prices, window_length=20):
    
    prices = np.asarray(prices)
    highs = np.full_like(prices, np.nan, dtype=float)
    lows = np.full_like(prices, np.nan, dtype=float)
    
    #Shift window and always exclude current value
    for i in range(window_length, len(prices)):
        highs[i] = np.max(prices[i - window_length:i])
        lows[i] = np.min(prices[i - window_length:i])
    
    return highs, lows

#Function that creates Signal based on donchian channel
def donchian_signals(prices, window_length=20):
    
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = 0.0

    price_array = prices.to_numpy()
    highs, lows = donchian_channel(price_array, window_length)

    #Create buy and sell conditions
    buy_signal = price_array > highs
    sell_signal = price_array < lows

    #Create positions
    position = np.zeros(len(price_array), dtype=float)
    holding = 0
    for i in range(len(prices)):
        if holding == 0 and buy_signal[i]:
            holding = 1
        elif holding == 1 and sell_signal[i]:
            holding = 0
        position[i] = holding

    signals['signal'] = position
    signals['position_change'] = signals['signal'].diff().fillna(0)
    signals.iloc[0, signals.columns.get_loc('position_change')] = 0

    return signals

#Helper function that creates ADX
def compute_adx(prices, window):

    prices = np.asarray(prices)
    
    #Infer directional movement
    dm_pos = []
    dm_neg = []
    for i in range(1, len(prices)):
        price_diff = prices[i] - prices[i - 1]
        if price_diff > 0:
            dm_pos.append(price_diff)
            dm_neg.append(0)
        elif price_diff < 0:
            dm_pos.append(0)
            dm_neg.append(-price_diff)
        else:
            dm_pos.append(0)
            dm_neg.append(0)

    #Compute true ranges 
    true_ranges = []
    for i in range(1, len(prices)):
        high_low = prices[i] - prices[i - 1]
        high_close = abs(prices[i] - prices[i - 1])
        low_close = abs(prices[i] - prices[i - 1])
        true_ranges.append(max(high_low, high_close, low_close))

    #Wilder's smoothing to create directional index
    atr = [np.mean(true_ranges[:window])]
    di_pos = [np.mean(dm_pos[:window])]
    di_neg = [np.mean(dm_neg[:window])]   
    for i in range(window, len(dm_pos)):
        atr.append((atr[-1] * (window - 1) + true_ranges[i]) / window)
        di_pos.append((di_pos[-1] * (window - 1) + dm_pos[i]) / window)
        di_neg.append((di_neg[-1] * (window - 1) + dm_neg[i]) / window)
    di_pos = np.array(di_pos)
    di_neg = np.array(di_neg)   
    dx = np.abs((di_pos - di_neg) / (di_pos + di_neg)) * 100

    #Smooth DX to get ADX
    adx = np.full(len(prices), 1.0) #Initalize with neutral values for the 2* windowlength warmup phase

    #Smooth DX to get ADX
    adx[window*2-1] = np.mean(dx[:window])
    for i in range(window*2, len(dx)):
        adx[i] = (adx[i - 1] * (window - 1) + dx[i]) / window
        
    return adx.flatten()

#Full implemntation of third Signal with
def signal03(prices, adx_window_length, adx_threshhold, donchian_window_length):
    
    signals = pd.DataFrame(index=prices.index)
    signals['signal'] = 0.0
    
    #Compute ADX and Donchian Channel
    adx = compute_adx(prices, adx_window_length)
    donchian_sig = donchian_signals(prices, donchian_window_length)
    donchian_sig_array = np.asarray(donchian_sig['signal'])

    #Custom tradig logic: since adx only detects trends but not in which direction -> combine_two_subsignals() function doesn't work here
    position = np.zeros(len(prices), dtype=float)
    holding = 0
    for i in range(len(prices)):
        if np.isnan(adx[i]):
            continue
        if holding == 0 and donchian_sig_array[i] == 1 and adx[i] > adx_threshhold:
            holding = 1
        elif holding == 1 and donchian_sig_array[i] == 0 and adx[i] > adx_threshhold:
            holding = 0
        position[i] = holding

    signals['signal'] = position
    signals['position_change'] = signals['signal'].diff().fillna(0)
    signals['donchian_position_change'] = donchian_sig['position_change']
    signals.iloc[0, signals.columns.get_loc('position_change')] = 0

    return signals

############################################################################


########################### Optimizer ######################################

#Gridsearch function that can take any Signal as input and optimizes based on the target metric passed
def gridsearch_strategy(price, param_grid, signal_fn, metric):
    
    #Create list where every possible parameter combination from the passed grid is unpacked 
    keys = list(param_grid.keys())
    shapes = [len(param_grid[k]) for k in keys]
    total_combos = np.prod(shapes)
    all_params = []
    for i in range(total_combos):
        idxs = []
        rem = i
        for s in reversed(shapes):
            idxs.append(rem % s)
            rem //= s
        idxs = idxs[::-1]
        combo = {keys[j]: param_grid[keys[j]][idxs[j]] for j in range(len(keys))}
        all_params.append(combo)

    #Initalize storage
    best_params = None
    results = []

    #Loop over all param combinations and try signal
    for param in all_params:
        signals = signal_fn(price, **param)
        signal = signals['signal']
        metrics = backtest_strategy(price, signal)
        result_row = {
        **param,
        'cumret': metrics['Strategy Cumulative Return'],
        'b&h cumret': metrics['BuyHold Cumulative Return'],
        'sharpe': metrics['Strategy Sharpe'],
        'b&h sharpe': metrics['BuyHold Sharpe']}
        results.append(result_row)

    #Convert all results to DataFrame
    df_results = pd.DataFrame(results)
    ascending = True if metric in ['max_dd', 'volatility'] else False
    df_sorted = df_results.sort_values(by=metric, ascending=ascending).reset_index(drop=True)
    
    #Retrieve best parameter combo by sorting by target metric
    best_row = df_sorted.iloc[0]
    best_params = {k: int(best_row[k]) for k in param_grid.keys()}
    best_score = best_row[metric]

    return best_params, best_score, df_sorted

#################################################################



############### Metrics functions ###############################

#Returns array of daily returns based on trading signal
def strategy_returns(prices, signals):
    positions = np.asarray(signals)
    positions[0] = 0 #No signal in period one
    prices = np.asarray(prices)
    daily_returns = (prices[1:] / prices[:-1]) - 1
    daily_strategy_returns = positions[:-1] * daily_returns
    return daily_strategy_returns

#Returns total cumulative return over the period
def cumulative_return(returns):
    return np.prod(1 + returns) - 1

#Returns total cumulative return series over time
def cumulative_return_series(returns):
    return np.cumprod(1 + returns)

#Calculates annualized Sharpe-Ratio
def sharpe(returns, periods_per_year=252):
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0
    return (mean / std) * np.sqrt(periods_per_year)

#Calculates annualized volatility
def volatility(returns, periods_per_year=252):
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)

#Calculates maximum drawdown from a returns series
def max_drawdown(returns):
    cum_returns = cumulative_return_series(returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = 1 - cum_returns / peak
    return np.max(drawdown)

#Calculates BH Sharpe-Ratio
def buy_and_hold_sharpe(prices, periods_per_year=252):    
    prices = prices.to_numpy()
    daily_returns = (prices[1:] / prices[:-1]) 
    mean = np.mean(daily_returns)
    std = np.std(daily_returns, ddof=1)
    if std == 0:
        return 0
    return (mean / std) * np.sqrt(periods_per_year)

#Calculates modilgiani ratio
def modigliani_ratio(sharpe_ratio, benchmark_std, risk_free_rate):
    return risk_free_rate + sharpe_ratio * benchmark_std

#Calculates sortino ratio
def sortino_ratio(returns, risk_free_rate, periods_per_year=252):

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf

    downside_returns = np.where(excess_returns < 0, excess_returns, 0)
    downside_std = np.sqrt(np.mean(downside_returns**2))
    if downside_std == 0:
        return np.nan

    #Annualize
    mean_excess = np.mean(excess_returns)
    
    return (mean_excess / downside_std) * np.sqrt(periods_per_year)

#Calculates jensens alpha
def compute_jensens_alpha(portfolio_returns, benchmark_returns, rf_rate_annual, trading_days=252):

    #daily avg
    avg_port = np.sum(portfolio_returns) / len(portfolio_returns)
    avg_bench = np.sum(benchmark_returns) / len(benchmark_returns)

    #compute cov
    cov = np.sum((portfolio_returns - avg_port) * (benchmark_returns - avg_bench)) / (len(benchmark_returns) - 1)
    var_bench = np.sum((benchmark_returns - avg_bench) ** 2) / (len(benchmark_returns) - 1)
    cov_ann = cov * trading_days
    var_ann = var_bench * trading_days

    #Compute beta and alpha
    beta = cov_ann / var_ann
    alpha = (avg_port * trading_days) - (rf_rate_annual + beta * ((avg_bench * trading_days) - rf_rate_annual))

    return alpha, beta

#Computes jensens alpha
def treynor_ratio(returns, beta, risk_free_rate, periods_per_year=252):
    if beta == 0:
        return np.nan  #dont devide by zero

    #Mean and annulize
    mean_daily_return = np.sum(returns) / len(returns)
    annual_return = mean_daily_return * periods_per_year

    return (annual_return - risk_free_rate) / beta

#Computes information ratio
def information_ratio(portfolio_returns, benchmark_returns, periods_per_year=252):
    
    #Calculate excess returns and mean and std
    excess_returns = portfolio_returns - benchmark_returns
    mean_excess = np.mean(excess_returns)
    tracking_error = np.std(excess_returns, ddof=1)
    
    if tracking_error == 0:
        return np.nan    
 
    #Annualize
    return (mean_excess / tracking_error) * np.sqrt(periods_per_year)

#Computes max drawdown
def max_drawdown(returns):
    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdown = 1 - cum_returns / peak
    return np.max(drawdown)

#Computes the calmar ratio for a given window
def calmar_ratio_from_window(window_returns, periods_per_year=252):
   
    #Calculate cumulative return
    cumulative = np.cumprod(1 + window_returns)

    #Track the running maximum
    peak = np.maximum.accumulate(cumulative)

    #Compute drawdowns
    drawdown = 1 - cumulative / peak

    #Max drawdown
    max_drawdown = np.max(drawdown)

    #Mena and annulize
    mean_return = np.mean(window_returns)
    annualized_return = mean_return * periods_per_year

    if max_drawdown == 0:
        return np.nan

    return annualized_return / max_drawdown

#################################################################


#################### Backtesting #################################    

#Function computes all relevant metrics based on signals provided and compares them to the respective BH metrics
def backtest_strategy(prices, signals, periods_per_year=252):
    
    #Strategy and BH Returns
    strat_returns = strategy_returns(prices, signals)
    prices = prices.to_numpy()
    bh_returns = prices[1:] / prices[:-1] - 1

    #Cumulative return series
    strat_cumret_series = cumulative_return_series(strat_returns)
    bh_cumret_series = cumulative_return_series(bh_returns)

    results = {
        #Cumulative returns
        'Strategy Cumulative Return': strat_cumret_series[-1] - 1,
        'BuyHold Cumulative Return': bh_cumret_series[-1] - 1,

        #Sharpe ratios
        'Strategy Sharpe': sharpe(strat_returns, periods_per_year),
        'BuyHold Sharpe': sharpe(bh_returns, periods_per_year),
        'Sharpe Delta': sharpe(strat_returns, periods_per_year) - sharpe(bh_returns, periods_per_year),

        #Max Drawdowns
        'Strategy Max Drawdown': max_drawdown(strat_returns),
        'BuyHold Max Drawdown': max_drawdown(bh_returns),

        #Volatility
        'Strategy Volatility': volatility(strat_returns, periods_per_year),
        'BuyHold Volatility': volatility(bh_returns, periods_per_year),

        #Return series for plotting
        'Strategy Daily Returns': strat_returns,
        'BuyHold Daily Returns': bh_returns,
        'Strategy CumRet Series': strat_cumret_series,
        'BuyHold CumRet Series': bh_cumret_series,
    }

    return results

##################################################

####### Plotting #################################  
    
def plot_buy_and_sell_signals(position_change, prices, ticker, sig_name):
    
    #Convert to np
    position_change = np.asarray(position_change)
    dates = prices.index.to_numpy()
    prices = np.asarray(prices[ticker])

    #Plot
    plt.figure(figsize=(15, 4))
    plt.plot(dates, prices, label=f"{ticker} Price", color='black', alpha=0.8)

    #Buy and sell points
    buy = position_change == 1
    sell = position_change == -1
    plt.plot(dates[buy], prices[buy], 'g^', label='Buy', markersize=8)
    plt.plot(dates[sell], prices[sell], 'rv', label='Sell', markersize=8)

    #Formatting
    plt.title(f"{ticker} - Buy/Sell Signals from: {sig_name}")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#This function computes and plot a rolling performance metric for a given metric e.g. sharpe
def rolling_metric_plot(returns, index, window, metric_func, *args, label_name="Rolling Metric", color="blue", **kwargs):
    
    #initilize valeus
    rolling_values = []
    rolling_dates = []

    #Roll window forward
    for i in range(window, len(returns)):
        window_returns = returns[i - window:i]
        metric_value = metric_func(window_returns, *args, **kwargs)
        rolling_values.append(metric_value)
        rolling_dates.append(index[i])

    #Plot
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_dates, rolling_values, label=label_name, color=color)
    plt.title(label_name)
    plt.xlabel("Date")
    plt.ylabel(label_name)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
