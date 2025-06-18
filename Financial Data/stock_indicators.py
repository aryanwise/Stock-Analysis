import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load the dataset (replace 'meta_stock_data.csv' with your actual file)
try:
    df = pd.read_csv("HistoricalData_META.csv")
    df["Date"] = pd.to_datetime(df["Date"])  # Convert to datetime if needed
    df.set_index("Date", inplace=True)  # Set date as index
    print("Dataset loaded successfully!")
    print(df.head())
except FileNotFoundError:
    print("Error: File not found. Please check the filename and path.")
except Exception as e:
    print(f"An error occurred: {str(e)}")


# Simple Moving Average (SMA)
def calculate_sma(data, window=20):
    """
    Calculate Simple Moving Average (SMA)
    :param data: pandas Series with price data
    :param window: rolling window size
    :return: pandas Series with SMA values
    """
    return data.rolling(window=window).mean()


# Exponential Moving Average (EMA)
def calculate_ema(data, window=20):
    """
    Calculate Exponential Moving Average (EMA)
    :param data: pandas Series with price data
    :param window: rolling window size
    :return: pandas Series with EMA values
    """
    return data.ewm(span=window, adjust=False).mean()


# MACD (Moving Average Convergence Divergence)
def calculate_macd(data, fast=12, slow=26, signal=9):
    """
    Calculate MACD indicator
    :param data: pandas Series with price data
    :param fast: fast EMA period
    :param slow: slow EMA period
    :param signal: signal line period
    :return: DataFrame with MACD line and signal line
    """
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    return pd.DataFrame({"MACD": macd_line, "Signal": signal_line})


# Fibonacci Retracement Levels
def calculate_fibonacci_levels(high, low):
    """
    Calculate Fibonacci retracement levels
    :param high: highest price in the period
    :param low: lowest price in the period
    :return: dictionary with Fibonacci levels
    """
    difference = high - low
    return {
        "0%": high,
        "23.6%": high - difference * 0.236,
        "38.2%": high - difference * 0.382,
        "50%": high - difference * 0.5,
        "61.8%": high - difference * 0.618,
        "100%": low,
    }


# Stochastic Oscillator
def calculate_stochastic_oscillator(high, low, close, window=14, smooth_k=3):
    """
    Calculate Stochastic Oscillator
    :param high: high prices
    :param low: low prices
    :param close: closing prices
    :param window: lookback period
    :param smooth_k: smoothing period for %K
    :return: DataFrame with %K and %D lines
    """
    lowest_low = low.rolling(window=window).min()
    highest_high = high.rolling(window=window).max()

    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    k_smoothed = k.rolling(window=smooth_k).mean()  # %K line
    d = k_smoothed.rolling(window=3).mean()  # %D line (signal line)

    return pd.DataFrame({"%K": k_smoothed, "%D": d})


# Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    Calculate Bollinger Bands
    :param data: price data
    :param window: moving average window
    :param num_std: number of standard deviations for bands
    :return: DataFrame with upper, middle, lower bands
    """
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()

    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)

    return pd.DataFrame(
        {"Upper Band": upper_band, "Middle Band": sma, "Lower Band": lower_band}
    )


# Relative Strength Index (RSI)
def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index (RSI)
    :param data: price data
    :param window: lookback period
    :return: pandas Series with RSI values
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# Average Directional Index (ADX)
def calculate_adx(high, low, close, window=14):
    """
    Calculate Average Directional Index (ADX)
    :param high: high prices
    :param low: low prices
    :param close: closing prices
    :param window: lookback period
    :return: pandas Series with ADX values
    """
    # Calculate +DM and -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)

    # Smooth the values
    atr = true_range.rolling(window=window).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=window).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=window).mean() / atr)

    # Calculate DX and ADX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=window).mean()

    return adx


# Standard Deviation Indicator
def calculate_std_dev(data, window=20):
    """
    Calculate Standard Deviation
    :param data: price data
    :param window: rolling window size
    :return: pandas Series with standard deviation values
    """
    return data.rolling(window=window).std()


# Main calculation function
def calculate_technical_indicators(data):
    """
    Calculate all technical indicators for the dataset
    :param data: DataFrame with stock data (must contain 'High', 'Low', 'Close' columns)
    :return: DataFrame with all indicators added as columns
    """
    try:
        # Make a copy to avoid modifying original data
        df = data.copy()

        # Calculate indicators
        df["SMA_20"] = calculate_sma(df["Close"], 20)
        df["EMA_20"] = calculate_ema(df["Close"], 20)

        macd = calculate_macd(df["Close"])
        df["MACD"] = macd["MACD"]
        df["MACD_Signal"] = macd["Signal"]

        # Fibonacci levels (we'll calculate for the entire period)
        fib_levels = calculate_fibonacci_levels(df["High"].max(), df["Low"].min())
        print("\nFibonacci Retracement Levels:")
        for level, value in fib_levels.items():
            print(f"{level}: {value:.2f}")

        stoch = calculate_stochastic_oscillator(df["High"], df["Low"], df["Close"])
        df["Stoch_%K"] = stoch["%K"]
        df["Stoch_%D"] = stoch["%D"]

        bb = calculate_bollinger_bands(df["Close"])
        df["BB_Upper"] = bb["Upper Band"]
        df["BB_Middle"] = bb["Middle Band"]
        df["BB_Lower"] = bb["Lower Band"]

        df["RSI_14"] = calculate_rsi(df["Close"])
        df["ADX_14"] = calculate_adx(df["High"], df["Low"], df["Close"])
        df["Std_Dev_20"] = calculate_std_dev(df["Close"])

        print("\nTechnical indicators calculated successfully!")
        return df

    except KeyError as e:
        print(f"Error: Missing required column in dataset - {str(e)}")
        return None
    except Exception as e:
        print(f"An error occurred during calculations: {str(e)}")
        return None


# Execute the calculations if the dataset was loaded successfully
if "df" in locals():
    df_with_indicators = calculate_technical_indicators(df)

    # Display the results
    if df_with_indicators is not None:
        print("\nFirst 5 rows with indicators:")
        print(df_with_indicators.head())

        # Plot some indicators (optional)
        plt.figure(figsize=(12, 8))

        # Price with SMA and EMA
        plt.subplot(2, 2, 1)
        plt.plot(df_with_indicators["Close"], label="Close Price")
        plt.plot(df_with_indicators["SMA_20"], label="SMA 20")
        plt.plot(df_with_indicators["EMA_20"], label="EMA 20")
        plt.title("Price with SMA and EMA")
        plt.legend()

        # MACD
        plt.subplot(2, 2, 2)
        plt.plot(df_with_indicators["MACD"], label="MACD")
        plt.plot(df_with_indicators["MACD_Signal"], label="Signal Line")
        plt.title("MACD Indicator")
        plt.legend()

        # RSI
        plt.subplot(2, 2, 3)
        plt.plot(df_with_indicators["RSI_14"], label="RSI 14")
        plt.axhline(70, color="r", linestyle="--")
        plt.axhline(30, color="g", linestyle="--")
        plt.title("RSI Indicator")
        plt.legend()

        # Bollinger Bands
        plt.subplot(2, 2, 4)
        plt.plot(df_with_indicators["Close"], label="Close Price")
        plt.plot(df_with_indicators["BB_Upper"], label="Upper Band")
        plt.plot(df_with_indicators["BB_Middle"], label="Middle Band")
        plt.plot(df_with_indicators["BB_Lower"], label="Lower Band")
        plt.title("Bollinger Bands")
        plt.legend()

        plt.tight_layout()
        plt.show()
