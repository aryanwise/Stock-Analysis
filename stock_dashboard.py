import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import ta
from datetime import datetime


# Load stock data from CSV
def load_stock_data(ticker):
    try:
        file_path = f"Stock-Analysis/Financial Data/HistoricalData_{ticker}.csv"
        data = pd.read_csv(file_path)

        data["Date"] = pd.to_datetime(data["Date"])

        # Data Cleaning
        numeric_cols = ["Open", "High", "Low", "Close/Last", "Volume"]
        for col in numeric_cols:
            if col in data.columns:
                # Remove dollar signs and commas and cast as float
                data[col] = data[col].replace("[\$,]", "", regex=True).astype(float)

        # Close/Last -> Close
        data.rename(columns={"Close/Last": "Close"}, inplace=True)

        return data.sort_values("Date")

    except FileNotFoundError:
        st.error(f"Data file not found for {ticker}")
        return None
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return None


# Calculate basic metrics from the stock data
def calculate_metrics(data):
    try:
        last_close = float(data["Close"].iloc[-1])
        prev_close = float(data["Close"].iloc[0])
        change = last_close - prev_close
        pct_change = (change / prev_close) * 100 if prev_close != 0 else 0
        high = float(data["High"].max())
        low = float(data["Low"].min())
        volume = float(data["Volume"].sum())
        return last_close, change, pct_change, high, low, volume
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return 0, 0, 0, 0, 0, 0


# Technical indicators
def add_technical_indicators(data):
    data["SMA_20"] = ta.trend.sma_indicator(data["Close"], window=20)
    data["EMA_20"] = ta.trend.ema_indicator(data["Close"], window=20)
    return data


# Creating Dashboard
# Set up Streamlit page layout
st.set_page_config(layout="wide")
st.title("Stock Analysis Dashboard")

# Define available tickers
tickers = [
    "JNJ",
    "TSM",
    "WMT",
    "ORCL",
    "AAPL",
    "NKE",
    "V",
    "BABA",
    "JPM",
    "UNH",
    "MA",
    "TSLA",
    "NVDA",
    "NFLX",
    "CMCSA",
    "MSFT",
    "AMZN",
    "TM",
    "GOOGL",
    "XOM",
    "ASML",
    "DIS",
    "BAC",
    "ADBE",
    "KO",
    "META",
]

# 2A: SIDEBAR PARAMETERS ############
st.sidebar.header("Analysis Parameters")
selected_ticker = st.sidebar.selectbox("Select Ticker", tickers)
time_period = st.sidebar.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "All"])
chart_type = st.sidebar.selectbox("Chart Type", ["Candlestick", "Line"])
indicators = st.sidebar.multiselect("Technical Indicators", ["SMA 20", "EMA 20"])

# 2B: MAIN CONTENT AREA ############
if st.sidebar.button("Analyze"):
    # Load and process data
    data = load_stock_data(selected_ticker)
    if data is not None:
        # Filter data based on selected time period
        if time_period != "All":
            periods = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
            cutoff_date = datetime.now() - pd.Timedelta(days=periods[time_period])
            data = data[data["Date"] >= cutoff_date]

        data = add_technical_indicators(data)
        last_close, change, pct_change, high, low, volume = calculate_metrics(data)

        # Display metrics
        st.metric(
            label=f"{selected_ticker} Last Price",
            value=f"{last_close:.2f} USD",
            delta=f"{change:.2f} ({pct_change:.2f}%)",
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("High", f"{high:.2f} USD")
        col2.metric("Low", f"{low:.2f} USD")
        col3.metric("Volume", f"{volume:,}")

        # Create chart
        fig = go.Figure()

        if chart_type == "Candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=data["Date"],
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                    name="Price",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data["Close"],
                    name="Price",
                    line=dict(color="blue"),
                )
            )

        # Add indicators
        if "SMA 20" in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data["SMA_20"],
                    name="SMA 20",
                    line=dict(color="orange", dash="dot"),
                )
            )

        if "EMA 20" in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data["EMA_20"],
                    name="EMA 20",
                    line=dict(color="green", dash="dot"),
                )
            )

        fig.update_layout(
            title=f"{selected_ticker} {time_period} Price Analysis",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=600,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show data tables
        st.subheader("Historical Data")
        st.dataframe(
            data[["Date", "Open", "High", "Low", "Close", "Volume"]].sort_values(
                "Date", ascending=False
            )
        )

        st.subheader("Technical Indicators")
        st.dataframe(
            data[["Date", "SMA_20", "EMA_20"]].sort_values("Date", ascending=False)
        )

# 2C: SIDEBAR STOCK METRICS ############
st.sidebar.header("Stock Metrics")
for ticker in tickers[:5]:  # Show first 5 for space
    data = load_stock_data(ticker)
    if data is not None:
        last_close = data["Close"].iloc[-1]
        prev_close = data["Close"].iloc[0]
        pct_change = ((last_close - prev_close) / prev_close) * 100
        st.sidebar.metric(ticker, f"{last_close:.2f}", f"{pct_change:.2f}%")

# Sidebar information
st.sidebar.subheader("About")
st.sidebar.info(
    "This dashboard analyzes historical stock data with technical indicators."
)
