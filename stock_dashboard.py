# stock_dashboard.py
# Stock Market Trend Analysis Dashboard
# Author: Harsh Kumar

import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime

st.set_page_config(page_title="Stock Market Trend Analysis", layout="wide")
st.title("📈 Stock Market Trend Analysis Dashboard")

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Stock Selection")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY):", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.date(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
future_days = st.sidebar.slider("Days to Predict", 7, 60, 30)

# -------------------------------
# Fetch Data
# -------------------------------
st.write(f"Fetching stock data for **{stock_symbol}** from {start_date} to {end_date}...")
data = yf.download(stock_symbol, start=start_date, end=end_date)

if data.empty:
    st.error("No data found for this stock symbol. Try another one.")
else:
    # -------------------------------
    # Moving Averages
    # -------------------------------
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()

    st.subheader("Stock Price with Moving Averages")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(data['Close'], label="Closing Price")
    ax.plot(data['MA20'], label="20-day MA")
    ax.plot(data['MA50'], label="50-day MA")
    ax.set_title(f"{stock_symbol} Stock Price with Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # Predictive Model (Linear Regression)
    # -------------------------------
    data = data.dropna()
    data['Days'] = np.arange(len(data))  # X-axis: day number
    X = data[['Days']]
    y = data['Close']

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(data), len(data) + future_days).reshape(-1, 1)
    future_predictions = model.predict(future_X)

    st.subheader(f"Predicted Prices for Next {future_days} Days")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(data['Days'], data['Close'], label="Historical Prices")
    ax.plot(future_X, future_predictions, label="Predicted Prices", linestyle="--")
    ax.set_title(f"{stock_symbol} Stock Price Prediction")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # -------------------------------
    # Data Preview
    # -------------------------------
    st.subheader("Data Preview (Last 5 Rows)")
    st.write(data.tail())
