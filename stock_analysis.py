# stock_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime

# Example: Apple stock
stock_symbol = "AAPL"
start_date = "2018-01-01"
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

# Fetch stock data
data = yf.download(stock_symbol, start=start_date, end=end_date)

# Calculate Moving Averages
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# Plot
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label="Closing Price")
plt.plot(data['MA20'], label="20-day MA")
plt.plot(data['MA50'], label="50-day MA")
plt.legend()
plt.show()
