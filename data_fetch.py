import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/sample_data.csv")

print("First 5 rows of dataset:")
print(df.head())

# Plot close price trend
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Close"], label="Close Price", color="blue")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Price Trend")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
