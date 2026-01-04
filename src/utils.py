import matplotlib.pyplot as plt

def plot_price_ma(df, ticker="Stock"):
    plt.figure(figsize=(12,5))
    plt.plot(df['Close'], label='Close')
    plt.plot(df['MA20'], label='MA20')
    plt.plot(df['MA50'], label='MA50')
    plt.legend()
    plt.title(f"{ticker} Price & Moving Averages")
    plt.show()
