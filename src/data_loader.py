import os
import yfinance as yf
import pandas as pd

def download_stock_data(ticker="GOOGL", start="2015-01-01", end="2025-01-01", save_path=None):
    df = yf.download(ticker, start=start, end=end, multi_level_index=False)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.reset_index()
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✓ Stock data saved to: {save_path}")
    return df
