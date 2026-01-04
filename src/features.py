import pandas as pd

def add_technical_indicators(df):
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() /
                                    -df['Close'].diff().clip(upper=0).rolling(14).mean())))
    df['Trend'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df.dropna()
