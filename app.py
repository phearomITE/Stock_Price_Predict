import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import yfinance as yf
from flask import Flask, render_template, request, send_file, jsonify
import time
import threading
import warnings
warnings.filterwarnings('ignore')

# 🔥 LSTM imports
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

plt.style.use("fivethirtyeight")
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LIVE_PRICES = {}
LIVE_PRICES_LOCK = threading.Lock()

# 🔥 ULTIMATE yfinance ANTI-BLOCK (yfinance 1.0)
def safe_yf_download(ticker, period="2y", retries=5):
    """🚀 5x fallback methods for Yahoo Finance blocks"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for attempt in range(retries):
        try:
            print(f"📡 [{ticker}] Attempt {attempt+1}/{retries}")
            
            # Method 1: Ticker.history() - MOST RELIABLE
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, timeout=30, 
                             prepost=False, repair=True)
            if not df.empty and len(df) > 50:
                print(f"✅ [{ticker}] Method 1 SUCCESS: {len(df)} rows")
                return df
            
            # Method 2: download() with auto-adjust
            print(f"🔄 [{ticker}] Method 2...")
            df = yf.download(ticker, period=period, progress=False, 
                           threads=False, auto_adjust=True, repair=True)
            if not df.empty and len(df) > 50:
                print(f"✅ [{ticker}] Method 2 SUCCESS: {len(df)} rows")
                return df
            
            # Method 3: Longer period fallback
            print(f"🔄 [{ticker}] Method 3 (longer period)...")
            df = yf.download(ticker, period="max", progress=False, 
                           threads=False, auto_adjust=True)
            if not df.empty and len(df) > 100:
                df = df.tail(500)  # Take recent data
                print(f"✅ [{ticker}] Method 3 SUCCESS: {len(df)} rows")
                return df
            
            time.sleep(2 ** attempt)  # Backoff
            
        except Exception as e:
            print(f"❌ [{ticker}] Attempt {attempt+1}: {str(e)[:100]}")
            time.sleep(3)
    
    print(f"💥 [{ticker}] ALL METHODS FAILED")
    return pd.DataFrame()

def fix_yfinance_data(df):
    """🔥 Clean ANY yfinance format"""
    if df.empty:
        return None
        
    print(f"🔍 Raw shape: {df.shape}, Columns: {list(df.columns)}")
    
    # Reset index if Date is index
    if 'Date' not in df.columns:
        df = df.reset_index()
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Find columns by pattern matching
    date_col = next((col for col in df.columns if 'date' in str(col).lower()), 'Date')
    open_col = next((col for col in df.columns if 'open' in str(col).lower()), 'Open')
    high_col = next((col for col in df.columns if 'high' in str(col).lower()), 'High')
    low_col = next((col for col in df.columns if 'low' in str(col).lower()), 'Low')
    close_col = next((col for col in df.columns if 'close' in str(col).lower()), 'Close')
    vol_col = next((col for col in df.columns if 'volume' in str(col).lower()), 'Volume')
    
    required_cols = [date_col, open_col, high_col, low_col, close_col]
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Missing columns: {required_cols}")
        return None
    
    df_clean = pd.DataFrame({
        'Date': df[date_col],
        'Open': df[open_col],
        'High': df[high_col],
        'Low': df[low_col],
        'Close': df[close_col],
        'Volume': df.get(vol_col, pd.Series(0, index=df.index))
    }).dropna()
    
    print(f"✅ CLEAN DATA: {df_clean.shape}")
    return df_clean

def get_live_price(ticker):
    """🚀 Live price with multiple fallbacks"""
    try:
        stock = yf.Ticker(ticker)
        
        # Try 1h data first
        hist = stock.history(period="5d", interval="1h", timeout=20)
        if hist.empty:
            hist = stock.history(period="1d", interval="5m", timeout=20)
        
        if not hist.empty:
            latest = hist.iloc[-1]
            info = stock.info
            prev_close = info.get('regularMarketPreviousClose', latest['Close'])
            change_pct = ((latest['Close'] - prev_close) / prev_close) * 100
            
            return {
                'price': round(float(latest['Close']), 2),
                'change': round(float(latest['Close'] - prev_close), 2),
                'change_pct': round(change_pct, 2),
                'volume': int(latest['Volume']),
                'high': round(float(latest['High']), 2),
                'low': round(float(latest['Low']), 2),
                'time': dt.datetime.now().strftime("%H:%M:%S")
            }
    except:
        pass
    return None

def predict_future(df, days):
    """🧠 LSTM Price Prediction"""
    try:
        if len(df) < 60:
            return [float(df['Close'].iloc[-1])] * days
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, batch_size=32, epochs=25, verbose=0)
        
        # Predict future
        last_60 = scaled_data[-60:]
        future_preds = []
        for _ in range(days):
            X_pred = last_60.reshape(1, 60, 1)
            pred = model.predict(X_pred, verbose=0)
            future_preds.append(pred[0, 0])
            last_60 = np.append(last_60[1:], pred)
        
        future_preds = np.array(future_preds).reshape(-1, 1)
        return scaler.inverse_transform(future_preds).flatten().tolist()
        
    except Exception as e:
        print(f"❌ LSTM Error: {e}")
        return [float(df['Close'].iloc[-1])] * days

# 🔥 6 LIVE TICKERS: BTC + 5 Top Stocks
def update_live_prices():
    """🔄 Background live updates - 6 assets total"""
    tickers = ['BTC-USD', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']  # 🔥 6 TICKERS!
    while True:
        with LIVE_PRICES_LOCK:
            for ticker in tickers:
                try:
                    data = get_live_price(ticker)
                    if data:
                        LIVE_PRICES[ticker] = data
                except:
                    pass
        live_count = len([v for v in LIVE_PRICES.values() if v])
        btc_price = LIVE_PRICES.get('BTC-USD', {}).get('price', 'N/A')
        print(f"📈 LIVE ({live_count}/6): BTC ${btc_price} | Stocks OK")
        time.sleep(180)  # 3 minutes

# Start background thread
threading.Thread(target=update_live_prices, daemon=True).start()

@app.route('/')
def index():
    return render_template("index.html", live_prices=LIVE_PRICES)

@app.route('/analyze', methods=['POST'])
def analyze_stock():
    stock = request.form.get('stock', 'AAPL').upper().strip()
    pred_days = request.form.get('pred_days', '1')
    print(f"\n🚀 ANALYSIS + {pred_days}D PREDICTION: {stock}")
    
    df = safe_yf_download(stock)
    if df.empty or len(df) < 50:
        return render_template("index.html", 
                             error=f"No data for {stock}. Try: BTC-USD, MSFT, NVDA, TSLA, AMZN")
    
    df = fix_yfinance_data(df)
    if df is None:
        return render_template("index.html", 
                             error=f"Cannot process {stock} data")
    
    # Technical indicators
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df = df.dropna()
    
    live_data = get_live_price(stock)
    create_pro_charts(df, stock, live_data)
    
    # Predictions
    pred_1d_list = predict_future(df, 1)
    pred_5d_list = predict_future(df, 5)
    pred_7d_list = predict_future(df, 7)
    
    current_price = float(df['Close'].iloc[-1])
    pred_1d, pred_5d, pred_7d = pred_1d_list[0], pred_5d_list[-1], pred_7d_list[-1]
    
    # Trading signals
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    signals, confidence = [], 0
    
    # MA Trend
    if latest['Close'] > latest['MA20'] > latest['MA50']:
        signals.append("🟢 Strong Bullish"); confidence += 25
    elif latest['Close'] < latest['MA20'] < latest['MA50']:
        signals.append("🔴 Strong Bearish"); confidence -= 25
    else:
        signals.append("🟡 Neutral Trend")
    
    # RSI
    if latest['RSI'] < 30: signals.append("🟢 RSI Oversold"); confidence += 20
    elif latest['RSI'] > 70: signals.append("🔴 RSI Overbought"); confidence -= 20
    
    # MACD
    if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
        signals.append("🟢 MACD Golden Cross"); confidence += 15
    
    # Trading logic
    if confidence >= 50: recommendation, signal_class = "🚀 STRONG BUY", "buy-pro"
    elif confidence >= 10: recommendation, signal_class = "✅ BUY", "buy"
    elif confidence <= -50: recommendation, signal_class = "💥 STRONG SELL", "sell-pro"
    elif confidence <= -10: recommendation, signal_class = "❌ SELL", "sell"
    else: recommendation, signal_class = "⏸️ HOLD", "hold"
    
    changes = {
        '1d': ((pred_1d - current_price) / current_price) * 100,
        '5d': ((pred_5d - current_price) / current_price) * 100,
        '7d': ((pred_7d - current_price) / current_price) * 100
    }
    
    message = f"""💼 {stock} PROFESSIONAL ANALYSIS + LSTM PREDICTIONS
━━━━━━━━━━━━━━━━━━━━━━━
🎯 RECOMMENDATION: {recommendation}
📊 CONFIDENCE: {confidence:+.0f}%
💰 CURRENT: ${current_price:.2f}

🔮 LSTM FORECAST:
📅 1 Day:  ${pred_1d:.2f} ({changes['1d']:+.1f}%)
📅 5 Days: ${pred_5d:.2f} ({changes['5d']:+.1f}%)
📅 7 Days: ${pred_7d:.2f} ({changes['7d']:+.1f}%)

📊 TECHNICALS:
RSI: {latest['RSI']:.1f} | MACD: {latest['MACD']:.3f}"""
    
    # Summary table
    summary_table = df[['Close', 'RSI', 'MACD']].tail(5).round(2).to_html(classes="table table-sm table-striped")
    
    # Save CSV
    os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
    csv_path = f"static/{stock}_pro_analysis.csv"
    df.to_csv(os.path.join(BASE_DIR, csv_path), index=False)
    
    return render_template(
        "index.html",
        stock=stock,
        recommendation=recommendation,
        signal_class=signal_class,
        trading_message=message,
        plot_dashboard=True,
        live_data=live_data,
        data_desc=summary_table,
        dataset_link=csv_path,
        confidence=f"{confidence:+.0f}%",
        pred_1d=f"${pred_1d:.2f}",
        pred_5d=f"${pred_5d:.2f}",
        pred_7d=f"${pred_7d:.2f}",
        pred_change_1d=f"{changes['1d']:+.1f}%",
        pred_change_5d=f"{changes['5d']:+.1f}%",
        pred_change_7d=f"{changes['7d']:+.1f}%",
        signals=signals,
        live_prices=LIVE_PRICES,
        current_price=f"${current_price:.2f}",
        error=None
    )

@app.route('/api/live')
def api_live():
    with LIVE_PRICES_LOCK:
        return jsonify({k: v for k, v in LIVE_PRICES.items() if v})

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(BASE_DIR, "static", filename), as_attachment=True)

def create_pro_charts(df, stock, live_data):
    """🎨 Professional charts"""
    plt.ioff()
    os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
    
    fig = plt.figure(figsize=(20, 16))
    
    # Price Chart
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df['Date'], df['Close'], 'b-', linewidth=3, label='Close')
    ax1.plot(df['Date'], df['MA20'], 'orange', linewidth=2.5, label='MA20')
    ax1.plot(df['Date'], df['MA50'], 'g-', linewidth=2.5, label='MA50')
    ax1.plot(df['Date'], df['BB_Upper'], 'gray', '--', alpha=0.7, label='BB Upper')
    ax1.plot(df['Date'], df['BB_Lower'], 'gray', '--', alpha=0.7, label='BB Lower')
    ax1.fill_between(df['Date'], df['BB_Upper'], df['BB_Lower'], alpha=0.1, color='gray')
    
    if live_data:
        ax1.scatter(df['Date'].iloc[-1], live_data['price'], 
                   color='red', s=400, marker='*', zorder=5, label=f'LIVE: ${live_data["price"]}')
    
    ax1.set_title(f'{stock} PROFESSIONAL DASHBOARD + LSTM PREDICTIONS', fontsize=24, fontweight='bold', pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RSI
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(df['Date'], df['RSI'], 'r-', linewidth=2, label='RSI')
    ax2.axhline(70, color='r', linestyle='--', alpha=0.7, label='Overbought')
    ax2.axhline(30, color='g', linestyle='--', alpha=0.7, label='Oversold')
    ax2.set_ylabel('RSI')
    
    # Volume
    ax3 = plt.subplot(3, 1, 3)
    colors = ['green' if c > o else 'red' for c, o in zip(df['Close'], df['Open'])]
    ax3.bar(df['Date'], df['Volume'], color=colors, alpha=0.7)
    ax3.set_ylabel('Volume')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"static/{stock}_pro_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("🚀 ULTIMATE STOCK + CRYPTO DASHBOARD v5.1")
    print("📈 LIVE TRACKING (6 assets): BTC-USD, AAPL, MSFT, GOOGL, TSLA, NVDA")
    print("🎯 Enter ANY ticker: BTC-USD, AAPL, TSLA, ETH-USD, NVDA, etc.")
    print("🔮 1D/5D/7D LSTM AI Predictions + 5 Technical Indicators")
    app.run(host="0.0.0.0", port=5000, debug=True)
