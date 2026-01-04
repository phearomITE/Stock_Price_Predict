import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import joblib

from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model

# Import custom modules from your src directory
from src.features import add_technical_indicators
from src.preprocessing import create_sequences

# Set plot style
plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Use absolute paths to find the models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "stock_prediction_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Verify the file actually exists on the disk
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Looking for model at {MODEL_PATH}")
    print(f"Current Directory contains: {os.listdir(BASE_DIR)}")

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = ['Open','High','Low','Close','Volume','MA20','MA50','MACD','RSI']
WINDOW_SIZE = 60

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock', 'GOOGL')

        start = dt.datetime(2015, 1, 1)
        end = dt.datetime.now()

        # Download stock data
        df = yf.download(stock, start=start, end=end, multi_level_index=False)
        df = df.reset_index()
        df = df[['Date','Open','High','Low','Close','Volume']]

        # Add indicators
        df = add_technical_indicators(df)

        # Descriptive stats
        data_desc = df.describe()

        # Scale features
        scaled_data = scaler.transform(df[FEATURES])

        # Create sequences
        X, y = create_sequences(scaled_data, df['Trend'].values, WINDOW_SIZE)

        # Predict trend
        y_pred_prob = model.predict(X)
        y_pred = (y_pred_prob.flatten() > 0.5).astype(int)

        # Ensure static folder exists for saving plots
        if not os.path.exists(os.path.join(BASE_DIR, "static")):
            os.makedirs(os.path.join(BASE_DIR, "static"))

        # -----------------------------
        # Plot EMA 20 / 50
        # -----------------------------
        ema20 = df.Close.ewm(span=20).mean()
        ema50 = df.Close.ewm(span=50).mean()

        plt.figure(figsize=(12,6))
        plt.plot(df.Close, label="Close")
        plt.plot(ema20, label="EMA 20")
        plt.plot(ema50, label="EMA 50")
        plt.legend()
        plt.title(f"{stock} EMA 20 & 50")
        ema_20_50_path = os.path.join("static", "ema_20_50.png")
        plt.savefig(os.path.join(BASE_DIR, ema_20_50_path))
        plt.close()

        # -----------------------------
        # Plot EMA 100 / 200
        # -----------------------------
        ema100 = df.Close.ewm(span=100).mean()
        ema200 = df.Close.ewm(span=200).mean()

        plt.figure(figsize=(12,6))
        plt.plot(df.Close, label="Close")
        plt.plot(ema100, label="EMA 100")
        plt.plot(ema200, label="EMA 200")
        plt.legend()
        plt.title(f"{stock} EMA 100 & 200")
        ema_100_200_path = os.path.join("static", "ema_100_200.png")
        plt.savefig(os.path.join(BASE_DIR, ema_100_200_path))
        plt.close()

        # -----------------------------
        # Plot Trend Prediction
        # -----------------------------
        plt.figure(figsize=(12,6))
        plt.plot(y[-200:], label="Actual Trend")
        plt.plot(y_pred[-200:], label="Predicted Trend")
        plt.legend()
        plt.title(f"{stock} Predicted vs Actual Trend")
        prediction_path = os.path.join("static", "stock_prediction.png")
        plt.savefig(os.path.join(BASE_DIR, prediction_path))
        plt.close()

        # Save dataset
        csv_path = os.path.join("static", f"{stock}_dataset.csv")
        df.to_csv(os.path.join(BASE_DIR, csv_path), index=False)

        return render_template(
            "index.html",
            plot_path_ema_20_50=True,
            plot_path_ema_100_200=True,
            plot_path_prediction=True,
            data_desc=data_desc.to_html(classes="table table-bordered"),
            dataset_link=csv_path
        )

    return render_template("index.html")

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(BASE_DIR, "static", filename), as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Port set by cloud platform (e.g., Railway)
    app.run(debug=False, host="0.0.0.0", port=port)
