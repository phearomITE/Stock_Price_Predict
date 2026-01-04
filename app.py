import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import joblib

from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model

from src.features import add_technical_indicators
from src.preprocessing import create_sequences

plt.style.use("fivethirtyeight")

app = Flask(__name__)

# -----------------------------
# Load trained model & scaler
# -----------------------------
MODEL_PATH = "models/stock_prediction_model.h5"
SCALER_PATH = "models/scaler.pkl"

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
        plt.title("EMA 20 & 50")
        ema_20_50_path = "static/ema_20_50.png"
        plt.savefig(ema_20_50_path)
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
        plt.title("EMA 100 & 200")
        ema_100_200_path = "static/ema_100_200.png"
        plt.savefig(ema_100_200_path)
        plt.close()

        # -----------------------------
        # Plot Trend Prediction
        # -----------------------------
        plt.figure(figsize=(12,6))
        plt.plot(y[-200:], label="Actual Trend")
        plt.plot(y_pred[-200:], label="Predicted Trend")
        plt.legend()
        plt.title("Predicted vs Actual Trend")
        prediction_path = "static/stock_prediction.png"
        plt.savefig(prediction_path)
        plt.close()

        # Save dataset
        csv_path = f"static/{stock}_dataset.csv"
        df.to_csv(csv_path, index=False)

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
    return send_file(f"static/{filename}", as_attachment=True)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Port set by cloud platform
    app.run(debug=False, host="0.0.0.0", port=port)
