import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import joblib

from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model

# Custom modules
from src.features import add_technical_indicators
from src.preprocessing import create_sequences

plt.style.use("fivethirtyeight")
app = Flask(__name__)

# --- Load model & scaler with absolute paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "stock_prediction_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

# Load model (TensorFlow-CPU recommended in requirements.txt)
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURES = ['Open','High','Low','Close','Volume','MA20','MA50','MACD','RSI']
WINDOW_SIZE = 60

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock', 'GOOGL')

        # OPTIMIZATION: Use 2022 instead of 2015 to save memory on Render Free Tier
        start = dt.datetime(2022, 1, 1)
        end = dt.datetime.now()

        # Fetch and process data
        df = yf.download(stock, start=start, end=end, multi_level_index=False)
        if df.empty:
            return render_template("index.html", error="No data found for this ticker.")
            
        df = df.reset_index()
        df = df[['Date','Open','High','Low','Close','Volume']]
        df = add_technical_indicators(df)
        data_desc = df.describe()

        # Predict
        scaled_data = scaler.transform(df[FEATURES])
        X, y = create_sequences(scaled_data, df['Trend'].values, WINDOW_SIZE)
        y_pred_prob = model.predict(X)
        y_pred = (y_pred_prob.flatten() > 0.5).astype(int)

        # Plotting
        if not os.path.exists(os.path.join(BASE_DIR, "static")):
            os.makedirs(os.path.join(BASE_DIR, "static"))

        # Plot EMA
        plt.figure(figsize=(10,5))
        plt.plot(df.Close, label="Close")
        plt.plot(df.Close.ewm(span=20).mean(), label="EMA 20")
        plt.legend()
        plt.savefig(os.path.join(BASE_DIR, "static", "ema_20_50.png"))
        plt.close()

        # Plot Prediction
        plt.figure(figsize=(10,5))
        plt.plot(y[-100:], label="Actual")
        plt.plot(y_pred[-100:], label="Predicted")
        plt.legend()
        plt.savefig(os.path.join(BASE_DIR, "static", "stock_prediction.png"))
        plt.close()

        csv_path = f"static/{stock}_dataset.csv"
        df.to_csv(os.path.join(BASE_DIR, csv_path), index=False)

        return render_template(
            "index.html",
            stock=stock,
            plot_path_ema_20_50=True,
            plot_path_prediction=True,
            data_desc=data_desc.to_html(classes="table table-sm table-striped"),
            dataset_link=csv_path
        )

    return render_template("index.html")

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(BASE_DIR, "static", filename), as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
