from src.data_loader import download_stock_data
from src.features import add_technical_indicators
from src.preprocessing import scale_features, create_sequences
from src.model import build_bilstm_model, train_model, save_model
from src.evaluation import plot_training_history, evaluate_model, plot_trend_comparison
from src.utils import plot_price_ma
import numpy as np
import joblib
import os

# --------------------------
# Paths
# --------------------------
RAW_DATA_PATH = "./data/raw/stock_data.csv"
SCALER_PATH = "./models/scaler.pkl"
MODEL_PATH = "./models/stock_prediction_model.h5"

# --------------------------
# Step 1: Download Data
# --------------------------
df = download_stock_data(ticker="GOOGL", save_path=RAW_DATA_PATH)

# --------------------------
# Step 2: Add Technical Indicators
# --------------------------
df = add_technical_indicators(df)

# --------------------------
# Step 3: Plot Price & Moving Averages
# --------------------------
plot_price_ma(df, ticker="GOOGL")

# --------------------------
# Step 4: Scale Features & Create Sequences
# --------------------------
features = ['Open','High','Low','Close','Volume','MA20','MA50','MACD','RSI']
scaled_data, scaler = scale_features(df, features, scaler_path=SCALER_PATH)
X, y = create_sequences(scaled_data, df['Trend'].values, window=60)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --------------------------
# Step 5: Build & Train BiLSTM Model
# --------------------------
model = build_bilstm_model((X_train.shape[1], X_train.shape[2]))
history = train_model(model, X_train, y_train)
save_model(model, MODEL_PATH)

# --------------------------
# Step 6: Plot Training History
# --------------------------
plot_training_history(history)

# --------------------------
# Step 7: Evaluate Model
# --------------------------
y_pred = evaluate_model(model, X_test, y_test)
plot_trend_comparison(y_test, y_pred)
