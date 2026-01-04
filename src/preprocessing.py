import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib

def scale_features(df, feature_cols, scaler_path=None):
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    if scaler_path:
        joblib.dump(scaler, scaler_path)
    return scaled_data, scaler

def create_sequences(data, targets, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(targets[i])
    return np.array(X), np.array(y)
