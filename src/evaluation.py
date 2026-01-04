import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

def plot_training_history(history):
    plt.figure(figsize=(12,5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Training Loss")
    plt.show()

    plt.figure(figsize=(12,5))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title("Training Accuracy")
    plt.show()

def evaluate_model(model, X_test, y_test):
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob.flatten() > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Directional Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:\n", cm)
    return y_pred

def plot_trend_comparison(y_test, y_pred):
    plt.figure(figsize=(14,5))
    plt.plot(y_test, label='Actual Trend')
    plt.plot(y_pred, label='Predicted Trend', alpha=0.7)
    plt.legend()
    plt.title("Actual vs Predicted Trend")
    plt.show()
