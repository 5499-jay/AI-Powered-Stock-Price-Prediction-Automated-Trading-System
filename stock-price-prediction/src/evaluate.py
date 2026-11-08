"""Metrics and plotting."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def plot_predictions(dates, y_true, preds: dict, ticker: str, path: str):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Actual", color="black", linewidth=1.5)
    for name, p in preds.items():
        plt.plot(dates, p, label=name.upper(), alpha=0.8)
    plt.title(f"{ticker}: Actual vs Predicted Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_loss(histories: dict, path: str):
    plt.figure(figsize=(8, 4))
    for name, h in histories.items():
        plt.plot(h.history["val_loss"], label=f"{name.upper()} val")
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
