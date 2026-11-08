"""Train and compare LSTM / GRU / RNN models.

Usage:
    python -m src.train --ticker AAPL --model all --epochs 30
"""
import argparse
import json
import os

import joblib
from tensorflow import keras

from src.data import prepare_dataset, inverse_close
from src.models import build_model
from src.evaluate import compute_metrics, plot_predictions, plot_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", default="all", choices=["lstm", "gru", "rnn", "all"])
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    ds = prepare_dataset(args.ticker, args.start, args.window)
    X_train, y_train = ds["X_train"], ds["y_train"]
    X_test, y_test = ds["X_test"], ds["y_test"]
    scaler = ds["scaler"]
    y_true = inverse_close(scaler, y_test)

    kinds = ["lstm", "gru", "rnn"] if args.model == "all" else [args.model]
    preds, histories, metrics = {}, {}, {}

    for kind in kinds:
        print(f"\n=== Training {kind.upper()} ===")
        model = build_model(kind, X_train.shape[1:])
        early = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        hist = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[early],
            verbose=1,
            shuffle=False,
        )
        pred = inverse_close(scaler, model.predict(X_test).ravel())
        preds[kind], histories[kind] = pred, hist
        metrics[kind] = compute_metrics(y_true, pred)
        model.save(f"models/{args.ticker}_{kind}.keras")
        print(kind.upper(), metrics[kind])

    joblib.dump(scaler, f"models/{args.ticker}_scaler.pkl")
    dates = ds["dates"][-len(y_true):]
    plot_predictions(dates, y_true, preds, args.ticker, f"results/{args.ticker}_predictions.png")
    plot_loss(histories, f"results/{args.ticker}_loss.png")
    with open(f"results/{args.ticker}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSummary")
    print(f"{'Model':<6} {'RMSE':>10} {'MAE':>10} {'MAPE%':>8} {'R2':>8}")
    for k, m in metrics.items():
        print(f"{k.upper():<6} {m['RMSE']:>10.2f} {m['MAE']:>10.2f} {m['MAPE']:>8.2f} {m['R2']:>8.3f}")


if __name__ == "__main__":
    main()
