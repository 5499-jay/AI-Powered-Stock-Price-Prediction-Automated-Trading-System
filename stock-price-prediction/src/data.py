"""Data loading, feature engineering, scaling and windowing."""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

FEATURES = ["Close", "Volume", "SMA_20", "EMA_20", "RSI_14", "MACD"]


def download_data(ticker: str, start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    df["SMA_20"] = close.rolling(20).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    return df.dropna()


def make_windows(scaled: np.ndarray, window: int, target_idx: int = 0):
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i])
        y.append(scaled[i, target_idx])
    return np.array(X), np.array(y)


def prepare_dataset(ticker: str, start: str = "2015-01-01", window: int = 60, train_ratio: float = 0.8):
    df = add_indicators(download_data(ticker, start))
    data = df[FEATURES].values

    split = int(len(data) * train_ratio)
    scaler = MinMaxScaler()
    scaler.fit(data[:split])  # fit on train only to avoid leakage
    scaled = scaler.transform(data)

    X, y = make_windows(scaled, window)
    # window i predicts row i+window, so the train/test boundary shifts by `window`
    train_end = split - window
    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[train_end:], y[train_end:]
    test_dates = df.index[split:]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler, "dates": test_dates, "df": df,
    }


def inverse_close(scaler: MinMaxScaler, values: np.ndarray) -> np.ndarray:
    """Convert scaled Close predictions back to price units."""
    dummy = np.zeros((len(values), len(FEATURES)))
    dummy[:, 0] = np.asarray(values).ravel()
    return scaler.inverse_transform(dummy)[:, 0]
