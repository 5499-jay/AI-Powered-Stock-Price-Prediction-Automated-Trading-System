# AI-Powered Stock Price Prediction

Predict next-day stock closing prices using deep learning sequence models: **LSTM**, **GRU**, and a **Simple RNN**, with a side-by-side comparison of their performance.

> ⚠️ **Disclaimer:** This is an educational/academic project. It is not financial advice, and model predictions should not be used for real trading decisions.

## Features

- Downloads historical OHLCV data with `yfinance`
- Adds technical indicators (SMA, EMA, RSI, MACD)
- Sliding-window sequence builder (default: 60 days -> predict next day)
- Three interchangeable models: LSTM, GRU, SimpleRNN (Keras/TensorFlow)
- Evaluation with RMSE, MAE, MAPE and R²
- Plots: actual vs predicted, training loss, model comparison
- Optional Streamlit app for interactive predictions

## Project Structure

```
stock-price-prediction/
├── src/
│   ├── data.py        # download, indicators, scaling, windowing
│   ├── models.py      # LSTM / GRU / RNN builders
│   ├── train.py       # train one or all models
│   └── evaluate.py    # metrics and plots
├── app.py             # Streamlit demo
├── notebooks/         # exploration notebooks
├── models/            # saved models (git-ignored)
├── data/              # cached datasets (git-ignored)
├── results/           # plots and metrics
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/stock-price-prediction.git
cd stock-price-prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train and compare all three models on a ticker:

```bash
python -m src.train --ticker AAPL --start 2015-01-01 --epochs 30 --model all
```

Train a single model:

```bash
python -m src.train --ticker TCS.NS --model lstm --epochs 50
```

Run the demo app:

```bash
streamlit run app.py
```

Tickers follow Yahoo Finance format, e.g. `AAPL`, `RELIANCE.NS` (NSE), `TCS.BO` (BSE).

## Methodology

1. **Data**: daily OHLCV from Yahoo Finance.
2. **Features**: Close, Volume, SMA(20), EMA(20), RSI(14), MACD.
3. **Scaling**: MinMax scaling fitted on the training split only (avoids data leakage).
4. **Split**: chronological 80/20 train/test (no shuffling).
5. **Windowing**: last 60 days of features -> next day's Close.
6. **Models**: two stacked recurrent layers + dropout + dense output, Adam optimizer, MSE loss, early stopping.
7. **Metrics**: RMSE, MAE, MAPE, R² on the held-out test set.

## Results

Run `src.train` to generate results in `results/`. Add your final comparison table here:

| Model | RMSE | MAE | MAPE | R² |
|-------|------|-----|------|----|
| LSTM  |      |     |      |    |
| GRU   |      |     |      |    |
| RNN   |      |     |      |    |

## Future Work

- Transformer / attention-based models
- Sentiment features from news and social media
- Multi-step forecasting
- Hyperparameter tuning and walk-forward validation

## License

MIT License. See `LICENSE`.
