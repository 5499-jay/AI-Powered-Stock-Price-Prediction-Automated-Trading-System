"""Streamlit demo: load a trained model and view test-set predictions."""
import os

import joblib
import pandas as pd
import streamlit as st
from tensorflow import keras

from src.data import prepare_dataset, inverse_close

st.set_page_config(page_title="AI Stock Price Prediction", layout="wide")
st.title("AI-Powered Stock Price Prediction")
st.caption("Educational project. Not financial advice.")

ticker = st.text_input("Ticker (Yahoo Finance format)", "AAPL").upper()
model_kind = st.selectbox("Model", ["lstm", "gru", "rnn"])

model_path = f"models/{ticker}_{model_kind}.keras"
scaler_path = f"models/{ticker}_scaler.pkl"

if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
    st.warning(f"No trained model for {ticker}. Run: python -m src.train --ticker {ticker}")
    st.stop()

if st.button("Predict"):
    ds = prepare_dataset(ticker)
    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    pred = inverse_close(scaler, model.predict(ds["X_test"]).ravel())
    actual = inverse_close(scaler, ds["y_test"])
    dates = ds["dates"][-len(actual):]

    chart = pd.DataFrame({"Actual": actual, "Predicted": pred}, index=dates)
    st.line_chart(chart)
    st.metric("Latest predicted close", f"{pred[-1]:.2f}")
