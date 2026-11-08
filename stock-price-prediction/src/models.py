"""Recurrent model builders."""
from tensorflow import keras
from tensorflow.keras import layers


def build_model(kind: str, input_shape: tuple, units: int = 64, dropout: float = 0.2) -> keras.Model:
    layer_map = {"lstm": layers.LSTM, "gru": layers.GRU, "rnn": layers.SimpleRNN}
    if kind not in layer_map:
        raise ValueError(f"Unknown model '{kind}'. Choose from {list(layer_map)}.")
    Rec = layer_map[kind]

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        Rec(units, return_sequences=True),
        layers.Dropout(dropout),
        Rec(units // 2),
        layers.Dropout(dropout),
        layers.Dense(25, activation="relu"),
        layers.Dense(1),
    ], name=f"{kind}_model")

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model
