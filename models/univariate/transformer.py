import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class TransformerModel:
    def __init__(self, window_size=12, n_heads=4, d_model=64, n_layers=2, dropout=0.1):
        self.window_size = window_size
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.scaler = None
        self.model = None
        self.data = None

    def build_model(self, input_shape):
        from keras.layers import (
            MultiHeadAttention,
            LayerNormalization,
            Dense,
            Input,
        )
        from keras.models import Model
        import tensorflow as tf

        inputs = Input(shape=input_shape)
        x = inputs

        # Transformer Encoder
        for _ in range(self.n_layers):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=self.n_heads, key_dim=self.d_model
            )(x, x)
            attention_output = tf.keras.layers.Dropout(self.dropout)(attention_output)
            x = LayerNormalization(epsilon=1e-6)(x + attention_output)

            # Feed forward
            ffn_output = Dense(self.d_model * 4, activation="relu")(x)
            ffn_output = Dense(self.d_model)(ffn_output)
            ffn_output = tf.keras.layers.Dropout(self.dropout)(ffn_output)
            x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

        # Output
        outputs = Dense(1)(
            x[:, -1, :]
        )  # Use only the last sequence element for prediction

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, data: pd.Series):
        self.data = data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        self.scaler = scaler

        X, y = [], []
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i - self.window_size : i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)

        # Reshape for transformer (batch_size, sequence_length, features)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        self.model = self.build_model((X.shape[1], 1))
        self.model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    def predict(self, steps) -> pd.Series:
        last_sequence = (
            self.scaler.transform(self.data.values[-self.window_size :].reshape(-1, 1))
            .flatten()
            .tolist()
        )
        forecast = []

        for _ in range(steps):
            input_seq = np.array(last_sequence[-self.window_size :]).reshape(
                (1, self.window_size, 1)
            )
            pred = self.model.predict(input_seq, verbose=0)
            forecast.append(pred[0, 0])
            last_sequence.append(pred[0, 0])

        forecast = self.scaler.inverse_transform(
            np.array(forecast).reshape(-1, 1)
        ).flatten()
        return pd.Series(
            forecast,
            index=pd.date_range(
                self.data.index[-1] + pd.DateOffset(months=1), periods=steps, freq="M"
            ),
        )
