import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import GRU, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from setup_module.model_base import BaseForecastModel, ModelMetadata, ModelCategory


class GRUModel(BaseForecastModel):
    """GRU - Gated Recurrent Unit Neural Network."""

    def __init__(self, epochs=50, batch_size=1):
        super().__init__(epochs=epochs, batch_size=batch_size)
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        self.last_sequence = None
        self.data = None

    def __del__(self):
        """Cleanup Keras session when model is deleted"""
        if self.model is not None:
            K.clear_session()
            del self.model

    def fit(self, data: pd.Series, **kwargs):
        # Clear any previous session
        K.clear_session()

        # Same implementation as before, but using self.epochs and self.batch_size
        self.data = data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        self.scaler = scaler

        X, y = [], []
        window_size = 12
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i - window_size : i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        self.model = Sequential()
        self.model.add(GRU(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        self.model.add(GRU(50))
        self.model.add(Dense(1))
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.is_fitted = True

    def predict(self, steps) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() trainiert werden")
        last_sequence = (
            self.scaler.transform(self.data.values[-12:].reshape(-1, 1))
            .flatten()
            .tolist()
        )
        forecast = []
        for _ in range(steps):
            input_seq = np.array(last_sequence[-12:]).reshape((1, 12, 1))
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

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="GRU",
            description="Gated Recurrent Unit - Schnellere Alternative zu LSTM",
            category=ModelCategory.DEEP_LEARNING,
            requires_long_history=True,
            min_data_points=50,
            default_params={"epochs": 50, "batch_size": 1},
        )
