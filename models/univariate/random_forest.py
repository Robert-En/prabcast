import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from setup_module.model_base import BaseForecastModel, ModelMetadata, ModelCategory


class RandomForestModel(BaseForecastModel):
    """Random Forest - Ensemble von Entscheidungsbäumen."""

    def __init__(self, n_estimators=200, window_size=12):
        super().__init__(n_estimators=n_estimators, window_size=window_size)
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.model = RandomForestRegressor(n_estimators=self.n_estimators)

    def create_features(self, data):
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i - self.window_size : i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def fit(self, data: pd.Series, **kwargs):
        self.data = data
        X, y = self.create_features(data.values)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, steps) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() trainiert werden")
        last_sequence = self.data.values[-self.window_size :].tolist()
        forecast = []
        for _ in range(steps):
            X_input = np.array(last_sequence[-self.window_size :]).reshape(1, -1)
            pred = self.model.predict(X_input)[0]
            forecast.append(pred)
            last_sequence.append(pred)
        return pd.Series(
            forecast,
            index=pd.date_range(
                self.data.index[-1] + pd.DateOffset(months=1), periods=steps, freq="M"
            ),
        )

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="Random Forest",
            description="Ensemble von Entscheidungsbäumen - robust gegen Overfitting",
            category=ModelCategory.MACHINE_LEARNING,
            requires_long_history=True,
            min_data_points=50,
            default_params={"n_estimators": 200, "window_size": 12},
        )
