import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from setup_module.model_base import BaseForecastModel, ModelMetadata, ModelCategory


class HoltWintersModel(BaseForecastModel):
    """Holt-Winters - Exponential Smoothing mit Trend und Saisonalität."""

    def __init__(self, seasonal_periods=12):
        super().__init__(seasonal_periods=seasonal_periods)
        self.seasonal_periods = seasonal_periods
        self.model = None

    def fit(self, data: pd.Series, **kwargs):
        self.model = ExponentialSmoothing(
            data, seasonal_periods=self.seasonal_periods, seasonal="add"
        ).fit()
        self.data = data
        self.is_fitted = True

    def predict(self, steps) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() trainiert werden")
        forecast = self.model.forecast(steps)
        return forecast

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="Holt-Winters",
            description="Exponential Smoothing mit Trend und Saisonalität",
            category=ModelCategory.STATISTICAL,
            supports_seasonality=True,
            min_data_points=24,
            default_params={"seasonal_periods": 12},
        )
