import pandas as pd

from setup_module.model_base import BaseForecastModel, ModelMetadata, ModelCategory


class MovingAverageModel(BaseForecastModel):
    """Moving Average Forecast - nutzt Durchschnitt der letzten N Werte."""

    def __init__(self, window=12):
        super().__init__(window=window)
        self.window = window

    def fit(self, data: pd.Series, **kwargs):
        self.data = data
        self.is_fitted = True

    def predict(self, steps) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() trainiert werden")
        mean_value = self.data.iloc[-self.window :].mean()
        forecast = pd.Series(
            [mean_value] * steps,
            index=pd.date_range(
                self.data.index[-1] + pd.DateOffset(months=1), periods=steps, freq="M"
            ),
        )
        return forecast

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="Moving Average",
            description="Durchschnitt der letzten N Werte - einfache Baseline",
            category=ModelCategory.NAIVE,
            min_data_points=12,
            default_params={"window": 12},
        )
