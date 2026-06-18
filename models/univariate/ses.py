import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from setup_module.model_base import BaseForecastModel, ModelMetadata, ModelCategory


class SESModel(BaseForecastModel):
    """SES - Simple Exponential Smoothing."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.data = None

    def fit(self, data: pd.Series, **kwargs):
        self.model = SimpleExpSmoothing(data).fit()
        self.data = data
        self.is_fitted = True

    def predict(self, steps) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() trainiert werden")
        forecast = self.model.forecast(steps)
        return pd.Series(
            forecast,
            index=pd.date_range(
                self.data.index[-1] + pd.DateOffset(months=1), periods=steps, freq="M"
            ),
        )

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="SES",
            description="Simple Exponential Smoothing - für Daten ohne Trend/Saisonalität",
            category=ModelCategory.STATISTICAL,
            min_data_points=15,
            default_params={},
        )
