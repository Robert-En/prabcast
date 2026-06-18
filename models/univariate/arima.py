import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from setup_module.model_base import BaseForecastModel, ModelMetadata, ModelCategory
from setup_module.quantile_helpers import (
    monthly_forecast_index,
    normal_quantiles_from_forecast,
)


class ARIMAModel(BaseForecastModel):
    """ARIMA - AutoRegressive Integrated Moving Average."""

    def __init__(self, order=(1, 1, 1)):
        super().__init__(order=order)
        self.order = order
        self.model = None

    def fit(self, data: pd.Series, **kwargs):
        self.data = data
        self.model = ARIMA(data, order=self.order).fit()
        self.is_fitted = True

    def predict(self, steps) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() trainiert werden")
        return self.predict_quantiles(steps, [0.5])[0.5]

    def predict_quantiles(
        self, steps: int, quantile_levels: list[float]
    ) -> dict[float, pd.Series]:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() trainiert werden")

        forecast_result = self.model.get_forecast(steps=steps)
        index = monthly_forecast_index(self.data, steps)
        return normal_quantiles_from_forecast(
            mean=forecast_result.predicted_mean,
            se=forecast_result.se_mean,
            levels=quantile_levels,
            index=index,
        )

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="ARIMA",
            description="AutoRegressive Integrated Moving Average - klassisches Zeitreihenmodell",
            category=ModelCategory.STATISTICAL,
            requires_stationarity=True,
            is_probabilistic=True,
            supports_quantiles=True,
            min_data_points=30,
            default_params={"order": (1, 1, 1)},
        )
