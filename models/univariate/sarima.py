import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from setup_module.model_base import BaseForecastModel, ModelMetadata, ModelCategory
from setup_module.quantile_helpers import monthly_forecast_index, normal_quantiles_from_forecast


class SARIMAModel(BaseForecastModel):
    """SARIMA - Seasonal ARIMA mit Saisonalitäts-Unterstützung."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        super().__init__(order=order, seasonal_order=seasonal_order)
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None

    def fit(self, data: pd.Series, **kwargs):
        self.data = data
        self.model = SARIMAX(
            data, order=self.order, seasonal_order=self.seasonal_order
        ).fit()
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
            name="SARIMA",
            description="Seasonal ARIMA - für Daten mit Saisonalität",
            category=ModelCategory.STATISTICAL,
            requires_stationarity=True,
            supports_seasonality=True,
            is_probabilistic=True,
            supports_quantiles=True,
            min_data_points=40,
            default_params={"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
        )
