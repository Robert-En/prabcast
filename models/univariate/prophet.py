import pandas as pd
from prophet import Prophet

from setup_module.model_base import BaseForecastModel, ModelMetadata, ModelCategory
from setup_module.quantile_helpers import (
    monthly_forecast_index,
    unique_confidence_levels,
    confidence_level_from_quantile,
)


def standardize_data(data):
    """
    Standardizes input data for all models.
    Returns both original format and Prophet format (ds, y).
    """
    # Original format - datetime index and values
    std_data = data.copy()
    if not isinstance(std_data.index, pd.DatetimeIndex):
        std_data.index = pd.to_datetime(std_data.index)

    # Prophet format - ds and y columns
    prophet_data = pd.DataFrame({"ds": std_data.index, "y": std_data.values})

    return std_data, prophet_data

class ProphetModel(BaseForecastModel):
    """Prophet - Facebook's Zeitreihenmodell mit Trend und Saisonalität."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.data = None
        self.params = {}

    def fit(
        self,
        data: pd.Series,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        interval_width=0.80,
        **kwargs,
    ):
        _, prophet_data = standardize_data(data)
        self.data = data
        self.params = {
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "interval_width": interval_width,
            **kwargs,
        }
        self.model = Prophet(**self.params)
        self.model.fit(prophet_data)
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

        future_dates = monthly_forecast_index(self.data, steps)
        future_df = pd.DataFrame({"ds": future_dates})

        quantile_to_confidence = {
            q: confidence_level_from_quantile(q) for q in quantile_levels
        }
        forecasts_by_level: dict[float, pd.DataFrame] = {}
        point_forecast = None
        if any(abs(q - 0.5) < 1e-9 for q in quantile_levels):
            self.model.interval_width = self.params.get("interval_width", 0.80)
            point_forecast = self.model.predict(future_df)

        for confidence_level in unique_confidence_levels(
            [q for q in quantile_levels if abs(q - 0.5) >= 1e-9]
        ):
            self.model.interval_width = confidence_level
            forecasts_by_level[confidence_level] = self.model.predict(future_df)

        result: dict[float, pd.Series] = {}
        for q in quantile_levels:
            if abs(q - 0.5) < 1e-9:
                result[q] = pd.Series(point_forecast["yhat"].values, index=future_dates)
            elif q < 0.5:
                result[q] = pd.Series(
                    forecasts_by_level[quantile_to_confidence[q]]["yhat_lower"].values,
                    index=future_dates,
                )
            else:
                result[q] = pd.Series(
                    forecasts_by_level[quantile_to_confidence[q]]["yhat_upper"].values,
                    index=future_dates,
                )
        return result

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="Prophet",
            description="Facebook Prophet - automatische Trend- und Saisonalitätserkennung",
            category=ModelCategory.STATISTICAL,
            supports_seasonality=True,
            is_probabilistic=True,
            supports_quantiles=True,
            min_data_points=20,
            default_params={"interval_width": 0.80},
        )
