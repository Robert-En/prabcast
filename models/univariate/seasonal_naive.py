import numpy as np
import pandas as pd

from setup_module.model_base import BaseForecastModel, ModelMetadata, ModelCategory


class SeasonalNaiveModel(BaseForecastModel):
    """Seasonal Naive Forecast - wiederholt letzte Saison."""

    def __init__(self, season_length=12):
        super().__init__(season_length=season_length)
        self.season_length = season_length

    def fit(self, data: pd.Series, **kwargs):
        self.data = data
        self.is_fitted = True

    def predict(self, steps) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() trainiert werden")
        last_season = self.data.iloc[-self.season_length :].values
        repeats = steps // self.season_length + 1
        forecast = np.tile(last_season, repeats)[:steps]
        return pd.Series(
            forecast.flatten(),
            index=pd.date_range(
                self.data.index[-1] + pd.DateOffset(months=1), periods=steps, freq="M"
            ),
        )

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="Seasonal Naive",
            description="Wiederholt die letzte Saison - einfache Baseline für saisonale Daten",
            category=ModelCategory.NAIVE,
            supports_seasonality=True,
            min_data_points=12,
            default_params={"season_length": 12},
        )
