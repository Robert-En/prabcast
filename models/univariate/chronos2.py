import pandas as pd
import torch
from chronos import Chronos2Pipeline

from setup_module.model_base import BaseForecastModel, ModelMetadata, ModelCategory
from setup_module.quantile_helpers import monthly_forecast_index


class Chronos2Model(BaseForecastModel):
    """Chronos-2 - Die zweite Generation der Amazon Zero-Shot Modelle (Encoder-Decoder)."""

    def __init__(self, model_size="standard", device=None):
        super().__init__(model_size=model_size)
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.data = None

    def fit(self, data: pd.Series, **kwargs):
        self.data = data
        if self.pipeline is None:
            # Mapping der IDs für Hugging Face
            if self.model_size == "standard" or self.model_size == "":
                model_id = "amazon/chronos-2"
            else:
                model_id = f"amazon/chronos-2-{self.model_size}"

            self.pipeline = Chronos2Pipeline.from_pretrained(
                model_id,
                device_map=self.device,
            )
        self.is_fitted = True

    def _build_context_df(self) -> pd.DataFrame:
        context_df = self.data.reset_index()
        context_df.columns = ["timestamp", "target"]
        context_df["item_id"] = (
            "H1"  # "H1" ist ein Platzhalter, wie im Quickstart-Notebook
        )
        return context_df

    def _forecast_index(self, steps: int) -> pd.DatetimeIndex:
        return monthly_forecast_index(self.data, steps)

    @staticmethod
    def _resolve_quantile_column(forecast_df: pd.DataFrame, level: float) -> str:
        for candidate in (str(level), f"{level:.3f}", f"{level:.2f}", f"{level:.1f}"):
            if candidate in forecast_df.columns:
                return candidate
        raise KeyError(
            f"Quantil {level} nicht in Spalten {list(forecast_df.columns)} gefunden"
        )

    def predict(self, steps) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() initialisiert werden")

        quantiles = self.predict_quantiles(steps, [0.5])
        return quantiles[0.5]

    def predict_quantiles(
        self, steps: int, quantile_levels: list[float]
    ) -> dict[float, pd.Series]:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() initialisiert werden")

        unique_levels = sorted(set(quantile_levels))
        forecast_df = self.pipeline.predict_df(
            self._build_context_df(),
            prediction_length=steps,
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            quantile_levels=unique_levels,
        )
        forecast_index = self._forecast_index(steps)
        return {
            level: pd.Series(
                forecast_df[self._resolve_quantile_column(forecast_df, level)].values,
                index=forecast_index,
            )
            for level in quantile_levels
        }

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="Chronos-2",
            description="Die nächste Generation der Foundation-Models für Zeitreihen. Es bietet ein massiv erweitertes Kontextfenster für historische Daten und unterstützt erstmals nativ multivariate Eingaben sowie externe Kovariaten, um komplexe Abhängigkeiten in industriellen Prozessen abzubilden.",
            category=ModelCategory.DEEP_LEARNING,
            requires_stationarity=False,
            is_probabilistic=True,
            supports_quantiles=True,
            supports_seasonality=True,
            min_data_points=12,
            default_params={"model_size": "standard"},
        )
