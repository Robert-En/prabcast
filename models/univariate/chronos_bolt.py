from importlib.metadata import PackageNotFoundError, version

import pandas as pd
import torch
from chronos import ChronosBoltPipeline

from setup_module.model_base import BaseForecastModel, ModelMetadata, ModelCategory
from setup_module.quantile_helpers import monthly_forecast_index, sample_quantiles_from_tensor


class ChronosBoltModel(BaseForecastModel):
    """Chronos-Bolt - Amazons optimiertes, schnelleres Zero-Shot Zeitreihenmodell."""

    def __init__(self, model_size="base", device=None):
        super().__init__(model_size=model_size)
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.data = None

    def fit(self, data: pd.Series, **kwargs):
        """Zero-Shot Modell: Lädt die Bolt-Gewichte von Hugging Face."""
        self.data = data
        if self.pipeline is None:
            model_id = f"amazon/chronos-bolt-{self.model_size}"

            try:
                self.pipeline = ChronosBoltPipeline.from_pretrained(
                    model_id,
                    device_map=self.device,
                    torch_dtype=torch.bfloat16
                    if self.device == "cuda"
                    else torch.float32,
                )
            except TypeError as e:
                if "input_patch_size" in str(e):
                    try:
                        installed_version = version("chronos-forecasting")
                    except PackageNotFoundError:
                        installed_version = "unbekannt"
                    raise RuntimeError(
                        "Chronos-Bolt ist inkompatibel mit der installierten "
                        f"'chronos-forecasting' Version ({installed_version}). "
                        "Bitte aktualisieren Sie die Umgebung auf eine Version, "
                        "die Chronos-Bolt-Konfigurationsfelder wie 'input_patch_size' unterstützt."
                    ) from e
                raise
        self.is_fitted = True

    def predict(self, steps) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() initialisiert werden")
        return self.predict_quantiles(steps, [0.5])[0.5]

    def predict_quantiles(
        self, steps: int, quantile_levels: list[float]
    ) -> dict[float, pd.Series]:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() initialisiert werden")

        context = torch.tensor(self.data.values)
        forecast_samples = self.pipeline.predict(context, steps)
        index = monthly_forecast_index(self.data, steps)
        return sample_quantiles_from_tensor(
            forecast_samples, quantile_levels, index, sample_dim=1
        )

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="Chronos-Bolt",
            description="Die performance-optimierte Variante der Chronos-Familie. Durch eine effizientere Architektur erreicht das Modell eine bis zu achtfach schnellere Inferenz bei minimalem Genauigkeitsverlust, was es besonders attraktiv für ressourcenschonende Echtzeit-Anwendungen macht.",
            category=ModelCategory.DEEP_LEARNING,
            requires_stationarity=False,
            is_probabilistic=True,
            supports_quantiles=True,
            supports_seasonality=False,
            min_data_points=12,
            default_params={"model_size": "base"},
        )
