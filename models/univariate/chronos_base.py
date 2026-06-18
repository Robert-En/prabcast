import pandas as pd
import torch
from chronos import ChronosPipeline

from setup_module.model_base import BaseForecastModel, ModelMetadata, ModelCategory
from setup_module.quantile_helpers import monthly_forecast_index, sample_quantiles_from_tensor


class ChronosModel(BaseForecastModel):
    """Chronos - Amazons Zero-Shot Zeitreihenmodell (Pre-trained Transformer)."""

    def __init__(self, model_size="base", device=None):
        super().__init__(model_size=model_size)
        self.model_size = model_size
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Automatische Geräteerkennung (GPU falls vorhanden)
        self.pipeline = None
        self.data = None

    def fit(self, data: pd.Series, **kwargs):
        """Chronos benötigt kein Training im klassischen Sinne (Zero-Shot)."""
        self.data = data
        if self.pipeline is None:
            self.pipeline = ChronosPipeline.from_pretrained(
                f"amazon/chronos-t5-{self.model_size}",
                device_map=self.device,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )
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
            name="Chronos",
            description="Ein universelles Zero-Shot-Prognosemodell basierend auf der Transformer-Architektur. "
            "Es behandelt Zeitreihen wie Textsequenzen und ermöglicht hochpräzise Vorhersagen ohne lokales "
            "Training, ideal für univariat-komplexe Datenmuster.",
            category=ModelCategory.DEEP_LEARNING,
            requires_stationarity=False,
            is_probabilistic=True,
            supports_quantiles=True,
            supports_seasonality=False,
            min_data_points=12,
            default_params={"model_size": "base"},
        )
