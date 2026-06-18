import numpy as np
import pandas as pd
import torch

from setup_module.model_base import BaseForecastModel, ModelMetadata, ModelCategory
from setup_module.quantile_helpers import (
    TIREX_QUANTILE_LEVELS,
    monthly_forecast_index,
    interpolate_quantile_levels,
)


class TiRexModel(BaseForecastModel):
    """TiRex - NX-AI Zero-Shot Zeitreihenmodell auf Basis von xLSTM."""

    def __init__(self, model_id="NX-AI/TiRex", device=None, backend=None):
        super().__init__(model_id=model_id, backend=backend)
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = backend
        self.model = None
        self.data = None

    def fit(self, data: pd.Series, **kwargs):
        """TiRex ist ein Zero-Shot Modell und lädt nur die vortrainierten Gewichte."""
        self.data = data
        if self.model is None:
            try:
                from tirex import load_model
            except ImportError as e:
                raise RuntimeError(
                    "TiRex benötigt das Paket 'tirex-ts'. Installieren Sie es mit "
                    "'pip install tirex-ts'."
                ) from e

            load_kwargs = {}
            if self.backend is not None:
                load_kwargs["backend"] = self.backend

            self.model = load_model(self.model_id, **load_kwargs)
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()

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

        values = np.asarray(self.data.values, dtype=np.float32).reshape(1, -1)
        context = torch.tensor(values, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            quantiles, _ = self.model.forecast(
                context=context,
                prediction_length=steps,
            )

        forecast_index = monthly_forecast_index(self.data, steps)
        quantile_values = quantiles.detach().cpu().numpy()
        interpolated = interpolate_quantile_levels(
            quantile_values,
            TIREX_QUANTILE_LEVELS,
            quantile_levels,
        )
        return {
            level: pd.Series(values, index=forecast_index)
            for level, values in interpolated.items()
        }

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="TiRex",
            description="NX-AI TiRex - kompaktes xLSTM-basiertes Zero-Shot Foundation Model "
            "für Zeitreihenprognosen mit Punkt- und Quantilschätzungen.",
            category=ModelCategory.DEEP_LEARNING,
            requires_stationarity=False,
            is_probabilistic=True,
            supports_quantiles=True,
            supports_seasonality=True,
            min_data_points=12,
            default_params={"model_id": "NX-AI/TiRex", "backend": None},
        )
