# models.py
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU

from keras import backend as K
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings
import torch
from chronos import ChronosPipeline, ChronosBoltPipeline, Chronos2Pipeline
from importlib.metadata import PackageNotFoundError, version

# Model Management
from setup_module.model_base import BaseForecastModel, ModelMetadata, ModelCategory
from setup_module.quantile_helpers import (
    TIREX_FIXED_QUANTILE_LEVELS,
    monthly_forecast_index,
    normal_quantiles_from_forecast,
    sample_quantiles_from_tensor,
    unique_confidence_levels,
    confidence_level_from_quantile,
    interpolate_quantile_levels,
)

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output


class SeasonalNaiveModel(BaseForecastModel):
    """Seasonal Naive Forecast - wiederholt letzte Saison."""

    def __init__(self, season_length=12):
        super().__init__(season_length=season_length)
        self.season_length = season_length

    def fit(self, data, **kwargs):
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


class MovingAverageModel(BaseForecastModel):
    """Moving Average Forecast - nutzt Durchschnitt der letzten N Werte."""

    def __init__(self, window=12):
        super().__init__(window=window)
        self.window = window

    def fit(self, data, **kwargs):
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


class ARIMAModel(BaseForecastModel):
    """ARIMA - AutoRegressive Integrated Moving Average."""

    def __init__(self, order=(1, 1, 1)):
        super().__init__(order=order)
        self.order = order
        self.model = None

    def fit(self, data, **kwargs):
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


# Update ProphetModel class
class ProphetModel(BaseForecastModel):
    """Prophet - Facebook's Zeitreihenmodell mit Trend und Saisonalität."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.data = None
        self.params = {}

    def fit(
        self,
        data,
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


class LSTMModel(BaseForecastModel):
    """LSTM - Long Short-Term Memory Neural Network."""

    def __init__(self, epochs=50, batch_size=1):
        super().__init__(epochs=epochs, batch_size=batch_size)
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        self.last_sequence = None
        self.data = None

    def __del__(self):
        """Cleanup Keras session when model is deleted"""
        if self.model is not None:
            K.clear_session()
            del self.model

    def fit(self, data, **kwargs):
        # Clear any previous session
        K.clear_session()

        self.data = data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        self.scaler = scaler

        X, y = [], []
        window_size = 12
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i - window_size : i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        self.model.add(LSTM(50))
        self.model.add(Dense(1))
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.is_fitted = True

    def predict(self, steps) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() trainiert werden")
        last_sequence = (
            self.scaler.transform(self.data.values[-12:].reshape(-1, 1))
            .flatten()
            .tolist()
        )
        forecast = []
        for _ in range(steps):
            input_seq = np.array(last_sequence[-12:]).reshape((1, 12, 1))
            pred = self.model.predict(input_seq, verbose=0)
            forecast.append(pred[0, 0])
            last_sequence.append(pred[0, 0])
        forecast = self.scaler.inverse_transform(
            np.array(forecast).reshape(-1, 1)
        ).flatten()
        return pd.Series(
            forecast,
            index=pd.date_range(
                self.data.index[-1] + pd.DateOffset(months=1), periods=steps, freq="M"
            ),
        )

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="LSTM",
            description="Long Short-Term Memory - Deep Learning für komplexe Muster",
            category=ModelCategory.DEEP_LEARNING,
            requires_long_history=True,
            min_data_points=50,
            default_params={"epochs": 50, "batch_size": 1},
        )


class GRUModel(BaseForecastModel):
    """GRU - Gated Recurrent Unit Neural Network."""

    def __init__(self, epochs=50, batch_size=1):
        super().__init__(epochs=epochs, batch_size=batch_size)
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        self.last_sequence = None
        self.data = None

    def __del__(self):
        """Cleanup Keras session when model is deleted"""
        if self.model is not None:
            K.clear_session()
            del self.model

    def fit(self, data, **kwargs):
        # Clear any previous session
        K.clear_session()

        # Same implementation as before, but using self.epochs and self.batch_size
        self.data = data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        self.scaler = scaler

        X, y = [], []
        window_size = 12
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i - window_size : i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        self.model = Sequential()
        self.model.add(GRU(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        self.model.add(GRU(50))
        self.model.add(Dense(1))
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.is_fitted = True

    def predict(self, steps) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() trainiert werden")
        last_sequence = (
            self.scaler.transform(self.data.values[-12:].reshape(-1, 1))
            .flatten()
            .tolist()
        )
        forecast = []
        for _ in range(steps):
            input_seq = np.array(last_sequence[-12:]).reshape((1, 12, 1))
            pred = self.model.predict(input_seq, verbose=0)
            forecast.append(pred[0, 0])
            last_sequence.append(pred[0, 0])
        forecast = self.scaler.inverse_transform(
            np.array(forecast).reshape(-1, 1)
        ).flatten()
        return pd.Series(
            forecast,
            index=pd.date_range(
                self.data.index[-1] + pd.DateOffset(months=1), periods=steps, freq="M"
            ),
        )

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="GRU",
            description="Gated Recurrent Unit - Schnellere Alternative zu LSTM",
            category=ModelCategory.DEEP_LEARNING,
            requires_long_history=True,
            min_data_points=50,
            default_params={"epochs": 50, "batch_size": 1},
        )


class SESModel(BaseForecastModel):
    """SES - Simple Exponential Smoothing."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.data = None

    def fit(self, data, **kwargs):
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


class SARIMAModel(BaseForecastModel):
    """SARIMA - Seasonal ARIMA mit Saisonalitäts-Unterstützung."""

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        super().__init__(order=order, seasonal_order=seasonal_order)
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None

    def fit(self, data, **kwargs):
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


class HoltWintersModel(BaseForecastModel):
    """Holt-Winters - Exponential Smoothing mit Trend und Saisonalität."""

    def __init__(self, seasonal_periods=12):
        super().__init__(seasonal_periods=seasonal_periods)
        self.seasonal_periods = seasonal_periods
        self.model = None

    def fit(self, data, **kwargs):
        self.model = ExponentialSmoothing(
            data, seasonal_periods=self.seasonal_periods, seasonal="add"
        ).fit()
        self.data = data
        self.is_fitted = True

    def predict(self, steps) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() trainiert werden")
        forecast = self.model.forecast(steps)
        return forecast

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="Holt-Winters",
            description="Exponential Smoothing mit Trend und Saisonalität",
            category=ModelCategory.STATISTICAL,
            supports_seasonality=True,
            min_data_points=24,
            default_params={"seasonal_periods": 12},
        )


class XGBoostModel(BaseForecastModel):
    """XGBoost - Gradient Boosting für Zeitreihen."""

    def __init__(self, n_estimators=200, window_size=12):
        super().__init__(n_estimators=n_estimators, window_size=window_size)
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.model = XGBRegressor(n_estimators=self.n_estimators)

    def create_features(self, data):
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i - self.window_size : i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def fit(self, data, **kwargs):
        self.data = data
        X, y = self.create_features(data.values)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, steps) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() trainiert werden")
        last_sequence = self.data.values[-self.window_size :].tolist()
        forecast = []
        for _ in range(steps):
            X_input = np.array(last_sequence[-self.window_size :]).reshape(1, -1)
            pred = self.model.predict(X_input)[0]
            forecast.append(pred)
            last_sequence.append(pred)
        return pd.Series(
            forecast,
            index=pd.date_range(
                self.data.index[-1] + pd.DateOffset(months=1), periods=steps, freq="M"
            ),
        )

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="XGBoost",
            description="Extreme Gradient Boosting - leistungsstarkes ML-Modell",
            category=ModelCategory.MACHINE_LEARNING,
            requires_long_history=True,
            min_data_points=50,
            default_params={"n_estimators": 200, "window_size": 12},
        )


class RandomForestModel(BaseForecastModel):
    """Random Forest - Ensemble von Entscheidungsbäumen."""

    def __init__(self, n_estimators=200, window_size=12):
        super().__init__(n_estimators=n_estimators, window_size=window_size)
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.model = RandomForestRegressor(n_estimators=self.n_estimators)

    def create_features(self, data):
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i - self.window_size : i])
            y.append(data[i])
        return np.array(X), np.array(y)

    def fit(self, data, **kwargs):
        self.data = data
        X, y = self.create_features(data.values)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, steps) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("Modell muss erst mit fit() trainiert werden")
        last_sequence = self.data.values[-self.window_size :].tolist()
        forecast = []
        for _ in range(steps):
            X_input = np.array(last_sequence[-self.window_size :]).reshape(1, -1)
            pred = self.model.predict(X_input)[0]
            forecast.append(pred)
            last_sequence.append(pred)
        return pd.Series(
            forecast,
            index=pd.date_range(
                self.data.index[-1] + pd.DateOffset(months=1), periods=steps, freq="M"
            ),
        )

    @classmethod
    def get_metadata(cls) -> ModelMetadata:
        return ModelMetadata(
            name="Random Forest",
            description="Ensemble von Entscheidungsbäumen - robust gegen Overfitting",
            category=ModelCategory.MACHINE_LEARNING,
            requires_long_history=True,
            min_data_points=50,
            default_params={"n_estimators": 200, "window_size": 12},
        )


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

    def fit(self, data, **kwargs):
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


class ChronosBoltModel(BaseForecastModel):
    """Chronos-Bolt - Amazons optimiertes, schnelleres Zero-Shot Zeitreihenmodell."""

    def __init__(self, model_size="base", device=None):
        super().__init__(model_size=model_size)
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.data = None

    def fit(self, data, **kwargs):
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


class Chronos2Model(BaseForecastModel):
    """Chronos-2 - Die zweite Generation der Amazon Zero-Shot Modelle (Encoder-Decoder)."""

    def __init__(self, model_size="standard", device=None):
        super().__init__(model_size=model_size)
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.data = None

    def fit(self, data, **kwargs):
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


# Am Ende der Datei nach der EnsembleModel-Klasse einfügen:


class TiRexModel(BaseForecastModel):
    """TiRex - NX-AI Zero-Shot Zeitreihenmodell auf Basis von xLSTM."""

    def __init__(self, model_id="NX-AI/TiRex", device=None, backend=None):
        super().__init__(model_id=model_id, backend=backend)
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = backend
        self.model = None
        self.data = None

    def fit(self, data, **kwargs):
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
            TIREX_FIXED_QUANTILE_LEVELS,
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


class TransformerModel:
    def __init__(self, window_size=12, n_heads=4, d_model=64, n_layers=2, dropout=0.1):
        self.window_size = window_size
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.scaler = None
        self.model = None
        self.data = None

    def build_model(self, input_shape):
        from keras.layers import (
            MultiHeadAttention,
            LayerNormalization,
            Dense,
            Input,
        )
        from keras.models import Model
        import tensorflow as tf

        inputs = Input(shape=input_shape)
        x = inputs

        # Transformer Encoder
        for _ in range(self.n_layers):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=self.n_heads, key_dim=self.d_model
            )(x, x)
            attention_output = tf.keras.layers.Dropout(self.dropout)(attention_output)
            x = LayerNormalization(epsilon=1e-6)(x + attention_output)

            # Feed forward
            ffn_output = Dense(self.d_model * 4, activation="relu")(x)
            ffn_output = Dense(self.d_model)(ffn_output)
            ffn_output = tf.keras.layers.Dropout(self.dropout)(ffn_output)
            x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

        # Output
        outputs = Dense(1)(
            x[:, -1, :]
        )  # Use only the last sequence element for prediction

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, data):
        self.data = data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        self.scaler = scaler

        X, y = [], []
        for i in range(self.window_size, len(scaled_data)):
            X.append(scaled_data[i - self.window_size : i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)

        # Reshape for transformer (batch_size, sequence_length, features)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        self.model = self.build_model((X.shape[1], 1))
        self.model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    def predict(self, steps) -> pd.Series:
        last_sequence = (
            self.scaler.transform(self.data.values[-self.window_size :].reshape(-1, 1))
            .flatten()
            .tolist()
        )
        forecast = []

        for _ in range(steps):
            input_seq = np.array(last_sequence[-self.window_size :]).reshape(
                (1, self.window_size, 1)
            )
            pred = self.model.predict(input_seq, verbose=0)
            forecast.append(pred[0, 0])
            last_sequence.append(pred[0, 0])

        forecast = self.scaler.inverse_transform(
            np.array(forecast).reshape(-1, 1)
        ).flatten()
        return pd.Series(
            forecast,
            index=pd.date_range(
                self.data.index[-1] + pd.DateOffset(months=1), periods=steps, freq="M"
            ),
        )


class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def fit(self, data):
        for model in self.models:
            model.fit(data)

    def predict(self, steps):
        forecasts = pd.DataFrame()
        for model in self.models:
            forecasts[model.__class__.__name__] = model.predict(steps)
        ensemble_forecast = forecasts.mean(axis=1)
        return ensemble_forecast
