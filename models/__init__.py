"""models package"""
import warnings

# Multivariate models
from .multivariate.lstm_multi import build_lstm_model
from .multivariate.xgb_multi import build_xgboost_model
# Univariate models
from .univariate.arima import ARIMAModel
from .univariate.chronos_base import ChronosModel
from .univariate.chronos2 import Chronos2Model
from .univariate.chronos_bolt import ChronosBoltModel
from .univariate.gru import GRUModel
from .univariate.holt_winters import HoltWintersModel
from .univariate.lstm import LSTMModel
from .univariate.moving_average import MovingAverageModel
from .univariate.prophet import ProphetModel
from .univariate.random_forest import RandomForestModel
from .univariate.sarima import SARIMAModel
from .univariate.seasonal_naive import SeasonalNaiveModel
from .univariate.ses import SESModel
from .univariate.tirex import TiRexModel
from .univariate.ensemble import EnsembleModel
from .univariate.transformer import TransformerModel
from .univariate.xgboost_model import XGBoostModel



# Ignoriert NUR Warnungen, die aus dem 'models'-Ordner oder dessen Unterordnern kommen
warnings.filterwarnings("ignore", module="^models\\.")

__all__ = [
    # Univariate models
    "ARIMAModel",
    "ChronosModel",
    "Chronos2Model",
    "ChronosBoltModel",
    "GRUModel",
    "HoltWintersModel",
    "LSTMModel",
    "MovingAverageModel",
    "ProphetModel",
    "RandomForestModel",
    "SARIMAModel",
    "SeasonalNaiveModel",
    "SESModel",
    "TiRexModel",
    "XGBoostModel",
    "TransformerModel",
    "EnsembleModel",

    # Multivariate models
    "build_xgboost_model",
    "build_lstm_model",
]
