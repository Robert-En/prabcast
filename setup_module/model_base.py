"""
Base Classes für Forecast-Modelle.

Dieses Modul stellt die Basis-Infrastruktur für alle Forecast-Modelle bereit:
- BaseForecastModel: Abstract base class mit standardisiertem Interface
- ModelMetadata: Metadata-Container für Modell-Eigenschaften
- ModelCategory: Kategorisierung (Statistical, ML, DL, Naive)

Alle Modelle sollten von BaseForecastModel ableiten und get_metadata() implementieren.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd
from enum import Enum


class ModelCategory(Enum):
    """
    Kategorisierung von Forecast-Modellen.
    
    - STATISTICAL: Klassische statistische Modelle (ARIMA, SARIMA, SES, Holt-Winters)
    - MACHINE_LEARNING: ML-Modelle (XGBoost, Random Forest)
    - DEEP_LEARNING: Neural Networks (LSTM, GRU, Transformer)
    - NAIVE: Einfache Baseline-Modelle (Seasonal Naive, Moving Average)
    """
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "ml"
    DEEP_LEARNING = "deep_learning"
    NAIVE = "naive"


@dataclass
class ModelMetadata:
    """
    Metadata für ein Forecast-Modell.
    
    Enthält alle wichtigen Informationen über ein Modell:
    - Name, Beschreibung, Kategorie
    - Anforderungen (Stationarität, Saisonalität, Datenmenge)
    - Default-Parameter
    
    Attributes:
        name: Display-Name des Modells (z.B. "ARIMA", "Prophet")
        description: Kurzbeschreibung für User
        category: ModelCategory (Statistical, ML, DL, Naive)
        requires_stationarity: True wenn Modell stationäre Daten benötigt
        supports_seasonality: True wenn Modell Saisonalität modellieren kann
        requires_long_history: True wenn viele Datenpunkte nötig
        is_probabilistic: True wenn Modell Konfidenzintervalle liefert
        supports_quantiles: True wenn Modell native Quantil-Prognosen für Fan-Charts liefert
        min_data_points: Minimum erforderliche Datenpunkte
        default_params: Default-Parameter für Modell-Initialisierung
    """
    name: str
    description: str
    category: ModelCategory
    requires_stationarity: bool = False
    supports_seasonality: bool = False
    requires_long_history: bool = False
    is_probabilistic: bool = False
    supports_quantiles: bool = False
    min_data_points: int = 10
    default_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure default_params is never None."""
        if self.default_params is None:
            self.default_params = {}


class BaseForecastModel(ABC):
    """
    Abstract Base Class für alle Forecast-Modelle.
    
    Standardisiert das Interface für fit() und predict().
    Alle Modelle sollten von dieser Klasse ableiten.
    
    Usage:
        class MyModel(BaseForecastModel):
            def fit(self, data, **kwargs):
                # Training logic
                
            def predict(self, steps):
                # Prediction logic
                
            @classmethod
            def get_metadata(cls):
                return ModelMetadata(
                    name="MyModel",
                    description="My custom model",
                    category=ModelCategory.STATISTICAL
                )
    """
    
    def __init__(self, **params):
        """
        Initialisiert Modell mit Parametern.
        
        Args:
            **params: Modell-spezifische Parameter
        """
        self.params = params
        self.model = None
        self.data = None
        self.is_fitted = False
        
        # Performance tracking (optional)
        self.train_time = None
        self.memory_used = None
    
    @abstractmethod
    def fit(self, data: pd.Series, **kwargs) -> None:
        """
        Trainiert das Modell auf den Daten.
        
        MUSS von Subklassen implementiert werden.
        
        Args:
            data: Zeitreihen-Daten (pandas Series mit DatetimeIndex)
            **kwargs: Zusätzliche fit-Parameter
            
        Postcondition:
            - self.is_fitted = True
            - self.data = data (für spätere Verwendung)
            - self.model = trainiertes Modell
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int) -> pd.Series:
        """
        Erstellt Forecast für n Schritte in die Zukunft.
        
        MUSS von Subklassen implementiert werden.
        
        Args:
            steps: Anzahl Schritte in die Zukunft
            
        Returns:
            Forecast als pandas Series mit DatetimeIndex
            
        Raises:
            RuntimeError: Wenn Modell nicht fitted ist
        """
        pass

    def predict_quantiles(
        self, steps: int, quantile_levels: list[float]
    ) -> dict[float, pd.Series]:
        """Optionale native Quantil-Prognosen für Fan-Charts.

        Nur Modelle mit ``supports_quantiles=True`` in ``get_metadata()`` implementieren
        diese Methode. Rückgabe: Mapping Quantil-Stufe → ``pd.Series`` der Länge ``steps``.

        Args:
            steps: Anzahl Prognose-Horizonte.
            quantile_levels: Angefragte Quantil-Stufen (z. B. 0.25, 0.975).

        Returns:
            Dictionary mit Quantil-Stufen als Keys und Prognose-Serien als Values.

        Raises:
            NotImplementedError: Wenn das Modell keine nativen Quantile unterstützt.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} unterstützt keine nativen Quantil-Prognosen."
        )
    
    @classmethod
    @abstractmethod
    def get_metadata(cls) -> ModelMetadata:
        """
        Gibt Model-Metadata zurück.
        
        MUSS von Subklassen implementiert werden.
        
        Returns:
            ModelMetadata mit allen Modell-Eigenschaften
        """
        pass

    @classmethod
    def supports_native_quantiles(cls) -> bool:
        """Gibt zurück, ob das Modell native Quantil-Prognosen unterstützt.

        Returns:
            True wenn ``supports_quantiles`` in den Modell-Metadaten gesetzt ist.
        """
        return cls.get_metadata().supports_quantiles
    
    def __str__(self):
        """String representation."""
        return f"{self.__class__.__name__}({self.params})"
    
    def __repr__(self):
        """Detailed representation."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.__class__.__name__}({self.params}, {fitted_status})"
