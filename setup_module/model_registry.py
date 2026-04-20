"""
Zentrale Model-Registry für alle Forecast-Modelle.

Dieses Modul stellt eine zentrale Registry bereit, die als Single Source of Truth
für alle verfügbaren Forecast-Modelle dient.

Features:
- Automatische Model-Registrierung
- Abruf von Modellen by name
- Filterung nach Kategorie
- Intelligente Model-Empfehlungen basierend auf Daten-Eigenschaften

Usage:
    from setup_module.model_registry import get_model_registry
    
    registry = get_model_registry()
    
    # Alle verfügbaren Modelle
    all_models = registry.get_all_names()
    
    # Modell abrufen
    ARIMAModel = registry.get("ARIMA")
    model = ARIMAModel(order=(1,1,1))
    
    # Nach Kategorie filtern
    statistical_models = registry.get_by_category(ModelCategory.STATISTICAL)
    
    # Passende Modelle für Daten finden
    suitable = registry.get_suitable_models(data_points=100, has_seasonality=True)
"""

from typing import Dict, List, Type, Optional
import streamlit as st
import inspect
import importlib
from setup_module.model_base import BaseForecastModel, ModelCategory, ModelMetadata


class ModelRegistry:
    """
    Zentrale Registry für alle Forecast-Modelle.
    
    Verwaltet alle verfügbaren Modelle und stellt Methoden bereit um:
    - Modelle zu registrieren
    - Modelle abzurufen (by name, category, data properties)
    - Metadata abzurufen
    """
    
    def __init__(self):
        """Initialisiert leere Registry."""
        self._models: Dict[str, Type[BaseForecastModel]] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
    
    def register(self, model_class: Type[BaseForecastModel]) -> None:
        """
        Registriert ein Modell in der Registry.
        
        Args:
            model_class: Modell-Klasse (muss BaseForecastModel ableiten und get_metadata() implementieren)
            
        Raises:
            ValueError: Wenn Modell bereits registriert
            AttributeError: Wenn get_metadata() nicht implementiert
        """
        try:
            metadata = model_class.get_metadata()
        except (AttributeError, NotImplementedError) as e:
            raise AttributeError(
                f"Modell {model_class.__name__} muss get_metadata() implementieren"
            ) from e
        
        if metadata.name in self._models:
            # Allow re-registration (z.B. bei Tests), aber warne
            import warnings
            warnings.warn(f"Modell '{metadata.name}' wird überschrieben")
        
        self._models[metadata.name] = model_class
        self._metadata[metadata.name] = metadata
    
    def get(self, name: str) -> Type[BaseForecastModel]:
        """
        Holt Modell-Klasse by name.
        
        Args:
            name: Modell-Name (z.B. "ARIMA", "Prophet")
            
        Returns:
            Modell-Klasse
            
        Raises:
            ValueError: Wenn Modell nicht registriert
        """
        if name not in self._models:
            available = ", ".join(sorted(self._models.keys()))
            raise ValueError(
                f"Modell '{name}' nicht registriert. "
                f"Verfügbare Modelle: {available}"
            )
        return self._models[name]
    
    def get_metadata(self, name: str) -> ModelMetadata:
        """
        Holt Metadata für Modell.
        
        Args:
            name: Modell-Name
            
        Returns:
            ModelMetadata
            
        Raises:
            ValueError: Wenn Modell nicht registriert
        """
        if name not in self._metadata:
            raise ValueError(f"Modell '{name}' nicht registriert")
        return self._metadata[name]
    
    def get_all_names(self) -> List[str]:
        """
        Gibt alle registrierten Modell-Namen zurück (sortiert).
        
        Returns:
            Sortierte Liste von Modell-Namen
        """
        return sorted(self._models.keys())
    
    def get_by_category(self, category: ModelCategory) -> Dict[str, Type[BaseForecastModel]]:
        """
        Filtert Modelle nach Kategorie.
        
        Args:
            category: ModelCategory (STATISTICAL, MACHINE_LEARNING, etc.)
            
        Returns:
            Dict {name: model_class} für alle Modelle der Kategorie
        """
        return {
            name: cls
            for name, cls in self._models.items()
            if self._metadata[name].category == category
        }
    
    def get_category_names(self, category: ModelCategory) -> List[str]:
        """
        Gibt Namen aller Modelle einer Kategorie zurück (sortiert).
        
        Args:
            category: ModelCategory
            
        Returns:
            Sortierte Liste von Modell-Namen
        """
        return sorted([
            name for name, metadata in self._metadata.items()
            if metadata.category == category
        ])
    
    def get_suitable_models(
        self,
        data_points: int,
        has_seasonality: bool = False,
        is_stationary: Optional[bool] = None,
        prefer_probabilistic: bool = False
    ) -> List[str]:
        """
        Gibt passende Modelle basierend auf Daten-Eigenschaften zurück.
        
        Intelligente Empfehlung: Filtert Modelle die für die Daten geeignet sind.
        
        Args:
            data_points: Anzahl verfügbarer Datenpunkte
            has_seasonality: Daten haben erkennbare Saisonalität
            is_stationary: Daten sind stationär (None = unbekannt)
            prefer_probabilistic: Bevorzuge Modelle mit Konfidenzintervallen
            
        Returns:
            Liste passender Modell-Namen (sortiert nach Eignung)
        """
        suitable = []
        
        for name, metadata in self._metadata.items():
            # Check: Minimum data points
            if data_points < metadata.min_data_points:
                continue
            
            # Check: Seasonality support (nur wenn Saisonalität vorhanden)
            if has_seasonality and not metadata.supports_seasonality:
                continue
            
            # Check: Stationarity requirement
            if is_stationary is False and metadata.requires_stationarity:
                continue
            
            # Check: Long history requirement
            if metadata.requires_long_history and data_points < 100:
                continue
            
            suitable.append(name)
        
        # Sort: Probabilistic models first if preferred
        if prefer_probabilistic:
            suitable.sort(key=lambda n: (
                not self._metadata[n].is_probabilistic,  # False < True, so probabilistic first
                n  # Then alphabetically
            ))
        else:
            suitable.sort()
        
        return suitable
    
    def count(self) -> int:
        """Gibt Anzahl registrierter Modelle zurück."""
        return len(self._models)
    
    def __len__(self):
        """Ermöglicht len(registry)."""
        return self.count()
    
    def __contains__(self, name: str):
        """Ermöglicht 'ARIMA' in registry."""
        return name in self._models
    
    def __repr__(self):
        """String representation."""
        return f"ModelRegistry(models={len(self._models)})"


# ============================================================================
# GLOBAL REGISTRY INSTANCE (Singleton)
# ============================================================================

_global_registry: Optional[ModelRegistry] = None


# ✨ CACHE: Registry wird nur einmal initialisiert und wiederverwendet
@st.cache_resource
def get_model_registry() -> ModelRegistry:
    """
    Gibt globale Model-Registry zurück (Singleton).
    
    Beim ersten Aufruf wird die Registry initialisiert und alle Modelle
    werden automatisch registriert.
    
    Returns:
        ModelRegistry instance
    """
    global _global_registry
    
    if _global_registry is None:
        _global_registry = ModelRegistry()
        _register_all_models(_global_registry)
    
    return _global_registry


def _register_all_models(registry: ModelRegistry) -> None:
    """
    Registriert automatisch alle Modelle, die von BaseForecastModel erben
    und im Modul 'app.models' definiert sind.
    """
    try:
        # 1. Das Modul dynamisch laden
        module = importlib.import_module("app.models")

        # 2. Alle Klassen im Modul durchlaufen
        for name, obj in inspect.getmembers(module):
            # Prüfen: Ist es eine Klasse? Erbt sie von BaseForecastModel?
            # Und: Ist es nicht die Basisklasse selbst?
            if (
                    inspect.isclass(obj) and
                    issubclass(obj, BaseForecastModel) and
                    obj is not BaseForecastModel
            ):
                try:
                    registry.register(obj)
                except Exception as e:
                    import warnings
                    warnings.warn(f"Fehler beim Registrieren von {name}: {e}")

    except ImportError as e:
        import warnings
        warnings.warn(f"Konnte app.models nicht dynamisch laden: {e}")
    
    except ImportError as e:
        # Wenn models.py nicht importiert werden kann, erstelle leere Registry
        import warnings
        warnings.warn(f"Konnte Modelle nicht importieren: {e}")


def reset_registry() -> None:
    """
    Reset Registry (hauptsächlich für Tests).
    
    ACHTUNG: Löscht alle registrierten Modelle!
    """
    global _global_registry
    _global_registry = None
