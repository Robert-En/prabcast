# setup_module/fanchart.py
"""
Erstellung eines konfigurierbaren Fan-Charts mit Plotly, das Unsicherheitsbänder basierend auf
modellspezifischer RMSE anzeigt. Inklusive Streamlit-UI für flexible Einstellungen und
Rolling-Origin-Backtesting zur RMSE-Schätzung.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statistics import NormalDist

SIGMA_SOURCE_MODEL_RMSE = "Modellunsicherheit (Kreuzvalidiert, RMSE)"
SIGMA_SOURCE_NATIVE_QUANTILES = "Modell-Quantile (nativ)"


def render_fanchart_settings_ui(key_prefix: str = "fanchart") -> dict:
    """Rendert Fan-Chart-UI und liefert validierte Konfiguration zurück.

    Args:
        key_prefix: Präfix für Streamlit-Widget-Keys, um Kollisionen zwischen Tabs zu vermeiden.

    Returns:
        Dictionary mit Fan-Chart-Einstellungen: ``enabled``, ``ci_lower_bound``,
        ``ci_upper_bound``, ``fan_band_count``, ``rolling_min_train``,
        ``rolling_max_folds``, ``smoothing_window`` und ``is_valid``.
    """
    st.markdown("### Fan-Chart Einstellungen")
    use_fan_chart = st.checkbox(
        "Fan Chart mit Konfidenzbändern anzeigen",
        value=True,
        help="Zeigt gestaffelte Konfidenzintervalle rund um die Prognose.",
        key=f"{key_prefix}_enabled",
    )
    with st.expander("Weitere Einstellungen", expanded=False):
        conf_col1, conf_col2, conf_col3 = st.columns(3)
        with conf_col1:
            ci_lower_bound = st.slider(
                "Untere KI-Grenze (%)",
                min_value=50,
                max_value=95,
                value=50,
                step=1,
                help="Kleinste Konfidenzstufe, die im Fan-Chart dargestellt wird.",
                key=f"{key_prefix}_ci_lower",
            )
        with conf_col2:
            ci_upper_bound = st.slider(
                "Obere KI-Grenze (%)",
                min_value=55,
                max_value=99,
                value=95,
                step=1,
                help="Größte Konfidenzstufe, die im Fan-Chart dargestellt wird.",
                key=f"{key_prefix}_ci_upper",
            )
        with conf_col3:
            fan_band_count = st.slider(
                "Anzahl Konfidenzbänder",
                min_value=3,
                max_value=10,
                value=6,
                step=1,
                help="Mehr Bänder erzeugen feinere Abstufungen im Fan-Chart.",
                key=f"{key_prefix}_band_count",
            )

        rolling_col1, rolling_col2 = st.columns(2)
        with rolling_col1:
            rolling_min_train = st.slider(
                "Rolling: Mindest-Trainingspunkte",
                min_value=12,
                max_value=120,
                value=24,
                step=1,
                key=f"{key_prefix}_rolling_min_train",
                help="Minimale Historienlänge vor erstem Rolling-Forecast.",
            )
        with rolling_col2:
            rolling_max_folds = st.slider(
                "Rolling: Maximale Folds",
                min_value=10,
                max_value=200,
                value=60,
                step=5,
                key=f"{key_prefix}_rolling_max_folds",
                help="Begrenzt Rechenzeit für die horizonabhängige RMSE-Schätzung.",
            )

        smoothing_window = st.slider(
            "RMSE-Glättung (Horizonte)",
            min_value=1,
            max_value=6,
            value=3,
            step=1,
            key=f"{key_prefix}_smoothing_window",
            help="Rolling-Mittel über Horizonte vor der cummax-Monotonisierung.",
        )

        if ci_lower_bound >= ci_upper_bound:
            st.error("Die untere KI-Grenze muss kleiner als die obere KI-Grenze sein.")

        return {
            "enabled": use_fan_chart,
            "ci_lower_bound": ci_lower_bound,
            "ci_upper_bound": ci_upper_bound,
            "fan_band_count": fan_band_count,
            "rolling_min_train": rolling_min_train,
            "rolling_max_folds": rolling_max_folds,
            "smoothing_window": smoothing_window,
            "is_valid": ci_lower_bound < ci_upper_bound,
        }


def _compute_rolling_horizon_rmse_profile(
    historical_data,
    horizon: int,
    model_class,
    train_model_func,
    model_params: dict,
    min_train_points: int,
    max_folds: int,
) -> np.ndarray:
    """Schätzt modellspezifische RMSE je Horizont via Rolling-Origin-Backtesting.

    Args:
        historical_data: Historische Zeitreihe (Series oder DataFrame-Spalte).
        horizon: Anzahl Prognose-Horizonte, für die RMSE geschätzt wird.
        model_class: Modellklasse, die ``train_model_func`` instanziiert.
        train_model_func: Callable zum Trainieren eines Modells auf einer Teilserie.
        model_params: Zusätzliche Keyword-Argumente für ``train_model_func``.
        min_train_points: Minimale Trainingslänge vor dem ersten Rolling-Fold.
        max_folds: Maximale Anzahl Rolling-Origin-Folds zur Laufzeitbegrenzung.

    Returns:
        Array der Länge ``horizon`` mit RMSE-Werten pro Horizont; fehlende Werte sind ``nan``.
    """
    series = pd.Series(historical_data).dropna()
    n = len(series)
    rmse_profile = np.full(horizon, np.nan, dtype=float)

    errors_per_horizon = [[] for _ in range(horizon)]

    last_possible_origin = n - horizon
    if last_possible_origin < min_train_points:
        return rmse_profile

    origins = list(range(min_train_points, last_possible_origin + 1))
    if len(origins) > max_folds:
        origins = origins[-max_folds:]

    for origin in origins:
        train_series = series.iloc[:origin]
        try:
            model = train_model_func(model_class, train_series, **model_params)
            pred_series = model.predict(steps=horizon)
        except Exception:
            continue

        for h in range(1, horizon + 1):
            try:
                actual = float(series.iloc[origin + h - 1])
                pred = float(pred_series.iloc[h - 1])
                errors_per_horizon[h - 1].append(actual - pred)
            except Exception:
                continue

    for h_idx, errs in enumerate(errors_per_horizon):
        if errs:
            rmse_profile[h_idx] = float(np.sqrt(np.mean(np.square(errs))))

    return rmse_profile


def _smooth_and_monotonize_profile(
    rmse_profile: np.ndarray, smoothing_window: int
) -> np.ndarray:
    """Glättet RMSE-Profil und macht es mit cummax monoton nicht-fallend.

    Args:
        rmse_profile: Rohes RMSE-Profil je Horizont, ggf. mit ``nan``-Lücken.
        smoothing_window: Fensterbreite für das gleitende Mittel über Horizonte.

    Returns:
        Geglättetes, monoton nicht-fallendes Sigma-Profil gleicher Länge wie ``rmse_profile``.
    """
    profile = rmse_profile.copy()
    valid_mask = np.isfinite(profile) & (profile > 0)
    if not np.any(valid_mask):
        return np.full_like(profile, 1e-6)

    global_fallback = float(np.nanmedian(profile[valid_mask]))
    profile[~valid_mask] = global_fallback
    smoothed = (
        pd.Series(profile)
        .rolling(window=max(1, smoothing_window), min_periods=1, center=True)
        .mean()
        .to_numpy()
    )
    monotonic = np.maximum.accumulate(smoothed)
    monotonic[~np.isfinite(monotonic) | (monotonic <= 0)] = max(global_fallback, 1e-6)
    return monotonic


def _fan_levels_from_config(config: dict) -> np.ndarray:
    """Erzeugt gleichmäßig verteilte Konfidenzstufen aus der Fan-Chart-Konfiguration.

    Args:
        config: Fan-Chart-Konfiguration aus ``render_fanchart_settings_ui``.

    Returns:
        Array mit Konfidenzstufen als Bruchteile (z. B. 0.50 bis 0.95).
    """
    return np.linspace(
        config["ci_lower_bound"] / 100,
        config["ci_upper_bound"] / 100,
        config["fan_band_count"],
    )


def _required_quantile_levels(fan_levels: np.ndarray) -> list[float]:
    """Leitet benötigte Quantil-Stufen aus den Fan-Chart-Konfidenzniveaus ab.

    Args:
        fan_levels: Konfidenzstufen als Bruchteile (z. B. 0.50, 0.80, 0.95).

    Returns:
        Sortierte, eindeutige Liste der Quantil-Stufen ``(1±L)/2``.
    """
    levels = set()
    for level in fan_levels:
        levels.add((1 - level) / 2)
        levels.add((1 + level) / 2)
    return sorted(levels)


def _build_intervals_from_quantiles(
    quantile_map: dict[float, pd.Series], fan_levels: np.ndarray
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Baut Konfidenzintervalle aus nativen Modell-Quantilen.

    Args:
        quantile_map: Mapping Quantil-Stufe → Prognose-Serie je Horizont.
        fan_levels: Konfidenzstufen als Bruchteile.

    Returns:
        Liste von Tupeln ``(level, lower, upper)`` mit NumPy-Arrays je Horizont.
    """
    intervals = []
    for level in fan_levels:
        lower_q = (1 - level) / 2
        upper_q = (1 + level) / 2
        lower = quantile_map[lower_q].values
        upper = quantile_map[upper_q].values
        intervals.append((level, lower, upper))
    return intervals


def _build_intervals_from_rmse(
    forecast, fan_levels: np.ndarray, sigma_profile: np.ndarray
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Baut symmetrische Normal-Intervalle aus einem RMSE-Sigma-Profil.

    Args:
        forecast: Prognose-Serie mit Werten je Horizont.
        fan_levels: Konfidenzstufen als Bruchteile.
        sigma_profile: Sigma-Werte je Horizont.

    Returns:
        Liste von Tupeln ``(level, lower, upper)`` mit NumPy-Arrays je Horizont.
    """
    intervals = []
    for level in fan_levels:
        z_value = NormalDist().inv_cdf(0.5 + level / 2)
        band_delta = z_value * sigma_profile
        lower = forecast.values - band_delta
        upper = forecast.values + band_delta
        intervals.append((level, lower, upper))
    return intervals


def _render_fanchart_bands(
    fig: go.Figure,
    forecast,
    intervals: list[tuple[float, np.ndarray, np.ndarray]],
    model_name: str,
    base_color: str,
) -> go.Figure:
    """Zeichnet verschachtelte Fan-Chart-Bänder auf eine Plotly-Figur.

    Args:
        fig: Bestehende Plotly-Figur.
        forecast: Prognose-Serie mit Index und Werten.
        intervals: Liste von ``(level, lower, upper)`` je Konfidenzstufe.
        model_name: Anzeigename des Modells in Legende und Hover-Text.
        base_color: Hex-Farbe für die Band-Füllung.

    Returns:
        Die Figur mit ergänzten Fan-Chart-Traces.
    """
    rgb = px.colors.hex_to_rgb(base_color)
    legend_added = False
    for idx in range(len(intervals) - 1):
        inner_level, inner_lower, inner_upper = intervals[idx]
        outer_level, outer_lower, outer_upper = intervals[idx + 1]
        opacity = 0.1 + 0.03 * idx

        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=outer_upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=inner_upper,
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                showlegend=False,
                fillcolor=f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})",
                hovertemplate=(
                    f"{model_name}<br>"
                    f"Band: {int(inner_level * 100)}-{int(outer_level * 100)}%<br>"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=inner_lower,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast.index,
                y=outer_lower,
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                name=f"Fan Chart ({model_name})" if not legend_added else None,
                showlegend=not legend_added,
                fillcolor=f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})",
                hoverinfo="skip",
            )
        )
        legend_added = True

    return fig


def apply_fanchart_to_figure(
    fig: go.Figure,
    historical_data,
    forecast,
    config: dict,
    model_name: str = "Modell",
    base_color: str = "#636EFA",
    model_class=None,
    train_model_func=None,
    model_params: dict | None = None,
    fitted_model=None,
) -> go.Figure:
    """Fügt einer Forecast-Figur Fan-Chart-Bänder hinzu.

    Nutzt native Modell-Quantile wenn ``supports_quantiles`` gesetzt ist, sonst
    Rolling-Origin-RMSE-Backtesting.

    Args:
        fig: Bestehende Plotly-Figur, um die Fan-Chart-Traces ergänzt werden.
        historical_data: Historische Zeitreihe für Rolling-Origin-RMSE-Schätzung.
        forecast: Prognose-Serie mit Index und Werten.
        config: Fan-Chart-Konfiguration aus ``render_fanchart_settings_ui``.
        model_name: Anzeigename des Modells in Legende und Hover-Text.
        base_color: Hex-Farbe der Prognoselinie für die Band-Füllung.
        model_class: Modellklasse; ohne Angabe werden keine Bänder gezeichnet.
        train_model_func: Trainings-Callable für Rolling-Backtesting.
        model_params: Optionale Modellparameter für ``train_model_func``.
        fitted_model: Bereits trainiertes Modell für native Quantil-Prognosen.

    Returns:
        Die übergebene Figur mit ergänzten Fan-Chart-Traces, oder unverändert wenn
        Fan-Chart deaktiviert, ungültige Konfiguration, leere Prognose oder fehlende Modell-Callbacks.
    """
    if not config.get("enabled", False):
        return fig
    if not config.get("is_valid", True):
        return fig
    if forecast is None or len(forecast) == 0:
        return fig
    if model_class is None:
        return fig

    horizon = len(forecast)
    fan_levels = _fan_levels_from_config(config)

    if model_class.supports_native_quantiles() and fitted_model is not None:
        try:
            q_levels = _required_quantile_levels(fan_levels)
            quantile_map = fitted_model.predict_quantiles(horizon, q_levels)
            intervals = _build_intervals_from_quantiles(quantile_map, fan_levels)
        except Exception as exc:
            st.error(f"Quantil-Fan-Chart konnte nicht erstellt werden: {exc}")
            return fig
    else:
        if train_model_func is None:
            return fig

        model_params = model_params or {}
        rmse_profile = _compute_rolling_horizon_rmse_profile(
            historical_data=historical_data,
            horizon=horizon,
            model_class=model_class,
            train_model_func=train_model_func,
            model_params=model_params,
            min_train_points=int(config.get("rolling_min_train", 24)),
            max_folds=int(config.get("rolling_max_folds", 60)),
        )
        sigma_profile = _smooth_and_monotonize_profile(
            rmse_profile=rmse_profile,
            smoothing_window=int(config.get("smoothing_window", 3)),
        )
        intervals = _build_intervals_from_rmse(forecast, fan_levels, sigma_profile)

    return _render_fanchart_bands(
        fig, forecast, intervals, model_name, base_color
    )
