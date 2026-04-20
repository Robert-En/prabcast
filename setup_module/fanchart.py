# setup_module/fanchart.py
"""
Erstellung eines konfigurierbaren Fan-Charts mit Plotly, das Unsicherheitsbänder basierend auf historischer
Datenvolatilität oder modellspezifischer RMSE anzeigt. Inklusive Streamlit-UI für flexible Einstellungen und
Rolling-Origin-Backtesting zur RMSE-Schätzung.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statistics import NormalDist

SIGMA_SOURCE_MODEL_RMSE = "Modellunsicherheit (Kreuzvalidiert, RMSE)"


def render_fanchart_settings_ui(key_prefix: str = "fanchart") -> dict:
    """Rendert Fan-Chart-UI und liefert validierte Konfiguration zurück."""
    st.markdown("### Fan-Chart Einstellungen")
    use_fan_chart = st.checkbox(
        "Fan Chart mit Konfidenzbändern anzeigen",
        value=True,
        help="Zeigt gestaffelte Konfidenzintervalle rund um die Prognose.",
        key=f"{key_prefix}_enabled",
    )

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


def _compute_historical_sigma_profile(historical_data, horizon: int) -> np.ndarray:
    """Berechnet die Standardabweichung aus historischer Datenbasis mit Wurzel(h)-Skalierung."""
    hist_values = np.asarray(historical_data.values, dtype=float)
    sigma = (
        float(np.nanstd(hist_values, ddof=1))
        if len(hist_values) > 1
        else float(np.nanstd(hist_values))
    )
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1e-6
    horizon_scale = np.sqrt(np.arange(1, horizon + 1))
    return sigma * horizon_scale


def _compute_rolling_horizon_rmse_profile(
    historical_data,
    horizon: int,
    model_class,
    train_model_func,
    model_params: dict,
    min_train_points: int,
    max_folds: int,
) -> np.ndarray:
    """Schätzt modellspezifische RMSE je Horizont via Rolling-Origin-Backtesting."""
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
    """Glättet RMSE-Profil und macht es mit cummax monoton nicht-fallend."""
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
) -> go.Figure:
    """Fügt einer Forecast-Figur Fan-Chart-Bänder mit konfigurierbarer Sigma-Quelle hinzu."""
    if not config.get("enabled", False):
        return fig
    if not config.get("is_valid", True):
        return fig
    if forecast is None or len(forecast) == 0:
        return fig

    model_params = model_params or {}
    horizon = len(forecast)
    sigma_profile = _compute_historical_sigma_profile(historical_data, horizon)

    if model_class is not None and train_model_func is not None:
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

    fan_levels = np.linspace(
        config["ci_lower_bound"] / 100,
        config["ci_upper_bound"] / 100,
        config["fan_band_count"],
    )

    intervals = []
    for level in fan_levels:
        z_value = NormalDist().inv_cdf(0.5 + level / 2)
        band_delta = z_value * sigma_profile
        lower = forecast.values - band_delta
        upper = forecast.values + band_delta
        intervals.append((level, lower, upper))

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
