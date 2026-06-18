"""Gemeinsame Hilfsfunktionen für native Quantil-Prognosen in Fan-Charts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

TIREX_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def monthly_forecast_index(data: pd.Series, steps: int) -> pd.DatetimeIndex:
    """Erzeugt einen monatlichen Prognoseindex ab dem letzten Datenpunkt.

    Args:
        data: Historische Zeitreihe mit DatetimeIndex.
        steps: Anzahl Prognose-Horizonte.

    Returns:
        Monatlicher DatetimeIndex für die Prognose.
    """
    return pd.date_range(
        data.index[-1] + pd.DateOffset(months=1), periods=steps, freq="ME"
    )


def normal_quantiles_from_forecast(
    mean,
    se,
    levels: list[float],
    index: pd.DatetimeIndex,
) -> dict[float, pd.Series]:
    """Berechnet parametrische Normal-Quantile aus Mittelwert und Standardfehler.

    Args:
        mean: Prognose-Mittelwerte je Horizont.
        se: Standardfehler je Horizont.
        levels: Angefragte Quantil-Stufen.
        index: DatetimeIndex der Prognose.

    Returns:
        Mapping Quantil-Stufe → Prognose-Serie.
    """
    mean_arr = np.asarray(mean, dtype=float).flatten()
    se_arr = np.asarray(se, dtype=float).flatten()
    return {
        level: pd.Series(
            norm.ppf(level, loc=mean_arr, scale=se_arr),
            index=index,
        )
        for level in levels
    }


def sample_quantiles_from_tensor(
    samples: torch.Tensor,
    levels: list[float],
    index: pd.DatetimeIndex,
    sample_dim: int = 1,
) -> dict[float, pd.Series]:
    """Leitet Quantile aus einem Sample-Tensor ab.

    Args:
        samples: Forecast-Samples (typisch ``[batch, num_samples, horizon]``).
        levels: Angefragte Quantil-Stufen.
        index: DatetimeIndex der Prognose.
        sample_dim: Achse der stochastischen Samples.

    Returns:
        Mapping Quantil-Stufe → Prognose-Serie.
    """
    if samples.dim() == 2:
        sample_dim = 0

    result: dict[float, pd.Series] = {}
    for level in levels:
        q_values = torch.quantile(samples, level, dim=sample_dim).flatten()
        values = q_values.detach().cpu().numpy()
        result[level] = pd.Series(values[: len(index)], index=index)
    return result


def confidence_level_from_quantile(q: float) -> float:
    """Leitet ein symmetrisches Konfidenzniveau aus einer Quantil-Stufe ab.

    Args:
        q: Quantil-Stufe (unteres Quantil < 0.5, oberes >= 0.5).

    Returns:
        Konfidenzniveau als Bruchteil (z. B. 0.95 für das 95%-Intervall).
    """
    if q < 0.5:
        return round(1 - 2 * q, 6)
    return round(2 * q - 1, 6)


def unique_confidence_levels(quantile_levels: list[float]) -> set[float]:
    """Sammelt eindeutige Konfidenzniveaus aus Quantil-Anfragen.

    Args:
        quantile_levels: Angefragte Quantil-Stufen.

    Returns:
        Eindeutige symmetrische Konfidenzniveaus.
    """
    return {confidence_level_from_quantile(q) for q in quantile_levels}


def interpolate_quantile_levels(
    values: np.ndarray,
    fixed_levels: list[float],
    requested_levels: list[float],
) -> dict[float, np.ndarray]:
    """Interpoliert Quantil-Prognosen auf festen Stützstellen.

    Args:
        values: Array der Form ``[horizon, num_quantiles]`` oder
            ``[num_quantiles, horizon]`` mit optionaler Batch-Dimension.
        fixed_levels: Stützstellen der nativen Modell-Quantile.
        requested_levels: Angefragte Quantil-Stufen.

    Returns:
        Mapping Quantil-Stufe → Werte je Horizont.
    """
    arr = np.asarray(values, dtype=float)
    fixed = np.asarray(fixed_levels, dtype=float)
    num_fixed_levels = len(fixed)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            singleton_axes = [axis for axis, size in enumerate(arr.shape) if size == 1]
            if singleton_axes:
                arr = np.squeeze(arr, axis=singleton_axes[0])

    if arr.ndim != 2:
        raise ValueError(
            f"Erwartete Form [horizon, quantiles] oder [quantiles, horizon], "
            f"erhalten: {arr.shape}"
        )

    if arr.shape[1] == num_fixed_levels:
        horizon_quantiles = arr
    elif arr.shape[0] == num_fixed_levels:
        horizon_quantiles = arr.T
    else:
        raise ValueError(
            f"Quantil-Achse passt nicht zu den Stützstellen: Werteform {arr.shape}, "
            f"{num_fixed_levels} Quantil-Stützstellen"
        )

    result: dict[float, np.ndarray] = {}
    for q in requested_levels:
        q_clamped = float(np.clip(q, fixed.min(), fixed.max()))
        result[q] = np.array(
            [
                np.interp(q_clamped, fixed, horizon_quantiles[t])
                for t in range(horizon_quantiles.shape[0])
            ]
        )
    return result
