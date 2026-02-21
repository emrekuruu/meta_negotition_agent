"""
Forecasting evaluation metrics using established libraries.

Metrics:
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error
- CRPS: Continuous Ranked Probability Score
- MASE: Mean Absolute Scaled Error
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from typing import Optional


def calculate_rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    Args:
        actual: Ground truth values
        forecast: Predicted values

    Returns:
        RMSE value

    Raises:
        ValueError: If arrays are empty or have mismatched shapes
    """
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    if len(actual) == 0 or len(forecast) == 0:
        raise ValueError("Input arrays cannot be empty")

    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape} vs forecast {forecast.shape}")

    return np.sqrt(mean_squared_error(actual, forecast))


def calculate_mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).

    Args:
        actual: Ground truth values
        forecast: Predicted values

    Returns:
        MAPE value (as percentage, 0-100)

    Raises:
        ValueError: If arrays are empty, have mismatched shapes, or all actual values are near zero
    """
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    if len(actual) == 0 or len(forecast) == 0:
        raise ValueError("Input arrays cannot be empty")

    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape} vs forecast {forecast.shape}")

    # Check for near-zero values which would cause division issues
    if not np.any(np.abs(actual) > 1e-10):
        raise ValueError("All actual values are near zero - MAPE undefined")

    # Filter out near-zero actual values to avoid division by zero
    mask = np.abs(actual) > 1e-10

    if not np.any(mask):
        raise ValueError("No valid values after filtering near-zero actuals")

    # sklearn returns MAPE as a ratio (0-1), we convert to percentage (0-100)
    return mean_absolute_percentage_error(actual[mask], forecast[mask]) * 100


def calculate_crps(actual: np.ndarray, forecast: np.ndarray, ensemble_members: Optional[np.ndarray] = None) -> float:
    """
    Calculate Continuous Ranked Probability Score (CRPS).

    For deterministic forecasts, CRPS reduces to MAE.
    For ensemble forecasts, use the ensemble_members parameter.

    Args:
        actual: Ground truth values (1D array)
        forecast: Point forecast or ensemble mean (1D array)
        ensemble_members: Optional ensemble forecasts (2D array: samples x ensemble_size)

    Returns:
        CRPS value

    Raises:
        ValueError: If arrays are empty or have mismatched shapes
    """
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)

    if len(actual) == 0 or len(forecast) == 0:
        raise ValueError("Input arrays cannot be empty")

    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape} vs forecast {forecast.shape}")

    if ensemble_members is not None:
        try:
            from properscoring import crps_ensemble
            # ensemble_members should be (n_samples, n_ensemble_members)
            ensemble_members = np.asarray(ensemble_members)
            if ensemble_members.shape[0] != len(actual):
                raise ValueError(f"Ensemble shape mismatch: got {ensemble_members.shape[0]} samples, expected {len(actual)}")
            return np.mean([crps_ensemble(actual[i], ensemble_members[i]) for i in range(len(actual))])
        except ImportError:
            raise ImportError("properscoring library required for ensemble CRPS. Install with: pip install properscoring")
    else:
        # Deterministic forecast: CRPS = MAE
        return np.mean(np.abs(actual - forecast))


def calculate_mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    training_series: np.ndarray,
    seasonality: int = 1
) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE) using sktime library.

    MASE scales the MAE by the MAE of a naive seasonal forecast.
    A value < 1 indicates the forecast is better than the naive baseline.

    Args:
        actual: Ground truth values for test set
        forecast: Predicted values for test set
        training_series: Historical training data used to compute scaling factor
        seasonality: Seasonal period (1 for non-seasonal data)

    Returns:
        MASE value

    Raises:
        ValueError: If arrays are empty, have mismatched shapes, or training series is too short
        ImportError: If sktime is not installed
    """
    actual = np.asarray(actual)
    forecast = np.asarray(forecast)
    training_series = np.asarray(training_series)

    if len(actual) == 0 or len(forecast) == 0:
        raise ValueError("Input arrays cannot be empty")

    if actual.shape != forecast.shape:
        raise ValueError(f"Shape mismatch: actual {actual.shape} vs forecast {forecast.shape}")

    if len(training_series) < seasonality + 1:
        raise ValueError(f"Training series too short: need at least {seasonality + 1} points for seasonality={seasonality}")

    try:
        from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError
    except ImportError:
        raise ImportError("sktime library required for MASE. Install with: pip install sktime")

    # Use sktime's proper MASE implementation
    mase_metric = MeanAbsoluteScaledError(sp=seasonality)
    mase_value = mase_metric(y_true=actual, y_pred=forecast, y_train=training_series)

    return float(mase_value)


def calculate_all_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
    training_series: Optional[np.ndarray] = None,
    ensemble_members: Optional[np.ndarray] = None,
    seasonality: int = 1
) -> dict:
    """
    Calculate all available metrics for a forecast.

    Args:
        actual: Ground truth values
        forecast: Predicted values
        training_series: Optional training data for MASE
        ensemble_members: Optional ensemble forecasts for CRPS
        seasonality: Seasonal period for MASE

    Returns:
        Dictionary with metric names as keys and values as floats
        Metrics that cannot be calculated are omitted
    """
    metrics = {}

    try:
        metrics['rmse'] = calculate_rmse(actual, forecast)
    except (ValueError, Exception):
        pass

    try:
        metrics['mape'] = calculate_mape(actual, forecast)
    except (ValueError, Exception):
        pass

    try:
        metrics['crps'] = calculate_crps(actual, forecast, ensemble_members)
    except (ValueError, ImportError, Exception):
        pass

    if training_series is not None:
        try:
            metrics['mase'] = calculate_mase(actual, forecast, training_series, seasonality)
        except (ValueError, Exception):
            pass

    return metrics
