"""
forecast_evaluation.py - Forecast Evaluation & Backtesting Module
================================================================

Reusable functions for computing the Mean Absolute Scaled Error (MASE)
and running expanding-window backtests on univariate time series models.

References
----------
Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of
    forecast accuracy. International Journal of Forecasting, 22(4), 679-688.
West, K. D. (1996). Asymptotic inference about predictive ability.
    Econometrica, 64(5), 1067-1084.
Clark, T. E., & McCracken, M. W. (2001). Tests of equal forecast accuracy
    and encompassing for nested models. Journal of Econometrics, 105(1),
    85-110.

Author: [Your Name]
Course: ECON 5200, Lab 21
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable


def compute_mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    insample: np.ndarray,
    m: int = 1,
) -> float:
    """Compute the Mean Absolute Scaled Error (Hyndman & Koehler, 2006).

    MASE scales out-of-sample forecast errors by the in-sample mean
    absolute error of a seasonal-naive benchmark. It is scale-free,
    symmetric, and well-defined even when actuals contain zeros.

    Interpretation
    --------------
    MASE < 1 : the model outperforms the seasonal-naive benchmark.
    MASE = 1 : the model is equivalent to a random walk of period m.
    MASE > 1 : the seasonal-naive benchmark is more accurate.

    Parameters
    ----------
    actual : np.ndarray
        Realized values over the forecast horizon, shape (h,).
    forecast : np.ndarray
        Point forecasts aligned with `actual`, shape (h,).
    insample : np.ndarray
        Training observations used to compute the naive benchmark MAE.
    m : int, default 1
        Seasonal period. Use 1 for random-walk benchmark, 12 for monthly
        seasonal data, 4 for quarterly, 7 for daily-with-weekly-cycle.

    Returns
    -------
    float
        The MASE statistic.

    Raises
    ------
    ValueError
        If array shapes are incompatible, m is invalid, or the naive MAE
        is zero (degenerate benchmark).
    """
    actual   = np.asarray(actual,   dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    insample = np.asarray(insample, dtype=float)

    if actual.shape != forecast.shape:
        raise ValueError(
            f'actual and forecast must have the same shape; '
            f'got {actual.shape} vs {forecast.shape}'
        )
    if m < 1:
        raise ValueError(f'Seasonal period m must be >= 1; got {m}')
    if len(insample) <= m:
        raise ValueError(
            f'insample length ({len(insample)}) must exceed m ({m})'
        )

    # Numerator: out-of-sample MAE of the model
    mae_forecast = np.mean(np.abs(actual - forecast))

    # Denominator: in-sample MAE of the seasonal-naive benchmark.
    # For m=1 this is the MAE of first differences; for m=12 it is the
    # MAE of the year-over-year differences.
    naive_errors = insample[m:] - insample[:-m]
    mae_naive = np.mean(np.abs(naive_errors))

    if mae_naive == 0:
        raise ValueError(
            'Naive benchmark MAE is zero; MASE is undefined. '
            'This typically indicates a constant insample series.'
        )

    return float(mae_forecast / mae_naive)


def backtest_expanding_window(
    series: pd.Series,
    model_fn: Callable[[pd.Series], np.ndarray],
    min_train: int = 120,
    horizon: int = 12,
    step: int = 12,
    seasonal_period: int = 1,
) -> pd.DataFrame:
    """Expanding-window pseudo-out-of-sample backtest.

    At each origin t, fits the model on `series[:t]` and evaluates its
    h-step-ahead forecast against `series[t : t+horizon]`. Returns
    per-origin error metrics plus a pooled summary row.

    This follows the expanding-window protocol of West (1996). Setting
    `step = horizon` yields non-overlapping forecast windows, which is
    required for the forecast errors to be (approximately) serially
    uncorrelated -- a prerequisite for Diebold-Mariano tests.

    Parameters
    ----------
    series : pd.Series
        Univariate time series with a DatetimeIndex (monotonic,
        equally spaced). Must not contain NaNs.
    model_fn : Callable[[pd.Series], np.ndarray]
        Function that takes a training series and returns a 1-D numpy
        array of point forecasts of length exactly `horizon`. The caller
        is responsible for model specification; this function makes no
        assumptions about ARIMA, ETS, or other model families.
    min_train : int, default 120
        Size of the initial training window. For monthly macro data,
        120 = 10 years is a common default.
    horizon : int, default 12
        Number of periods to forecast at each origin.
    step : int, default 12
        Number of observations by which the training window expands
        between origins. `step == horizon` gives non-overlapping
        evaluation windows; `step < horizon` overlaps forecasts.
    seasonal_period : int, default 1
        Passed to compute_mase as `m`.

    Returns
    -------
    pd.DataFrame
        One row per origin plus a final 'POOLED' row. Columns:
        - origin_date : timestamp at which training ends
        - train_size  : number of training observations
        - mae, rmse, mape : standard accuracy metrics
        - mase        : Mean Absolute Scaled Error

    Raises
    ------
    ValueError
        If inputs are malformed or insufficient data is available.
    """
    if not isinstance(series, pd.Series):
        raise TypeError('series must be a pandas Series')
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError('series must have a DatetimeIndex')
    if series.isna().any():
        raise ValueError('series contains NaNs; clean the data first')
    if len(series) < min_train + horizon:
        raise ValueError(
            f'Need at least {min_train + horizon} observations; '
            f'got {len(series)}'
        )
    if horizon < 1 or step < 1 or min_train < 1:
        raise ValueError('horizon, step, and min_train must all be >= 1')

    records = []
    n = len(series)

    # Iterate over expanding origins. Origin t means: train on series[:t],
    # forecast series[t : t+horizon].
    for t in range(min_train, n - horizon + 1, step):
        train  = series.iloc[:t]
        actual = series.iloc[t : t + horizon].to_numpy()

        # Delegate fitting and forecasting to the user-supplied callable
        forecast = np.asarray(model_fn(train), dtype=float)

        if forecast.shape != (horizon,):
            raise ValueError(
                f'model_fn returned shape {forecast.shape}; '
                f'expected ({horizon},)'
            )

        err = actual - forecast
        mae  = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        # Guard MAPE against zero actuals
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = float(np.mean(np.abs(err / actual)) * 100) \
                   if np.all(actual != 0) else np.nan
        mase = compute_mase(actual, forecast, train.to_numpy(),
                            m=seasonal_period)

        records.append({
            'origin_date': train.index[-1],
            'train_size' : t,
            'mae'        : mae,
            'rmse'       : rmse,
            'mape'       : mape,
            'mase'       : mase,
        })

    results = pd.DataFrame(records)

    # Pooled summary row: average metrics across all origins
    pooled = pd.DataFrame([{
        'origin_date': 'POOLED',
        'train_size' : np.nan,
        'mae'        : results['mae'].mean(),
        'rmse'       : results['rmse'].mean(),
        'mape'       : results['mape'].mean(),
        'mase'       : results['mase'].mean(),
    }])

    return pd.concat([results, pooled], ignore_index=True)


# ---------------------------------------------------------------------------
# Self-test: runs when the module is executed directly, not when imported.
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Running self-tests for forecast_evaluation.py ...\n')

    # ----- Test 1: compute_mase on a trivial example -----
    # Random walk insample => naive MAE = mean(|increments|)
    rng = np.random.default_rng(seed=42)
    rw = np.cumsum(rng.standard_normal(200))
    actual   = rw[-12:]
    forecast = np.full(12, rw[-13])           # persistence forecast
    mase = compute_mase(actual, forecast, rw[:-12], m=1)
    print(f'Test 1 -- persistence forecast on random walk: MASE = {mase:.3f}')
    print('         (expected close to 1.0 by construction)')
    assert 0.5 < mase < 2.0, 'Persistence MASE should be near 1'

    # ----- Test 2: perfect forecast => MASE = 0 -----
    perfect = compute_mase(actual, actual, rw[:-12], m=1)
    print(f'Test 2 -- perfect forecast: MASE = {perfect:.6f}')
    assert perfect == 0.0

    # ----- Test 3: backtest with a trivial last-value model -----
    idx = pd.date_range('2010-01-01', periods=200, freq='MS')
    s   = pd.Series(rw, index=idx)

    def last_value_model(train: pd.Series) -> np.ndarray:
        """Forecast the next 12 months as the last observed value."""
        return np.full(12, train.iloc[-1])

    out = backtest_expanding_window(
        s, last_value_model,
        min_train=120, horizon=12, step=12, seasonal_period=1,
    )
    print(f'\nTest 3 -- backtest ran {len(out) - 1} origins:')
    print(out.round(3))

    print('\nAll self-tests passed.')
