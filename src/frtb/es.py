"""Historical VaR, expected shortfall, and the one-horizon capital number."""

import math

import numpy as np


HORIZONS = (10, 20, 40, 60, 120)


def historical_var(losses, level=0.99):
    """Loss quantile. level=0.99 is the 99% VaR."""
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or losses.size < 20:
        raise ValueError("losses must be a 1-d sample of at least 20 observations")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    return float(np.quantile(losses, level, method="linear"))


def expected_shortfall(losses, level=0.975):
    """Mean loss at or beyond the quantile. FRTB uses 97.5%."""
    losses = np.asarray(losses, dtype=float)
    var = historical_var(losses, level)
    tail = losses[losses >= var - 1e-15]
    if tail.size == 0:
        raise ValueError("empty tail")
    return float(np.mean(tail))


def scale_es(es_10d, liquidity_horizon):
    """ES(LH) = ES(10) * sqrt(LH / 10), for one horizon on the whole book."""
    if liquidity_horizon not in HORIZONS:
        raise ValueError(f"liquidity horizon must be one of {HORIZONS}")
    if es_10d < 0:
        raise ValueError("ES on a loss series should be non-negative")
    return es_10d * math.sqrt(liquidity_horizon / 10)


def max_es_window(losses, window=250, level=0.975, step=21):
    """12-month window with the largest ES. step=21 is a monthly grid."""
    losses = np.asarray(losses, dtype=float)
    if losses.size < window:
        raise ValueError("sample shorter than the stressed window")
    best = None
    for start in range(0, losses.size - window + 1, step):
        es = expected_shortfall(losses[start:start + window], level)
        if best is None or es > best[2]:
            best = (start, start + window, es)
    return best


def capital(losses_10d, liquidity_horizon=10, current_window=250, level=0.975):
    """Average of current ES and the worst 12-month ES, after horizon scaling.

    The regulatory charge across risk classes is a weighted sum of
    unconstrained and constrained ES. This is the one-horizon average.
    """
    losses_10d = np.asarray(losses_10d, dtype=float)
    if losses_10d.size < current_window:
        raise ValueError("need at least one current window of 10-day losses")
    es_current = expected_shortfall(losses_10d[-current_window:], level)
    start, end, es_stressed = max_es_window(losses_10d, current_window, level)
    es_current_h = scale_es(es_current, liquidity_horizon)
    es_stressed_h = scale_es(es_stressed, liquidity_horizon)
    return {
        "es_current_10d": es_current,
        "es_stressed_10d": es_stressed,
        "liquidity_horizon": liquidity_horizon,
        "es_current": es_current_h,
        "es_stressed": es_stressed_h,
        "capital": 0.5 * (es_current_h + es_stressed_h),
        "stress_start": start,
        "stress_end": end,
    }
