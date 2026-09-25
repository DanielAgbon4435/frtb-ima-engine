"""Equal-weight book, its capital number, the backtests, and a one-factor P&L.

Hypothetical P&L is the equal-weight return of names with a price that day.
Risk-theoretical P&L uses betas to the equal-weight return of the first
10 names, estimated on the first 250 sessions and then held fixed.
"""

import numpy as np

from frtb.backtest import es_backtest, pla_test, screen_factor, var_backtest
from frtb.es import capital, expected_shortfall, historical_var


def simple_returns(prices):
    previous = prices.shift(1)
    returns = prices / previous - 1.0
    returns = returns.where(previous > 0)
    return returns.iloc[1:]


def equal_weight_loss(returns):
    """Loss on a unit notional. A name with no return that day is left out."""
    count = returns.notna().sum(axis=1)
    if (count == 0).any():
        raise ValueError("a session has no valid returns")
    portfolio = returns.fillna(0.0).sum(axis=1) / count
    return -portfolio.to_numpy()


def risk_theoretical_pnl(returns, lookback=250, n_factors=10):
    """Hypothetical and risk-theoretical loss after the estimation window."""
    factor_loss = equal_weight_loss(returns.iloc[:, :n_factors])
    market = -factor_loss
    full_return = -equal_weight_loss(returns)
    values = returns.to_numpy()
    betas = np.ones(values.shape[1])
    market_est = market[:lookback]
    for i in range(values.shape[1]):
        hist = values[:lookback, i]
        mask = np.isfinite(hist) & np.isfinite(market_est)
        if int(mask.sum()) < 60:
            continue
        var = float(np.var(market_est[mask]))
        if var < 1e-18:
            continue
        betas[i] = float(np.cov(hist[mask], market_est[mask])[0, 1] / var)
    live = values[lookback:]
    factor = market[lookback:]
    explained = np.where(np.isfinite(live), betas * factor[:, None], 0.0)
    count = np.isfinite(live).sum(axis=1)
    rtpl = -(explained.sum(axis=1) / count)
    hpl = -full_return[lookback:]
    return hpl, rtpl


def modellability(prices, year=250):
    tail = prices.iloc[-year:]
    flags = []
    for name in tail.columns:
        observed = np.flatnonzero(np.isfinite(tail[name].to_numpy()))
        flags.append(screen_factor(name, observed, year_length=year))
    return flags


def run_book(prices, liquidity_horizon=10, lookback=250):
    returns = simple_returns(prices)
    daily_loss = equal_weight_loss(returns)
    losses_10d = np.convolve(daily_loss, np.ones(10), mode="valid")
    report = capital(losses_10d, liquidity_horizon=liquidity_horizon, current_window=lookback)

    var_fc = []
    var_975 = []
    es_fc = []
    realised = []
    for t in range(lookback, daily_loss.size):
        window = daily_loss[t - lookback:t]
        var_fc.append(historical_var(window, 0.99))
        var_975.append(historical_var(window, 0.975))
        es_fc.append(expected_shortfall(window, 0.975))
        realised.append(daily_loss[t])

    hpl, rtpl = risk_theoretical_pnl(returns, lookback)
    # Return dates start at prices.index[1]. The 10-day loss at index k
    # lines up with that return date.
    return_dates = prices.index[1:]
    return {
        "n_names": prices.shape[1],
        "n_sessions": len(prices),
        "sample_start": prices.index[0],
        "sample_end": prices.index[-1],
        "capital": report,
        "stress_start": return_dates[report["stress_start"]],
        "stress_end": return_dates[min(report["stress_end"], len(return_dates) - 1)],
        "var_test": var_backtest(realised, var_fc, 0.99),
        "es_test": es_backtest(realised, var_975, es_fc, 0.975),
        "pla": pla_test(hpl, rtpl),
        "factors": modellability(prices, lookback),
    }
