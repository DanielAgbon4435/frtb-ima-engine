"""VaR exception tests, an ES score, P&L attribution, and a price-count screen."""

import numpy as np
from scipy import stats


def traffic_light(exceptions, n=250):
    """Basel zones for a 250-day count: 0-4 green, 5-9 amber, 10 or more red.

    A longer sample is scaled back to 250 days before the bucket is chosen.
    """
    scaled = exceptions * 250 / n
    if scaled < 5:
        return "green"
    if scaled < 10:
        return "amber"
    return "red"


def kupiec_pof(exceptions, tail):
    """Kupiec proportion-of-failures likelihood-ratio p-value."""
    exceptions = np.asarray(exceptions, dtype=int)
    n = int(exceptions.size)
    x = int(exceptions.sum())
    p_hat = np.clip(x / n, 1e-12, 1 - 1e-12)
    ll_null = (n - x) * np.log(1 - tail) + x * np.log(tail)
    ll_alt = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)
    stat = max(0.0, -2.0 * (ll_null - ll_alt))
    return float(1 - stats.chi2.cdf(stat, df=1))


def christoffersen_independence(exceptions):
    """Christoffersen independence p-value. Clustered exceptions reject it."""
    hits = np.asarray(exceptions, dtype=int)
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(hits[:-1], hits[1:]):
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1

    def rate(num, den):
        return num / den if den else 0.0

    def loglik(p, n_from, n_to):
        if n_from == 0:
            return 0.0
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return n_to * np.log(p) + (n_from - n_to) * np.log(1 - p)

    p01 = rate(n01, n00 + n01)
    p11 = rate(n11, n10 + n11)
    p = rate(n01 + n11, n00 + n01 + n10 + n11)
    ll_ind = loglik(p, n00 + n01, n01) + loglik(p, n10 + n11, n11)
    ll_alt = loglik(p01, n00 + n01, n01) + loglik(p11, n10 + n11, n11)
    stat = max(0.0, -2.0 * (ll_ind - ll_alt))
    return float(1 - stats.chi2.cdf(stat, df=1))


def var_backtest(losses, var, level=0.99):
    losses = np.asarray(losses, dtype=float)
    var = np.asarray(var, dtype=float)
    if losses.shape != var.shape:
        raise ValueError("losses and VaR forecasts must align")
    hits = (losses > var).astype(int)
    n = int(hits.size)
    x = int(hits.sum())
    return {
        "n": n,
        "exceptions": x,
        "exception_rate": x / n,
        "kupiec_pvalue": kupiec_pof(hits, 1.0 - level),
        "christoffersen_pvalue": christoffersen_independence(hits),
        "traffic_light": traffic_light(x, n),
    }


def es_backtest(losses, var, es, level=0.975):
    """Acerbi-Szekely Z2.

    Z2 = (sum of tail losses) / (n * p * mean ES on those days) - 1,
    with p = 1 - level. Near zero means the ES matches the tail.
    A positive value means the tail was heavier than the forecast.
    """
    losses = np.asarray(losses, dtype=float)
    var = np.asarray(var, dtype=float)
    es = np.asarray(es, dtype=float)
    if not (losses.shape == var.shape == es.shape):
        raise ValueError("losses, VaR and ES must align")
    tail = losses > var
    if not np.any(tail):
        return {"n": int(losses.size), "level": level, "z2": -1.0,
                "mean_tail_loss": float("nan"), "mean_es": float("nan")}
    mean_es = float(np.mean(es[tail]))
    z2 = float(np.sum(losses[tail]) / (losses.size * (1.0 - level) * mean_es) - 1.0)
    return {
        "n": int(losses.size),
        "level": level,
        "z2": z2,
        "mean_tail_loss": float(np.mean(losses[tail])),
        "mean_es": mean_es,
    }


def pla_zone(spearman, ks):
    """Green: Spearman > 0.80 and KS < 0.09.
    Red: Spearman <= 0.70 or KS >= 0.12. Otherwise amber.
    """
    if spearman > 0.80 and ks < 0.09:
        return "green"
    if spearman <= 0.70 or ks >= 0.12:
        return "red"
    return "amber"


def pla_test(hypothetical, risk_theoretical):
    h = np.asarray(hypothetical, dtype=float)
    r = np.asarray(risk_theoretical, dtype=float)
    if h.shape != r.shape or h.ndim != 1 or h.size < 20:
        raise ValueError("P&L series must be aligned and at least length 20")
    spearman = float(stats.spearmanr(h, r).statistic)
    ks = float(stats.ks_2samp(h, r, method="asymp").statistic)
    return {"spearman": spearman, "ks": ks, "zone": pla_zone(spearman, ks), "n": int(h.size)}


def screen_factor(name, observed_day_index, year_length=250):
    """A factor needs 24 real prices in the year and no gap longer than a month.

    This only flags the name. It does not compute the stressed-ES add-on.
    """
    days = np.unique(np.asarray(observed_day_index, dtype=int))
    if days.size == 0:
        return {"name": name, "observations": 0, "max_gap": year_length,
                "modellable": False, "reason": "no real-price observations"}
    gaps = np.diff(days) if days.size > 1 else np.array([year_length])
    max_gap = int(gaps.max())
    if days.size < 24:
        reason = "fewer than 24 real prices in the year"
        modellable = False
    elif max_gap > 20:
        reason = "gap longer than one business month"
        modellable = False
    else:
        reason = "passes the observation screen"
        modellable = True
    return {"name": name, "observations": int(days.size), "max_gap": max_gap,
            "modellable": modellable, "reason": reason}
