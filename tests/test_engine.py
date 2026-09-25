import numpy as np
import pytest

from frtb.backtest import christoffersen_independence, kupiec_pof, pla_test, pla_zone, screen_factor, traffic_light, var_backtest
from frtb.es import capital, expected_shortfall, historical_var, scale_es


def test_var_and_es_on_a_known_sample():
    losses = np.arange(1, 101, dtype=float)
    assert historical_var(losses, 0.99) == pytest.approx(99.01)
    tail = losses[losses >= historical_var(losses, 0.975)]
    assert expected_shortfall(losses, 0.975) == pytest.approx(tail.mean())


def test_liquidity_scaling_matches_square_root():
    assert scale_es(10.0, 10) == pytest.approx(10.0)
    assert scale_es(10.0, 40) == pytest.approx(20.0)


def test_stressed_window_finds_the_volatile_year():
    calm = np.abs(np.random.default_rng(1).normal(1.0, 0.2, 250))
    storm = np.abs(np.random.default_rng(1).normal(5.0, 1.0, 250))
    losses = np.concatenate([calm, storm, calm])
    report = capital(losses, liquidity_horizon=10, current_window=250)
    # Monthly grid, so the window need not start on day 250.
    # It should sit on the volatile middle year.
    assert 200 <= report["stress_start"] <= 250
    assert report["es_stressed_10d"] > report["es_current_10d"]
    assert report["capital"] == pytest.approx(0.5 * (report["es_current"] + report["es_stressed"]))


def test_kupiec_accepts_the_null_rate():
    rng = np.random.default_rng(0)
    hits = rng.random(2000) < 0.01
    assert kupiec_pof(hits.astype(int), 0.01) > 0.05


def test_christoffersen_rejects_clustered_exceptions():
    hits = np.zeros(200, dtype=int)
    hits[20:40] = 1
    assert christoffersen_independence(hits) < 0.01


def test_traffic_light_buckets():
    assert traffic_light(4, 250) == "green"
    assert traffic_light(9, 250) == "amber"
    assert traffic_light(10, 250) == "red"


def test_pla_zones():
    assert pla_zone(0.90, 0.05) == "green"
    assert pla_zone(0.75, 0.10) == "amber"
    assert pla_zone(0.60, 0.05) == "red"
    rng = np.random.default_rng(3)
    h = rng.normal(size=400)
    result = pla_test(h, h + rng.normal(0, 0.01, 400))
    assert result["zone"] == "green"
    assert result["spearman"] > 0.95


def test_nmrf_screen():
    good = screen_factor("eq", np.arange(250))
    bad = screen_factor("spread", np.array([0, 30, 80]))
    assert good["modellable"]
    assert not bad["modellable"]


def test_var_backtest_green_on_well_specified_gaussian():
    rng = np.random.default_rng(4)
    losses = rng.normal(0, 1, 800)
    var = np.full(losses.shape, np.quantile(rng.normal(0, 1, 200_000), 0.99))
    report = var_backtest(losses, var, 0.99)
    assert report["traffic_light"] == "green"
    assert report["kupiec_pvalue"] > 0.01
