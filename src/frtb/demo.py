"""Print the capital number, the backtests, and the attribution result."""

import sys

from frtb.book import run_book
from frtb.prices import load_bloomberg_px


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python3 -m frtb.demo path/to/100_stocks.csv")
    result = run_book(load_bloomberg_px(sys.argv[1]), liquidity_horizon=10)
    capital = result["capital"]
    var_test = result["var_test"]
    es_test = result["es_test"]
    pla = result["pla"]
    failed = [flag for flag in result["factors"] if not flag["modellable"]]
    print(f"names: {result['n_names']}   sessions: {result['n_sessions']}")
    print(f"sample: {str(result['sample_start'])[:10]} to {str(result['sample_end'])[:10]}")
    print(f"liquidity horizon: {capital['liquidity_horizon']} days")
    print(f"current ES:  {capital['es_current']:.4f}")
    print(
        f"stressed ES: {capital['es_stressed']:.4f}  "
        f"({str(result['stress_start'])[:10]} to {str(result['stress_end'])[:10]})"
    )
    print(f"capital:     {capital['capital']:.4f}")
    print(
        f"VaR backtest: {var_test['exceptions']}/{var_test['n']} exceptions, "
        f"Kupiec p={var_test['kupiec_pvalue']:.3g}, "
        f"Christoffersen p={var_test['christoffersen_pvalue']:.3g}, "
        f"traffic light {var_test['traffic_light']}"
    )
    print(f"ES Z2: {es_test['z2']:.3f}")
    print(f"PLA: Spearman={pla['spearman']:.3f}, KS={pla['ks']:.3f}, {pla['zone']}")
    print(f"non-modellable in the last year: {len(failed)}")
    for flag in failed:
        print(f"  {flag['name']}: {flag['reason']}")


if __name__ == "__main__":
    main()
