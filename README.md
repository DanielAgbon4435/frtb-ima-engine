# FRTB IMA-style market-risk engine

Expected shortfall, a stressed window, VaR backtests, and a P&L attribution test for one equal-weight equity book.

The book is the frozen top-100 US names from the dissertation, 3 January 2006 to 14 August 2026. Prices are a local Bloomberg `px_last` file with full corporate-action adjustment. That file is not in the repository. Bloomberg prices cannot be redistributed. Pass the path when you run the demo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
PYTHONPATH=src python3 -m frtb.demo /path/to/100_stocks.csv
```

`pytest` checks the formulas on small constructed samples. The demo needs the price file.

## The book

Each day the portfolio is the equal-weight average of names that have a price that day and the day before. A missing name is left out for that day. Loss is minus the portfolio return, on a unit notional. The 10-day loss is the sum of ten overlapping daily losses.

Current expected shortfall is the historical 97.5% ES on the latest 250 of those 10-day losses. The stressed figure is the 250-day window in the whole sample with the largest ES. For this run that window is 11 April 2019 to 8 April 2020. A 250-day window around 15 September 2008 has a 10-day ES of about 22.8%, so the search lands on the COVID year because that year is larger in this sample, not because 2008 was skipped.

Liquidity horizons in the FRTB set are 10, 20, 40, 60 and 120 business days. With one horizon for the whole book, ES is scaled by \(\sqrt{LH/10}\). The number reported below uses a 10-day horizon, so the scale factor is 1. A book with several horizons would scale each risk-factor shock inside the scenario and then take ES. This code scales one ES number.

Capital here is the average of current ES and stressed ES. The regulatory internal-models charge is a weighted sum of unconstrained and constrained ES across risk classes. This repository is the one-book version of that average.

| | |
|---|---|
| Sample | 5,186 sessions, 3 Jan 2006 – 14 Aug 2026 |
| 10-day current ES | 3.81% |
| 10-day stressed ES | 25.11% (11 Apr 2019 – 8 Apr 2020) |
| Capital | 14.46% |

## Backtest

The 99% historical VaR is estimated on a rolling 250-day window and compared with the next day’s loss.

There are 88 exceptions in 4,935 days, an exception rate of 1.78%. Kupiec’s proportion-of-failures test gives p ≈ 6×10⁻⁷. Christoffersen’s independence test gives p ≈ 2×10⁻⁵, so the exceptions also cluster. Both tests reject a well-specified 99% VaR.

The Basel traffic-light zones were set for a 250-day count: 0–4 green, 5–9 amber, 10 or more red. Scaled back to 250 days, 88 exceptions are about 4.5, which is still green. On a twenty-year sample the exception rate is too high, and the exceptions bunch. The green bucket and the rejected tests are both part of the result.

The Acerbi–Szekely \(Z_2\) score on the 97.5% ES is 0.45. A value near zero would mean the ES matches the realised tail. A positive value means the days beyond the 97.5% VaR were heavier than the ES forecast.

## P&L attribution

Hypothetical P&L is the equal-weight return of every name priced that day. Risk-theoretical P&L uses a smaller model: each name’s beta to the equal-weight return of the first 10 names, with those betas fixed on the first 250 sessions, then applied to the rest of the sample.

The two series have Spearman correlation 0.901 and a two-sample KS statistic of 0.017. On the PLA zones used here, green is Spearman above 0.80 and KS below 0.09. Red is Spearman at or below 0.70, or KS at or above 0.12. Anything in between is amber. This book is green: the 10-name factor explains most of the full-book P&L, and the two distributions are close.

## Names with no recent price

A risk factor is treated as non-modellable when it has fewer than 24 real prices in a year, or a gap longer than a business month. The screen only flags those names. It does not add the stressed-ES capital charge that would sit on top of ES.

In the last 250 sessions of this file, 25 of the 100 columns have no usable price. They are names that stopped printing in this frozen 2006 extract: Lehman, Fannie Mae, Freddie Mac, and others. The Oracle column’s last print in the file is 12 July 2013. That is a gap in the extract. Those firms are not being described as unpriced today.

## Files

```
src/frtb/prices.py    Bloomberg price table
src/frtb/es.py        VaR, expected shortfall, liquidity scaling, stressed window
src/frtb/backtest.py  traffic light, Kupiec, Christoffersen, Z2, attribution, price screen
src/frtb/book.py      equal-weight book
src/frtb/demo.py      print the results for a local price file
tests/
```
