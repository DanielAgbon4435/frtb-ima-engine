"""Read a Bloomberg Desktop px_last export into a price table.

The file has a ticker row, a field-description row, then one row per Excel
serial date. #N/A is a missing price, including weekends.
"""

from pathlib import Path

import pandas as pd


def load_bloomberg_px(path):
    path = Path(path)
    prices = pd.read_csv(path, na_values=["#N/A", "N/A", ""])

    # First data row is the Bloomberg field description, not a price.
    prices = prices.drop(index=0)

    prices = prices.rename(columns={prices.columns[0]: "Date"})
    prices["Date"] = pd.to_numeric(prices["Date"], errors="coerce")
    prices["Date"] = pd.to_datetime(prices["Date"], unit="D", origin="1899-12-30")

    price_cols = prices.columns[1:]
    prices[price_cols] = prices[price_cols].apply(pd.to_numeric, errors="coerce")

    # Weekends and holidays in this extract are rows of missing prices.
    prices = prices.dropna(how="all", subset=price_cols)
    prices = prices.set_index("Date")
    return prices
