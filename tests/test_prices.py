from pathlib import Path

from frtb.prices import load_bloomberg_px


def test_loader_drops_blank_sessions_and_maps_excel_dates(tmp_path: Path):
    path = tmp_path / "px.csv"
    path.write_text(
        "DATES,AAA UN Equity,BBB UN Equity\n"
        "note,px,px\n"
        "38720,10,20\n"
        "38721,#N/A,#N/A\n"
        "38722,11,22\n"
    )
    prices = load_bloomberg_px(path)
    assert list(prices.columns) == ["AAA UN Equity", "BBB UN Equity"]
    assert len(prices) == 2
    assert str(prices.index[0])[:10] == "2006-01-03"
    assert prices.isna().sum().sum() == 0
    assert prices.iloc[1, 0] == 11
