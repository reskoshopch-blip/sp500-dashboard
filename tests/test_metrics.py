import math

import pandas as pd
import pytest

from src.analytics.aggregates import aggregate, balanced_panel, yoy_like_for_like
from src.analytics.metrics import NA, OK, PARTIAL, compute_metrics, compute_metrics_row, item_changes, total_debt
from src.data_sources.sp500 import members_at, parse_changes_table


def m(row, prev=None):
    return {k: (v, s, n) for k, v, s, n in compute_metrics_row(row, prev)}


BASE = {"total_assets": 1000.0, "total_liabilities": 600.0, "equity_total": 400.0, "current_assets": 300.0,
        "current_liabilities": 150.0, "cash": 60.0, "st_investments": 20.0, "receivables": 70.0,
        "ltd_noncurrent": 200.0, "debt_current_total": 50.0, "goodwill": 100.0}


def test_basic_ratios():
    r = m(BASE)
    assert r["total_debt"] == (250.0, OK, "ltd_noncurrent + debt_current_total")
    assert r["debt_to_equity"][0] == pytest.approx(0.625)
    assert r["liabilities_to_assets"][0] == pytest.approx(0.6)
    assert r["equity_ratio"][0] == pytest.approx(0.4)
    assert r["current_ratio"][0] == pytest.approx(2.0)
    assert r["quick_ratio"] == (pytest.approx(1.0), OK, "")
    assert r["working_capital"][0] == 150.0
    assert r["net_debt"][0] == 190.0
    assert r["goodwill_to_assets"][0] == pytest.approx(0.1)
    assert r["cash_to_debt"][0] == pytest.approx(0.24)


def test_growth_needs_prior_year_and_positive_base():
    assert m(BASE)["assets_growth"][1] == NA
    r = m(BASE, {**BASE, "total_assets": 800.0, "equity_total": -10.0})
    assert r["assets_growth"][0] == pytest.approx(0.25)
    assert r["equity_growth"][1] == NA and "negativ" in r["equity_growth"][2]


def test_division_by_zero_and_negative_equity():
    r = m({**BASE, "current_liabilities": 0.0, "equity_total": -5.0})
    assert r["current_ratio"][1] == NA and "null" in r["current_ratio"][2]
    assert r["debt_to_equity"][1] == NA
    assert r["liabilities_to_equity"][1] == NA


def test_missing_inputs_are_never_zero():
    row = {k: v for k, v in BASE.items() if k not in ("goodwill", "st_investments")}
    r = m(row)
    assert r["goodwill_to_assets"][0] is None and r["goodwill_to_assets"][1] == NA
    assert r["quick_ratio"][1] == PARTIAL and "st_investments" in r["quick_ratio"][2]
    assert r["quick_ratio"][0] == pytest.approx((60 + 70) / 150)


def test_banks_without_classified_balance_sheet():
    r = m({"total_assets": 1000.0, "total_liabilities": 900.0, "equity_total": 100.0})
    assert r["current_ratio"][1] == NA and "Banken" in r["current_ratio"][2]
    assert r["working_capital"][1] == NA
    assert r["total_debt"][1] == NA


def test_total_debt_definitions_never_double_count():
    assert total_debt({"ltd_total": 300.0, "st_debt": 10.0, "ltd_current": 40.0})[0] == 310.0
    v, s, _ = total_debt({"ltd_total": 300.0})
    assert v == 300.0 and s == PARTIAL
    v, s, _ = total_debt({"ltd_noncurrent": 200.0, "st_debt": 5.0, "ltd_current": 20.0, "debt_current_total": 25.0})
    assert v == 225.0 and s == OK
    r = m({**BASE, "ltd_noncurrent": 200.0, "debt_current_total": None, "st_debt": 5.0})
    assert r["total_debt"][1] == PARTIAL and r["net_debt"][1] == NA
    assert m({**BASE, "ltd_noncurrent": 0.0, "debt_current_total": 0.0})["cash_to_debt"][1] == NA


def test_compute_metrics_long_format():
    wide = pd.DataFrame([{"cik": 1, "fiscal_year": 2022, **BASE}, {"cik": 1, "fiscal_year": 2023, **BASE,
                                                                     "total_assets": 1100.0}])
    out = compute_metrics(wide)
    g = out[(out["fiscal_year"] == 2023) & (out["metric_key"] == "assets_growth")]
    assert g["value"].iloc[0] == pytest.approx(0.1)
    assert out[(out["fiscal_year"] == 2022) & (out["metric_key"] == "assets_growth")]["status"].iloc[0] == NA


def items_long():
    rows = []
    for cik, vals in {1: {2022: 100, 2023: 110}, 2: {2022: 50, 2023: None}, 3: {2023: 30}}.items():
        for y, v in vals.items():
            if v is not None:
                rows.append((cik, y, "total_assets", float(v)))
                rows.append((cik, y, "total_liabilities", float(v) / 2))
    return pd.DataFrame(rows, columns=["cik", "fiscal_year", "item_key", "value"])


def test_aggregate_available_reports_coverage_and_missing():
    a = aggregate(items_long(), ["total_assets"], [2022, 2023], {1, 2, 3, 4})
    r23 = a[a["fiscal_year"] == 2023].iloc[0]
    assert r23["sum"] == 140 and r23["n_reporting"] == 2 and r23["n_universe"] == 4
    assert r23["missing_ciks"] == [2, 4]


def test_aggregate_panel_only_complete_companies():
    assert balanced_panel(items_long(), ["total_assets", "total_liabilities"], [2022, 2023]) == {1}
    a = aggregate(items_long(), ["total_assets"], [2022, 2023], {1, 2, 3}, mode="panel")
    assert list(a["sum"]) == [100, 110]


def test_yoy_like_for_like():
    y = yoy_like_for_like(items_long(), "total_assets", 2023, {1, 2, 3})
    assert y["n"] == 1 and y["pct"] == pytest.approx(0.1)


def test_item_changes():
    c = item_changes(items_long())
    r = c[(c["cik"] == 1) & (c["fiscal_year"] == 2023) & (c["item_key"] == "total_assets")].iloc[0]
    assert r["abs_change"] == 10 and r["pct_change"] == pytest.approx(0.1)
    assert math.isnan(c[(c["cik"] == 3) & (c["item_key"] == "total_assets")]["pct_change"].iloc[0])


def test_membership_reconstruction():
    raw = pd.DataFrame({("Effective Date", "Effective Date"): ["March 1, 2024", "June 1, 2023"],
                        ("Added", "Ticker"): ["NEW", "MID"], ("Added", "Security"): ["n", "m"],
                        ("Removed", "Ticker"): ["OLD", "GONE"], ("Removed", "Security"): ["o", "g"],
                        ("Reason", "Reason"): ["x", "y"]})
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    ch = parse_changes_table(raw)
    cur = {"NEW", "MID", "AAA"}
    assert members_at(cur, ch, pd.Timestamp("2024-06-30").date()) == cur
    assert members_at(cur, ch, pd.Timestamp("2023-12-31").date()) == {"OLD", "MID", "AAA"}
    assert members_at(cur, ch, pd.Timestamp("2022-12-31").date()) == {"OLD", "GONE", "AAA"}
