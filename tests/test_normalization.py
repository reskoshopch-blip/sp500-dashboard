import pandas as pd
import pytest

from src.normalization.facts import (assign_fiscal_years, detect_periods, fallback_fiscal_year,
                                     flatten_instant_facts, normalize_company, ANNUAL_FORMS)
from src.normalization.taxonomy import ITEMS, TAG_TO_ITEM, overlap_documentation
from tests.fixtures.builder import company_a, company_b

YEARS = (2020, 2021, 2022, 2023, 2024, 2025)


def val(items: pd.DataFrame, fy: int, key: str):
    s = items[(items["fiscal_year"] == fy) & (items["item_key"] == key)]
    assert len(s) <= 1, f"Doppelte Position {key} {fy}"
    return None if s.empty else s.iloc[0]


def test_duration_facts_and_non_currency_units_excluded():
    df = flatten_instant_facts(company_a())
    assert "Revenues" not in set(df["tag"])
    assert not (df["val"] == 123456).any()           # Assets mit start-Datum = Duration
    assert set(df["unit"]) == {"USD"}                 # 'shares' verworfen


def test_period_detection_uses_assets_end_date():
    df = flatten_instant_facts(company_a())
    p = detect_periods(df, ANNUAL_FORMS)
    ends = dict(zip(p["accn"], p["period_end"].dt.strftime("%Y-%m-%d")))
    assert ends["0000001111-24-000001"] == "2023-12-31"
    assert ends["0000001111-23-000001"] == "2022-12-31"


def test_fiscal_year_for_retailer_ending_in_february():
    res = normalize_company(company_b(), YEARS)
    fy = dict(zip(res.periods["period_end"].dt.strftime("%Y-%m-%d"), res.periods["fiscal_year"]))
    assert fy["2024-02-03"] == 2023 and fy["2023-01-28"] == 2022
    assert fallback_fiscal_year(pd.Timestamp("2024-02-03")) == 2023
    assert fallback_fiscal_year(pd.Timestamp("2024-06-30")) == 2024


def test_implausible_fy_falls_back_to_rule():
    p = pd.DataFrame({"accn": ["x"], "form": ["10-K"], "filed": [pd.Timestamp("2024-02-01")],
                      "period_end": [pd.Timestamp("2023-12-31")], "fy_reported": pd.array([2019], dtype="Int64"),
                      "fp": ["FY"]})
    out, issues = assign_fiscal_years(p)
    assert int(out["fiscal_year"].iloc[0]) == 2023 and issues


def test_comparative_columns_fill_prior_years_and_restatement_uses_latest():
    res = normalize_company(company_a(), YEARS)
    a22 = val(res.annual_items, 2022, "total_assets")
    assert a22["value"] == 900 and a22["original_value"] == 880
    assert bool(a22["restated"]) and a22["n_versions"] == 2
    a21 = val(res.annual_items, 2021, "total_assets")
    assert a21["value"] == 800 and not bool(a21["restated"])


def test_quarterly_values_never_used_as_annual():
    res = normalize_company(company_a(), YEARS)
    assert val(res.annual_items, 2023, "total_assets")["value"] == 1000   # nicht 999 aus dem 10-Q
    assert 2024 not in set(res.annual_items["fiscal_year"])
    latest = res.latest_items
    assert latest["period_end"].iloc[0] == pd.Timestamp("2024-03-31")
    assert latest[latest["item_key"] == "total_assets"]["value"].iloc[0] == 1050
    assert latest["source_form"].iloc[0] == "10-Q"


def test_zero_is_kept_and_missing_stays_missing():
    res = normalize_company(company_a(), YEARS)
    cash = val(res.annual_items, 2023, "cash")
    assert cash is not None and cash["value"] == 0
    assert val(res.annual_items, 2023, "goodwill") is None
    assert val(res.annual_items, 2025, "total_assets") is None


def test_derived_total_liabilities_and_equity_marked():
    res = normalize_company(company_a(), YEARS)
    l21 = val(res.annual_items, 2021, "total_liabilities")          # Tag fehlt 2021
    assert l21["value"] == 800 - 350 and bool(l21["is_derived"]) and l21["tag"] == "berechnet"
    l23 = val(res.annual_items, 2023, "total_liabilities")          # gemeldet
    assert l23["value"] == 550 and not bool(l23["is_derived"])
    b = normalize_company(company_b(), YEARS)
    eq = val(b.annual_items, 2023, "equity_total")
    assert eq["value"] == -15 and "nci" in eq["derivation"]
    lb = val(b.annual_items, 2023, "total_liabilities")
    assert lb["value"] == 515


def test_no_duplicate_items_per_period():
    res = normalize_company(company_a(), YEARS)
    d = res.annual_items.groupby(["fiscal_year", "item_key"]).size()
    assert (d == 1).all()


def test_taxonomy_tags_unique_and_documented():
    assert len(TAG_TO_ITEM) == sum(len(i.tags) for i in ITEMS)
    assert len({i.key for i in ITEMS}) == len(ITEMS)
    assert any(r["Schlüssel"] == "debt_current_total" for r in overlap_documentation())


def test_all_instant_tags_captured_dynamically():
    res = normalize_company(company_a(), YEARS)
    assert {"AccountsReceivableNetCurrent", "DebtCurrent"} <= set(res.annual_all["tag"])


def test_empty_companyfacts():
    res = normalize_company({"cik": 1, "entityName": "X", "facts": {}}, YEARS)
    assert res.annual_items.empty and res.issues


def test_schema_rejects_malformed_response():
    from pydantic import ValidationError
    bad = {"cik": 1, "entityName": "X", "facts": {"us-gaap": {"Assets": {"units": {"USD": [{"val": "abc"}]}}}}}
    with pytest.raises(ValidationError):
        normalize_company(bad, YEARS)
