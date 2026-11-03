"""Bilanzkennzahlen. Jede Kennzahl erhält einen Status:
   ok         – alle benötigten Eingangswerte gemeldet
   teilweise  – berechnet, aber eine optionale Komponente fehlt (Hinweis nennt sie)
   NA         – nicht berechenbar (Hinweis nennt den Grund)
Fehlende Eingaben werden NIE als 0 angenommen. Division durch 0 bzw. durch negative Basen,
bei denen die Kennzahl ökonomisch nicht interpretierbar ist, ergibt NA.
Alle Kennzahlen sind abgeleitet und werden in der Oberfläche als "berechnet" gekennzeichnet.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

OK, PARTIAL, NA = "ok", "teilweise", "NA"


@dataclass(frozen=True)
class MetricDef:
    key: str
    label: str
    kind: str   # ratio | pct | usd
    formula: str


METRICS: tuple[MetricDef, ...] = (
    MetricDef("assets_growth", "Wachstum Bilanzsumme", "pct", "total_assets(t) / total_assets(t−1) − 1"),
    MetricDef("liabilities_growth", "Wachstum Verbindlichkeiten", "pct", "total_liabilities(t) / total_liabilities(t−1) − 1"),
    MetricDef("equity_growth", "Wachstum Eigenkapital", "pct", "equity_total(t) / equity_total(t−1) − 1; NA wenn Basis ≤ 0"),
    MetricDef("total_debt", "Finanzschulden gesamt", "usd",
              "ltd_total + st_debt, sonst ltd_noncurrent + (debt_current_total oder st_debt + ltd_current); ohne Leasing"),
    MetricDef("debt_to_equity", "Debt-to-Equity", "ratio", "total_debt / equity_total; NA wenn Eigenkapital ≤ 0"),
    MetricDef("liabilities_to_equity", "Verbindlichkeiten / Eigenkapital", "ratio", "total_liabilities / equity_total; NA wenn EK ≤ 0"),
    MetricDef("liabilities_to_assets", "Verbindlichkeiten / Bilanzsumme", "ratio", "total_liabilities / total_assets"),
    MetricDef("equity_ratio", "Eigenkapitalquote", "pct", "equity_total / total_assets"),
    MetricDef("current_ratio", "Current Ratio", "ratio", "current_assets / current_liabilities"),
    MetricDef("quick_ratio", "Quick Ratio", "ratio", "(cash + st_investments + receivables) / current_liabilities"),
    MetricDef("cash_to_debt", "Cash / Finanzschulden", "ratio", "cash / total_debt; NA wenn keine Finanzschulden"),
    MetricDef("net_debt", "Nettoverschuldung", "usd", "total_debt − cash; nur bei vollständiger Schuldendefinition"),
    MetricDef("working_capital", "Working Capital", "usd", "current_assets − current_liabilities"),
    MetricDef("goodwill_to_assets", "Goodwill / Bilanzsumme", "pct", "goodwill / total_assets"),
    MetricDef("cash_to_assets", "Cash / Bilanzsumme", "pct", "cash / total_assets"),
)
METRIC_BY_KEY = {m.key: m for m in METRICS}


def _v(row: dict, key: str) -> float | None:
    v = row.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return None
    return float(v)


def _div(num: float | None, den: float | None, den_name: str, positive_den: bool = True) -> tuple[float | None, str]:
    if num is None:
        return None, "Zähler nicht verfügbar"
    if den is None:
        return None, f"{den_name} nicht verfügbar"
    if den == 0:
        return None, f"{den_name} = 0 (Division durch null)"
    if positive_den and den < 0:
        return None, f"{den_name} negativ – Kennzahl nicht interpretierbar"
    return num / den, ""


def total_debt(row: dict) -> tuple[float | None, str, str]:
    ltd_total, ltd_nc = _v(row, "ltd_total"), _v(row, "ltd_noncurrent")
    st, ltd_cur, dct = _v(row, "st_debt"), _v(row, "ltd_current"), _v(row, "debt_current_total")
    if ltd_total is not None:
        if st is not None:
            return ltd_total + st, OK, "ltd_total + st_debt"
        if dct is not None and ltd_cur is not None:
            return ltd_total + max(dct - ltd_cur, 0.0), OK, "ltd_total + (debt_current_total − ltd_current)"
        return ltd_total, PARTIAL, "ltd_total; kurzfristige Finanzschulden nicht gemeldet"
    if ltd_nc is not None:
        if dct is not None:
            return ltd_nc + dct, OK, "ltd_noncurrent + debt_current_total"
        parts = [x for x in (st, ltd_cur) if x is not None]
        if len(parts) == 2:
            return ltd_nc + sum(parts), OK, "ltd_noncurrent + st_debt + ltd_current"
        missing = [n for n, x in (("st_debt", st), ("ltd_current", ltd_cur)) if x is None]
        return ltd_nc + sum(parts), PARTIAL, "ltd_noncurrent + verfügbare kurzfristige Teile; fehlt: " + ", ".join(missing)
    return None, NA, "Langfristige Finanzschulden nicht gemeldet"


def compute_metrics_row(row: dict, prev: dict | None) -> list[tuple[str, float | None, str, str]]:
    out: list[tuple[str, float | None, str, str]] = []

    def add(key, val, status=None, note=""):
        if val is None:
            out.append((key, None, NA, note or "nicht berechenbar"))
        else:
            out.append((key, float(val), status or OK, note))

    for key, item in (("assets_growth", "total_assets"), ("liabilities_growth", "total_liabilities"),
                      ("equity_growth", "equity_total")):
        if prev is None:
            add(key, None, note="Vorjahr nicht verfügbar")
            continue
        cur, base = _v(row, item), _v(prev, item)
        q, why = _div(cur, base, f"{item} Vorjahr")
        add(key, None if q is None else q - 1, note=why)

    debt, dstat, dnote = total_debt(row)
    add("total_debt", debt, dstat, dnote)

    eq = _v(row, "equity_total")
    q, why = _div(debt, eq, "Eigenkapital")
    add("debt_to_equity", q, dstat if q is not None else None, why or (dnote if dstat == PARTIAL else ""))
    q, why = _div(_v(row, "total_liabilities"), eq, "Eigenkapital")
    add("liabilities_to_equity", q, note=why)
    q, why = _div(_v(row, "total_liabilities"), _v(row, "total_assets"), "Bilanzsumme")
    add("liabilities_to_assets", q, note=why)
    q, why = _div(eq, _v(row, "total_assets"), "Bilanzsumme")
    add("equity_ratio", q, note=why)
    ca, cl = _v(row, "current_assets"), _v(row, "current_liabilities")
    q, why = _div(ca, cl, "Kurzfristige Verbindlichkeiten")
    add("current_ratio", q, note=why if ca is not None or cl is not None else "Keine Gliederung in kurz-/langfristig (z. B. Banken)")

    cash = _v(row, "cash")
    if cash is None:
        add("quick_ratio", None, note="Zahlungsmittel nicht verfügbar")
    else:
        sti, rec = _v(row, "st_investments"), _v(row, "receivables")
        num = cash + (sti or 0.0) + (rec or 0.0)
        missing = [n for n, x in (("st_investments", sti), ("receivables", rec)) if x is None]
        q, why = _div(num, cl, "Kurzfristige Verbindlichkeiten")
        add("quick_ratio", q, PARTIAL if missing else OK,
            why or ("ohne nicht gemeldete Komponenten: " + ", ".join(missing) if missing else ""))

    if debt is not None and debt == 0:
        add("cash_to_debt", None, note="Keine Finanzschulden")
    else:
        q, why = _div(cash, debt, "Finanzschulden")
        add("cash_to_debt", q, dstat if q is not None else None, why)

    if debt is not None and dstat == OK and cash is not None:
        add("net_debt", debt - cash)
    else:
        add("net_debt", None, note="Schuldendefinition unvollständig oder Cash fehlt – keine konsistente Definition")

    add("working_capital", None if ca is None or cl is None else ca - cl,
        note="" if ca is not None and cl is not None else "Umlaufvermögen oder kurzfr. Verbindlichkeiten fehlt")
    q, why = _div(_v(row, "goodwill"), _v(row, "total_assets"), "Bilanzsumme")
    add("goodwill_to_assets", q, note=why if _v(row, "goodwill") is not None else "Goodwill nicht gemeldet")
    q, why = _div(cash, _v(row, "total_assets"), "Bilanzsumme")
    add("cash_to_assets", q, note=why)
    return out


def compute_metrics(wide: pd.DataFrame) -> pd.DataFrame:
    """wide: Spalten cik, fiscal_year + item_keys. Ergebnis im Long-Format."""
    rows = []
    if wide.empty:
        return pd.DataFrame(columns=["cik", "fiscal_year", "metric_key", "value", "status", "note"])
    recs = wide.to_dict("records")
    by_key = {(int(r["cik"]), int(r["fiscal_year"])): r for r in recs}
    for r in recs:
        cik, fy = int(r["cik"]), int(r["fiscal_year"])
        prev = by_key.get((cik, fy - 1))
        for key, val, status, note in compute_metrics_row(r, prev):
            rows.append((cik, fy, key, val, status, note))
    return pd.DataFrame(rows, columns=["cik", "fiscal_year", "metric_key", "value", "status", "note"])


def item_changes(long_items: pd.DataFrame) -> pd.DataFrame:
    """Veränderung jeder Bilanzposition gegenüber dem Vorjahr (absolut und relativ)."""
    if long_items.empty:
        return long_items
    df = long_items[["cik", "fiscal_year", "item_key", "value"]].copy()
    prev = df.assign(fiscal_year=df["fiscal_year"] + 1).rename(columns={"value": "prev_value"})
    m = df.merge(prev, on=["cik", "fiscal_year", "item_key"], how="left")
    m["abs_change"] = m["value"] - m["prev_value"]
    ok = m["prev_value"].notna() & (m["prev_value"] > 0)
    m["pct_change"] = (m["abs_change"] / m["prev_value"]).where(ok)
    return m
