"""Aggregationen über Unternehmen – immer mit Angabe der Abdeckung.

  * 'available': Summe der verfügbaren Werte je Jahr; n_reporting / n_universe werden mitgeliefert,
    fehlende Unternehmen werden aufgelistet.
  * 'panel': nur Unternehmen, die ALLE gewählten Positionen in ALLEN gewählten Jahren melden.
    Damit sind Jahre und Positionen (Vermögen vs. Verbindlichkeiten vs. Eigenkapital) direkt vergleichbar.
  * Vorjahresveränderungen werden immer like-for-like berechnet (nur Unternehmen mit Wert in beiden Jahren).
Gesamtpositionen werden nie mit ihren Unterpositionen addiert, weil nur einzelne Positionen summiert werden.
"""
from __future__ import annotations

import pandas as pd


def balanced_panel(items: pd.DataFrame, item_keys: list[str], years: list[int]) -> set[int]:
    sub = items[items["item_key"].isin(item_keys) & items["fiscal_year"].isin(years) & items["value"].notna()]
    counts = sub.groupby("cik")[["item_key", "fiscal_year"]].apply(
        lambda g: len(set(zip(g["item_key"], g["fiscal_year"]))))
    need = len(item_keys) * len(years)
    return set(counts[counts == need].index.astype(int))


def aggregate(items: pd.DataFrame, item_keys: list[str], years: list[int], universe: set[int],
              mode: str = "available") -> pd.DataFrame:
    sub = items[items["item_key"].isin(item_keys) & items["fiscal_year"].isin(years) & items["value"].notna()
                & items["cik"].isin(universe)]
    if mode == "panel":
        panel = balanced_panel(sub, item_keys, years)
        sub = sub[sub["cik"].isin(panel)]
        denom = len(panel)
    else:
        denom = len(universe)
    rows = []
    for key in item_keys:
        for y in years:
            g = sub[(sub["item_key"] == key) & (sub["fiscal_year"] == y)]
            reporting = set(g["cik"].astype(int))
            rows.append({
                "item_key": key, "fiscal_year": y, "sum": g["value"].sum() if len(g) else float("nan"),
                "median": g["value"].median() if len(g) else float("nan"),
                "n_reporting": len(reporting), "n_universe": denom,
                "missing_ciks": sorted((universe if mode != "panel" else set(sub["cik"].astype(int))) - reporting),
            })
    return pd.DataFrame(rows)


def yoy_like_for_like(items: pd.DataFrame, item_key: str, year: int, universe: set[int]) -> dict:
    sub = items[(items["item_key"] == item_key) & items["cik"].isin(universe) & items["value"].notna()]
    cur = sub[sub["fiscal_year"] == year].set_index("cik")["value"]
    prev = sub[sub["fiscal_year"] == year - 1].set_index("cik")["value"]
    both = cur.index.intersection(prev.index)
    if len(both) == 0:
        return {"pct": float("nan"), "n": 0, "cur": float("nan"), "prev": float("nan")}
    c, p = cur.loc[both].sum(), prev.loc[both].sum()
    return {"pct": (c / p - 1) if p > 0 else float("nan"), "n": len(both), "cur": c, "prev": p}


def sector_aggregate(items: pd.DataFrame, companies: pd.DataFrame, item_key: str, year: int) -> pd.DataFrame:
    sub = items[(items["item_key"] == item_key) & (items["fiscal_year"] == year) & items["value"].notna()]
    m = companies[["cik", "sector"]].merge(sub[["cik", "value"]], on="cik", how="left")
    g = m.groupby("sector", dropna=False)
    return pd.DataFrame({
        "sum": g["value"].sum(min_count=1), "median": g["value"].median(),
        "n_reporting": g["value"].count(), "n_companies": g["cik"].nunique(),
    }).reset_index()
