"""Dokumentierte Ableitungen fehlender Summenpositionen. Abgeleitete Werte sind immer als
'berechnet' gekennzeichnet (is_derived=True) und tragen ihre Rechenformel in 'derivation'.

  equity_total      = equity_parent + nci            (wenn nci gemeldet)
                    = equity_parent                  (wenn keine Minderheiten gemeldet; Annahme dokumentiert)
  total_liabilities = liabilities_and_equity − equity_total − temporary_equity (falls gemeldet)
                      (liabilities_and_equity fehlt -> total_assets, Bilanzidentität)
"""
from __future__ import annotations

import pandas as pd


def _row(template: dict, key: str, value: float, derivation: str, sources: list[dict]) -> dict:
    r = dict(template)
    r.update({
        "item_key": key, "value": float(value), "unit": "USD", "tag": "berechnet",
        "accn": sources[0].get("accn"), "form": sources[0].get("form"),
        "filed": max(s.get("filed") for s in sources if s.get("filed") is not None) if sources else None,
        "original_value": None, "original_filed": None, "n_versions": None,
        "restated": any(bool(s.get("restated")) for s in sources), "conflict": False,
        "is_derived": True, "derivation": derivation, "tag_rank": None,
    })
    return r


def add_derived_items(items: pd.DataFrame, period_cols: list[str]) -> pd.DataFrame:
    if items is None or items.empty:
        return items
    new_rows = []
    for pkey, g in items.groupby(period_cols, sort=False):
        pkey = pkey if isinstance(pkey, tuple) else (pkey,)
        base = dict(zip(period_cols, pkey))
        recs = {r["item_key"]: r for r in g.to_dict("records")}

        eq_total = recs.get("equity_total")
        if eq_total is None and "equity_parent" in recs:
            ep = recs["equity_parent"]
            if "nci" in recs:
                eq_total = _row(base, "equity_total", ep["value"] + recs["nci"]["value"],
                                "equity_parent + nci", [ep, recs["nci"]])
            else:
                eq_total = _row(base, "equity_total", ep["value"],
                                "= equity_parent (keine nicht beherrschenden Anteile gemeldet)", [ep])
            new_rows.append(eq_total)

        if "total_liabilities" not in recs and eq_total is not None:
            le = recs.get("liabilities_and_equity")
            le_name = "liabilities_and_equity"
            if le is None and "total_assets" in recs:
                le, le_name = recs["total_assets"], "total_assets"
            if le is not None:
                val = le["value"] - eq_total["value"]
                formula = f"{le_name} − equity_total"
                srcs = [le, eq_total]
                if "temporary_equity" in recs:
                    val -= recs["temporary_equity"]["value"]
                    formula += " − temporary_equity"
                    srcs.append(recs["temporary_equity"])
                new_rows.append(_row(base, "total_liabilities", val, formula, srcs))
    if not new_rows:
        return items
    return pd.concat([items, pd.DataFrame(new_rows)], ignore_index=True)
