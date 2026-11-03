from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.metrics import METRIC_BY_KEY, METRICS
from src.components.data import item_label, item_options
from src.components.ui import export_buttons, plot, sidebar_context
from src.visualization.charts import bar_chart, fmt_metric, fmt_usd, heatmap, line_chart

RATIOS = ["equity_ratio", "liabilities_to_assets", "debt_to_equity", "current_ratio", "cash_to_assets",
          "goodwill_to_assets", "assets_growth"]


def page() -> None:
    ctx = sidebar_context()
    st.title("Branchen")
    year = st.select_slider("Geschäftsjahr", options=ctx.years, value=max(ctx.years), key="sec_year")
    items, met = ctx.items, ctx.metrics
    comps = ctx.companies

    rows = []
    for sector, g in comps.groupby("sector"):
        ciks = set(g["cik"])
        it = items[items["cik"].isin(ciks) & (items["fiscal_year"] == year)]
        r = {"Branche": sector, "Unternehmen": len(ciks)}
        for k in ("total_assets", "total_liabilities", "equity_total"):
            v = it[it["item_key"] == k]["value"]
            r[item_label(k)] = fmt_usd(v.sum()) if len(v) else "Nicht verfügbar"
            r[f"n {k}"] = len(v)
        rows.append(r)
    tab = pd.DataFrame(rows)
    st.dataframe(tab, width="stretch", hide_index=True,
                 column_config={f"n {k}": st.column_config.NumberColumn(f"n meldend ({item_label(k)})")
                                for k in ("total_assets", "total_liabilities", "equity_total")})
    st.caption("Summen der verfügbaren Werte; 'n meldend' zeigt, wie viele Unternehmen der Branche einen Wert haben.")

    m = met[(met["fiscal_year"] == year) & met["metric_key"].isin(RATIOS)].merge(comps[["cik", "sector"]], on="cik")
    med = m.groupby(["sector", "metric_key"])["value"].median().unstack().reindex(columns=RATIOS)
    text = pd.DataFrame({c: [fmt_metric(v, METRIC_BY_KEY[c].kind) for v in med[c]] for c in med.columns},
                        index=med.index)
    med.columns = text.columns = [METRIC_BY_KEY[c].label for c in RATIOS]
    plot(heatmap(med, text, ctx.mode, f"Median-Kennzahlen je Branche, GJ {year} (berechnet)"), "sec_heat")
    st.caption("Mediane über alle Unternehmen der Branche mit berechenbarer Kennzahl. Banken/Versicherungen haben "
               "meist keine Current Ratio (keine Gliederung in kurz-/langfristig).")

    st.divider()
    sector = st.selectbox("Unternehmen innerhalb einer Branche vergleichen", sorted(comps["sector"].unique()),
                          key="sec_pick")
    kind = st.radio("Größe", ["Bilanzposition", "Kennzahl"], horizontal=True, key="sec_kind")
    ciks = set(comps[comps["sector"] == sector]["cik"])
    if kind == "Bilanzposition":
        key = st.selectbox("Position", item_options(), format_func=item_label, key="sec_item")
        d = items[items["cik"].isin(ciks) & (items["item_key"] == key)]
        d = d[["primary_ticker", "fiscal_year", "value"]]
        vkind, title = "usd", item_label(key)
    else:
        key = st.selectbox("Kennzahl", [x.key for x in METRICS], format_func=lambda k: METRIC_BY_KEY[k].label,
                           key="sec_metric")
        d = met[met["cik"].isin(ciks) & (met["metric_key"] == key)].merge(
            comps[["cik", "primary_ticker"]], on="cik")[["primary_ticker", "fiscal_year", "value"]]
        vkind, title = METRIC_BY_KEY[key].kind, METRIC_BY_KEY[key].label
    cur = d[d["fiscal_year"] == year].dropna(subset=["value"]).sort_values("value", ascending=False)
    n_missing = len(ciks) - len(cur)
    plot(bar_chart(cur.head(30), "primary_ticker", "value", None, ctx.mode,
                   f"{title} – {sector}, GJ {year} (n={len(cur)}, {n_missing} ohne Wert)", kind=vkind,
                   horizontal=True, height=max(360, 22 * min(len(cur), 30) + 120)), "sec_rank")
    top = cur.head(6)["primary_ticker"].tolist()
    hist = d[d["primary_ticker"].isin(top)].dropna(subset=["value"]).copy()
    hist["Jahr"] = hist["fiscal_year"].astype(str)
    plot(line_chart(hist, "Jahr", "value", "primary_ticker", ctx.mode, f"{title} – Verlauf der Top 6", kind=vkind,
                    stable_order=top), "sec_hist")
    export_buttons(d.rename(columns={"value": title}), f"branche_{sector}", "sec_exp")
