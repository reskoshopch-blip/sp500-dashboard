from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.metrics import METRIC_BY_KEY, METRICS
from src.components.data import item_label, item_options
from src.components.ui import company_picker, export_buttons, plot, sidebar_context
from src.visualization.charts import bar_chart, fmt_metric, heatmap, line_chart

RATIO_KEYS = ["debt_to_equity", "liabilities_to_assets", "equity_ratio", "current_ratio", "quick_ratio",
              "cash_to_debt", "goodwill_to_assets", "cash_to_assets", "assets_growth", "equity_growth"]


def page() -> None:
    ctx = sidebar_context()
    st.title("Unternehmensvergleich")
    sel = company_picker(ctx, "Unternehmen (max. 8 für unterscheidbare Farben)", "cmp_sel",
                         default=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"])
    if sel.empty:
        st.info("Bitte mindestens ein Unternehmen wählen.")
        return
    if len(sel) > 8:
        st.warning("Mehr als 8 Unternehmen: zusätzliche Serien werden grau dargestellt. Für Lesbarkeit bitte reduzieren.")
    order = sel["primary_ticker"].tolist()
    ciks = set(sel["cik"])

    kind = st.radio("Größe", ["Bilanzposition", "Kennzahl"], horizontal=True, key="cmp_kind")
    if kind == "Bilanzposition":
        key = st.selectbox("Position", item_options(), format_func=item_label, key="cmp_item")
        d = ctx.items[ctx.items["cik"].isin(ciks) & (ctx.items["item_key"] == key)].copy()
        d["v"], vkind, title = d["value"], "usd", item_label(key)
    else:
        key = st.selectbox("Kennzahl", [m.key for m in METRICS], format_func=lambda k: METRIC_BY_KEY[k].label,
                           key="cmp_metric")
        d = ctx.metrics[ctx.metrics["cik"].isin(ciks) & (ctx.metrics["metric_key"] == key)].copy()
        d = d.merge(sel[["cik", "primary_ticker"]], on="cik")
        d["v"], vkind, title = d["value"], METRIC_BY_KEY[key].kind, METRIC_BY_KEY[key].label + " (berechnet)"
        st.caption(f"Formel: {METRIC_BY_KEY[key].formula}")
    d["Jahr"] = d["fiscal_year"].astype(str)
    d = d.sort_values(["fiscal_year"])
    left, right = st.columns(2)
    with left:
        plot(line_chart(d.dropna(subset=["v"]), "Jahr", "v", "primary_ticker", ctx.mode, f"{title} – Verlauf",
                        kind=vkind, stable_order=order), "cmp_line")
    with right:
        plot(bar_chart(d.dropna(subset=["v"]), "Jahr", "v", "primary_ticker", ctx.mode, f"{title} – je Jahr",
                       kind=vkind, stable_order=order), "cmp_bar")

    grid = (d.pivot_table(index="primary_ticker", columns="fiscal_year", values="v", aggfunc="first")
            .reindex(index=order, columns=ctx.years))
    show = grid.map(lambda v: fmt_metric(v, vkind)).replace("NA", "Nicht verfügbar")
    if len(ctx.years) >= 2:
        y0, y1 = min(ctx.years), max(ctx.years)
        chg = grid[y1] / grid[y0] - 1
        show[f"Δ {y0}→{y1}"] = [f"{v:+.1%}" if pd.notna(v) and grid.at[i, y0] > 0 else "NA"
                                for i, v in chg.items()]
    show.columns = [str(c) for c in show.columns]
    st.dataframe(show, width="stretch")
    if kind == "Kennzahl":
        st_tab = d.pivot_table(index="primary_ticker", columns="fiscal_year", values="note", aggfunc="first")
        with st.expander("Hinweise zu nicht oder teilweise berechenbaren Werten"):
            st_tab.columns = [str(c) for c in st_tab.columns]
            st.dataframe(st_tab.reindex(index=order), width="stretch")

    st.divider()
    year = st.select_slider("Jahr für die Kennzahlen-Heatmap", options=ctx.years, value=max(ctx.years), key="cmp_hy")
    m = ctx.metrics[ctx.metrics["cik"].isin(ciks) & (ctx.metrics["fiscal_year"] == year)
                    & ctx.metrics["metric_key"].isin(RATIO_KEYS)].merge(sel[["cik", "primary_ticker"]], on="cik")
    z = m.pivot_table(index="primary_ticker", columns="metric_key", values="value", aggfunc="first") \
        .reindex(index=order, columns=RATIO_KEYS)
    text = pd.DataFrame({c: [fmt_metric(v, METRIC_BY_KEY[c].kind) for v in z[c]] for c in z.columns}, index=z.index)
    z.columns = [METRIC_BY_KEY[c].label for c in z.columns]
    text.columns = z.columns
    plot(heatmap(z, text, ctx.mode, f"Bilanzkennzahlen GJ {year} – Farbe = Rang innerhalb der Spalte"), "cmp_heat")
    st.caption("Die Farbintensität zeigt die relative Position innerhalb der Auswahl (je Kennzahl), nicht 'gut' oder "
               "'schlecht'. Grau = nicht berechenbar.")
    export_buttons(d[["primary_ticker", "fiscal_year", "v"]].rename(columns={"v": title}), "vergleich", "cmp_exp")
