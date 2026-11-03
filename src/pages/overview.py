from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.aggregates import aggregate, balanced_panel, yoy_like_for_like
from src.components.data import item_label, pipeline_status
from src.components.ui import plot, sidebar_context
from src.visualization.charts import balance_structure, bar_chart, fmt_usd, line_chart

CORE = ["total_assets", "total_liabilities", "equity_total"]


def _missing_names(ctx, ciks) -> str:
    c = ctx.data["companies"].set_index("cik")
    names = [f"{c.at[k, 'primary_ticker']}" for k in ciks if k in c.index]
    return ", ".join(names) if names else "keine"


def page() -> None:
    ctx = sidebar_context()
    items, comps = ctx.items, ctx.companies
    st.title("S&P 500 Balance Sheet Intelligence")
    st.caption("Bilanzdaten aus SEC EDGAR XBRL (10-K). Summen sind Summen der verfügbaren Werte; "
               "die Anzahl meldender Unternehmen steht jeweils dabei.")

    year = max(ctx.years)
    n_univ = len(ctx.universe)
    n_loaded = int((comps["fetch_status"] == "ok").sum())
    ta = items[items["item_key"] == "total_assets"]
    cells = n_univ * len(ctx.years)
    coverage = ta[["cik", "fiscal_year"]].drop_duplicates().shape[0] / cells if cells else 0

    c = st.columns(3)
    c[0].metric("Erfasste Unternehmen", f"{n_loaded} / {n_univ}",
                help="Unternehmen mit erfolgreich geladenen SEC-Daten / Unternehmen im gewählten Universum")
    c[1].metric("Datenabdeckung (Bilanzsumme)", f"{coverage:.1%}",
                help=f"Anteil der Unternehmen×Jahre ({cells}) mit gemeldeter Bilanzsumme")
    stt = pipeline_status()
    meta = ctx.data["meta"]
    c[2].metric("Letzter Datenabruf", (meta.get("built_at") or "–")[:16].replace("T", " "),
                help=f"Status: {stt.get('state') if stt else 'unbekannt'} · {meta.get('last_run_counts', '')}")

    c = st.columns(3)
    for col, key in zip(c, CORE):
        agg = aggregate(items, [key], [year], ctx.universe)
        r = agg.iloc[0]
        yoy = yoy_like_for_like(ctx.data["items"], key, year, ctx.universe)
        delta = f"{yoy['pct']:+.1%} ggü. Vorjahr (n={yoy['n']})" if pd.notna(yoy["pct"]) else None
        derived = items[(items["item_key"] == key) & (items["fiscal_year"] == year) & items["is_derived"]].shape[0]
        col.metric(f"{item_label(key)} GJ {year}", fmt_usd(r["sum"]), delta,
                   help=f"Summe der verfügbaren Werte: {r['n_reporting']} von {r['n_universe']} Unternehmen. "
                        f"Davon {derived} berechnet. Veränderung like-for-like (nur Unternehmen mit Wert in beiden Jahren).")
    st.markdown(f"<div class='note'>Die drei Summen beruhen auf unterschiedlichen Unternehmensmengen, wenn Werte "
                f"fehlen – für einen direkten Vergleich unten den Modus „Konsistentes Panel“ wählen.</div>",
                unsafe_allow_html=True)
    with st.expander(f"Welche Unternehmen fehlen in den Summen für GJ {year}?"):
        for key in CORE:
            agg = aggregate(items, [key], [year], ctx.universe).iloc[0]
            st.markdown(f"**{item_label(key)}** – fehlend ({len(agg['missing_ciks'])}): "
                        f"{_missing_names(ctx, agg['missing_ciks'])}")

    st.divider()
    mode = st.radio("Aggregationsmodus", ["Konsistentes Panel", "Alle verfügbaren Werte"], horizontal=True,
                    help="Panel: nur Unternehmen, die Bilanzsumme, Verbindlichkeiten und Eigenkapital in allen "
                         "gewählten Jahren melden – damit sind Jahre und Positionen direkt vergleichbar.")
    m = "panel" if mode.startswith("Konsistent") else "available"
    agg = aggregate(items, CORE, ctx.years, ctx.universe, m)
    agg["Position"] = agg["item_key"].map(item_label)
    agg["Jahr"] = agg["fiscal_year"].astype(str)
    n_panel = len(balanced_panel(items, CORE, ctx.years))
    left, right = st.columns(2)
    with left:
        plot(line_chart(agg, "Jahr", "sum", "Position", ctx.mode, "Historische Entwicklung: Aktiva, Verbindlichkeiten, Eigenkapital",
                        stable_order=[item_label(k) for k in CORE]), "ov_hist")
        st.caption(f"Basis: {n_panel} Unternehmen im Panel" if m == "panel" else
                   "Basis je Jahr: " + ", ".join(f"{int(r.fiscal_year)}: n={r.n_reporting}"
                                                for r in agg[agg.item_key == 'total_assets'].itertuples()))
    with right:
        panel = balanced_panel(items, CORE, ctx.years)
        w = (items[items["cik"].isin(panel) & items["item_key"].isin(CORE + ["temporary_equity"])]
             .groupby(["fiscal_year", "item_key"])["value"].sum().unstack())
        w = w.reset_index()
        plot(balance_structure(w, ctx.mode, f"Gestapelte Bilanzstruktur (Panel, n={len(panel)})"), "ov_struct")
        st.caption("Aktiva vs. Passiva. Mezzanine-Kapital ist nur enthalten, soweit gemeldet; "
                   "Differenzen zwischen den Säulen entstehen dadurch bzw. durch Rundung.")

    st.divider()
    left, right = st.columns(2)
    met = ctx.metrics
    with left:
        debt = met[(met["metric_key"] == "total_debt") & (met["status"] == "ok")][["cik", "fiscal_year", "value"]]
        cash = items[items["item_key"] == "cash"][["cik", "fiscal_year", "value"]]
        both = debt.merge(cash, on=["cik", "fiscal_year"], suffixes=("_debt", "_cash"))
        pn = both.groupby("cik")["fiscal_year"].nunique()
        keep = set(pn[pn == len(ctx.years)].index)
        b = both[both["cik"].isin(keep)].groupby("fiscal_year")[["value_debt", "value_cash"]].sum().reset_index()
        long = pd.concat([b.assign(Serie="Finanzschulden (vollständig)", v=b["value_debt"]),
                          b.assign(Serie="Zahlungsmittel", v=b["value_cash"])])
        long["Jahr"] = long["fiscal_year"].astype(str)
        plot(line_chart(long, "Jahr", "v", "Serie", ctx.mode, f"Schulden- und Cash-Entwicklung (Panel, n={len(keep)})"),
             "ov_debt")
        st.caption("Nur Unternehmen mit vollständiger Schuldendefinition und gemeldeten Zahlungsmitteln in allen Jahren.")
    with right:
        gi = items[items["item_key"].isin(["goodwill", "intangibles"])]
        pn = gi.groupby("cik").apply(lambda g: len(set(zip(g["item_key"], g["fiscal_year"]))), include_groups=False)
        keep = set(pn[pn == 2 * len(ctx.years)].index)
        g = gi[gi["cik"].isin(keep)].groupby(["fiscal_year", "item_key"])["value"].sum().reset_index()
        g["Position"] = g["item_key"].map(item_label)
        g["Jahr"] = g["fiscal_year"].astype(str)
        plot(bar_chart(g, "Jahr", "value", "Position", ctx.mode,
                       f"Goodwill und immaterielle Vermögenswerte (Panel, n={len(keep)})"), "ov_gw")

    st.divider()
    left, right = st.columns(2)
    with left:
        key = st.selectbox("Branchenvergleich – Position", CORE + ["cash", "goodwill", "current_assets"],
                           format_func=item_label, key="ov_sec_item")
        s = items[(items["item_key"] == key) & (items["fiscal_year"] == year)].groupby("sector")["value"] \
            .agg(["sum", "count"]).reset_index().sort_values("sum", ascending=False)
        s["Branche"] = s["sector"] + " (n=" + s["count"].astype(str) + ")"
        plot(bar_chart(s, "Branche", "sum", None, ctx.mode, f"{item_label(key)} nach Branche, GJ {year}",
                       horizontal=True, height=460), "ov_sector")
    with right:
        key = st.selectbox("Ranking nach", CORE, format_func=item_label, key="ov_rank_item")
        r = items[(items["item_key"] == key) & (items["fiscal_year"] == year)].nlargest(15, "value")
        plot(bar_chart(r, "primary_ticker", "value", None, ctx.mode, f"Top 15: {item_label(key)}, GJ {year}",
                       horizontal=True, height=460), "ov_rank")

    st.divider()
    y0, y1 = min(ctx.years), max(ctx.years)
    key = st.selectbox("Prozentuale Veränderung über den Zeitraum", CORE, format_func=item_label, key="ov_chg")
    a = items[(items["item_key"] == key) & (items["fiscal_year"].isin([y0, y1]))] \
        .pivot_table(index=["cik", "primary_ticker"], columns="fiscal_year", values="value") \
        .reindex(columns=[y0, y1]).dropna()
    if y0 != y1 and len(a):
        a = a[a[y0] > 0]
        a["chg"] = a[y1] / a[y0] - 1
        a = a.reset_index()
        up = a.nlargest(10, "chg")
        down = a[~a["cik"].isin(up["cik"])].nsmallest(10, "chg")
        top = pd.concat([up.assign(G="Stärkste Zunahme"), down.assign(G="Schwächste Entwicklung")])
        plot(bar_chart(top, "primary_ticker", "chg", "G", ctx.mode,
                       f"{item_label(key)}: Veränderung GJ {y0} → GJ {y1} (n={len(a)} mit Werten in beiden Jahren)",
                       kind="pct", horizontal=True, height=560), "ov_change")
        st.caption(f"Median aller {len(a)} Unternehmen: {a['chg'].median():+.1%}. Unternehmen mit Basis ≤ 0 ausgeschlossen.")
    else:
        st.info("Für die Veränderung bitte mindestens zwei Geschäftsjahre wählen.")
