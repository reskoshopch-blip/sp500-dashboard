from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.metrics import METRIC_BY_KEY, METRICS, item_changes
from src.components.data import item_label, item_options
from src.components.ui import export_buttons, sidebar_context
from src.data_sources.sec_client import filing_index_url
from src.normalization.taxonomy import SECTION_ORDER


def page() -> None:
    ctx = sidebar_context()
    st.title("Datentabelle")
    view = st.radio("Inhalt", ["Bilanzpositionen", "Kennzahlen"], horizontal=True, key="tb_view")
    comps = ctx.companies
    c1, c2 = st.columns([2, 3])
    search = c1.text_input("Suche Unternehmen / Ticker / CIK", key="tb_search")
    cs = comps
    if search:
        s = search.strip().lower()
        cs = comps[comps["name"].str.lower().str.contains(s, regex=False)
                   | comps["ticker"].str.lower().str.contains(s, regex=False)
                   | comps["cik"].astype(str).str.contains(s, regex=False)]

    if view == "Bilanzpositionen":
        sections = c2.multiselect("Abschnitt", SECTION_ORDER, default=[], placeholder="Alle Abschnitte", key="tb_sec")
        keys = st.multiselect("Bilanzpositionen", item_options(sections or None), format_func=item_label,
                              default=["total_assets", "total_liabilities", "equity_total"], key="tb_items")
        f1, f2, f3 = st.columns(3)
        avail = f1.multiselect("Verfügbarkeit", ["gemeldet", "berechnet", "fehlt"],
                               default=["gemeldet", "berechnet", "fehlt"], key="tb_avail")
        tag_q = f2.text_input("Datenquelle: XBRL-Tag enthält", key="tb_tag")
        restated_only = f3.toggle("Nur Restatements", key="tb_rest")
        if not keys:
            st.info("Bitte mindestens eine Position wählen.")
            return
        grid = pd.MultiIndex.from_product([sorted(cs["cik"]), ctx.years, keys],
                                          names=["cik", "fiscal_year", "item_key"]).to_frame(index=False)
        it = ctx.items[ctx.items["item_key"].isin(keys)]
        chg = item_changes(ctx.data["items"][ctx.data["items"]["item_key"].isin(keys)])[
            ["cik", "fiscal_year", "item_key", "abs_change", "pct_change"]]
        df = grid.merge(it, on=["cik", "fiscal_year", "item_key"], how="left") \
                 .merge(chg, on=["cik", "fiscal_year", "item_key"], how="left")
        if ctx.membership_only:
            mem = ctx.data["membership"]
            df = df.merge(mem[mem["was_member"] == True][["cik", "fiscal_year"]],  # noqa: E712
                          on=["cik", "fiscal_year"])
        df = df.drop(columns=["primary_ticker", "name", "sector"], errors="ignore") \
               .merge(comps[["cik", "primary_ticker", "name", "sector"]], on="cik")
        df["Verfügbarkeit"] = df["value"].isna().map({True: "fehlt", False: "gemeldet"})
        df.loc[df["is_derived"] == True, "Verfügbarkeit"] = "berechnet"  # noqa: E712
        df = df[df["Verfügbarkeit"].isin(avail)]
        if tag_q:
            df = df[df["tag"].fillna("").str.contains(tag_q, case=False)]
        if restated_only:
            df = df[df["restated"] == True]  # noqa: E712
        df["Position"] = df["item_key"].map(item_label)
        df["Filing"] = [filing_index_url(c, a) if isinstance(a, str) else None for c, a in zip(df["cik"], df["accn"])]
        out = df[["primary_ticker", "name", "cik", "sector", "fiscal_year", "period_end", "Position", "value",
                  "unit", "Verfügbarkeit", "abs_change", "pct_change", "tag", "derivation", "form", "filed", "accn",
                  "original_value", "n_versions", "restated", "Filing"]].rename(columns={
            "primary_ticker": "Ticker", "name": "Unternehmen", "cik": "CIK", "sector": "Branche",
            "fiscal_year": "GJ", "period_end": "Stichtag", "value": "Wert (USD)", "unit": "Einheit",
            "abs_change": "Δ Vorjahr (USD)", "pct_change": "Δ Vorjahr (%)", "tag": "XBRL-Tag",
            "derivation": "Berechnung", "form": "Formular", "filed": "Eingereicht", "accn": "Accession",
            "original_value": "Erstmeldung", "n_versions": "Meldungen", "restated": "Restatement"})
        out["Δ Vorjahr (%)"] = out["Δ Vorjahr (%)"] * 100
        st.caption(f"{len(out):,} Zeilen".replace(",", "'") + " · leere Werte = nicht verfügbar (nie als 0 gezählt)")
        st.dataframe(out.sort_values(["Ticker", "Position", "GJ"]), hide_index=True, width="stretch", height=560,
                     column_config={
                         "Wert (USD)": st.column_config.NumberColumn(format="localized"),
                         "Δ Vorjahr (USD)": st.column_config.NumberColumn(format="localized"),
                         "Δ Vorjahr (%)": st.column_config.NumberColumn(format="%.1f %%"),
                         "Erstmeldung": st.column_config.NumberColumn(format="localized"),
                         "Filing": st.column_config.LinkColumn(display_text="öffnen"),
                         "GJ": st.column_config.NumberColumn(format="%d"),
                         "CIK": st.column_config.NumberColumn(format="%d")})
        export_buttons(out, "bilanzpositionen", "tb_exp")
    else:
        keys = st.multiselect("Kennzahlen", [m.key for m in METRICS], format_func=lambda k: METRIC_BY_KEY[k].label,
                              default=["equity_ratio", "debt_to_equity", "current_ratio"], key="tb_metrics")
        status = st.multiselect("Status", ["ok", "teilweise", "NA"], default=["ok", "teilweise", "NA"], key="tb_status")
        m = ctx.metrics[ctx.metrics["metric_key"].isin(keys) & ctx.metrics["status"].isin(status)
                        & ctx.metrics["cik"].isin(cs["cik"])]
        m = m.merge(comps[["cik", "primary_ticker", "name", "sector"]], on="cik")
        m["Kennzahl"] = m["metric_key"].map(lambda k: METRIC_BY_KEY[k].label)
        m["Formel"] = m["metric_key"].map(lambda k: METRIC_BY_KEY[k].formula)
        out = m[["primary_ticker", "name", "sector", "fiscal_year", "Kennzahl", "value", "status", "note", "Formel"]] \
            .rename(columns={"primary_ticker": "Ticker", "name": "Unternehmen", "sector": "Branche",
                             "fiscal_year": "GJ", "value": "Wert (berechnet)", "status": "Status", "note": "Hinweis"})
        st.dataframe(out.sort_values(["Ticker", "Kennzahl", "GJ"]), hide_index=True, width="stretch", height=560,
                     column_config={"GJ": st.column_config.NumberColumn(format="%d"),
                                    "Wert (berechnet)": st.column_config.NumberColumn(format="%.4f")})
        st.caption("Quoten als Dezimalzahl (0.25 = 25 %), USD-Kennzahlen in USD.")
        export_buttons(out, "kennzahlen", "tb_mexp")
