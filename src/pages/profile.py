from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics.metrics import METRIC_BY_KEY, METRICS
from src.components.data import db_version, item_label, load_company_facts
from src.components.ui import company_picker, export_buttons, plot, sidebar_context, to_excel_bytes
from src.data_sources.sec_client import cik10, filing_index_url
from src.normalization.taxonomy import ITEMS, SECTION_ORDER, TAG_TO_ITEM
from src.visualization.charts import balance_structure, fmt_metric, fmt_usd

MARK_DERIVED, MARK_RESTATED = "ᴮ", "ᴿ"


def _cell(r) -> str:
    if r is None:
        return "Nicht verfügbar"
    s = fmt_usd(r["value"])
    if bool(r.get("is_derived")):
        s += f" {MARK_DERIVED}"
    if bool(r.get("restated")):
        s += f" {MARK_RESTATED}"
    return s


def page() -> None:
    ctx = sidebar_context()
    st.title("Unternehmensprofil")
    sel = company_picker(ctx, "Unternehmen", "prof_sel", multi=False, default=["AAPL"])
    if sel.empty:
        st.info("Keine Unternehmen mit geladenen Daten im gewählten Universum.")
        return
    c = sel.iloc[0]
    cik = int(c["cik"])
    st.subheader(f"{c['name']} ({c['ticker']})")
    st.caption(f"CIK {cik10(cik)} · {c['sector']} · {c['sub_industry'] or '–'} · im S&P 500 seit "
               f"{c['date_added'] or 'unbekannt'} · Datenabruf {c['fetched_at'] or '–'}")
    if c.get("issues"):
        st.warning("Hinweise zur Datenqualität: " + c["issues"])

    period_mode = st.radio("Zeitraum", ["Geschäftsjahre", "Neuester Bilanzstichtag (10-Q/10-K)"], horizontal=True,
                           key="prof_mode")
    all_items = ctx.data["items"]
    it = all_items[all_items["cik"] == cik]
    if period_mode.startswith("Neuester"):
        lat = ctx.data["latest"]
        lat = lat[lat["cik"] == cik]
        if lat.empty:
            st.info("Kein Quartalsstichtag verfügbar.")
            return
        end = pd.Timestamp(lat["period_end"].iloc[0]).date()
        st.caption(f"Stichtag {end}, Quelle {lat['source_form'].iloc[0]}. Quartalsbilanzen sind ungeprüft und "
                   "werden nicht mit Geschäftsjahreswerten vermischt.")
        recs = {r["item_key"]: r for r in lat.to_dict("records")}
        rows = [{"Abschnitt": i.section, "Position": i.label_de, str(end): _cell(recs.get(i.key)),
                 "XBRL-Tag": recs[i.key]["tag"] if i.key in recs else ""} for s in SECTION_ORDER for i in ITEMS
                if i.section == s and (i.key in recs or s in ("Aktiva", "Passiva", "Eigenkapital"))]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=640)
        return

    years = ctx.years
    it = it[it["fiscal_year"].isin(years)]
    recs = {(r["item_key"], r["fiscal_year"]): r for r in it.to_dict("records")}
    has = {k for k, _ in recs}
    rows = []
    for s in SECTION_ORDER:
        for i in ITEMS:
            if i.section != s:
                continue
            if s not in ("Aktiva", "Passiva", "Eigenkapital", "Abstimmung") and i.key not in has:
                continue  # branchenspezifische Positionen nur zeigen, wenn gemeldet
            row = {"Abschnitt": s, "Position": i.label_de}
            for y in years:
                row[str(y)] = _cell(recs.get((i.key, y)))
            if len(years) >= 2:
                a, b = recs.get((i.key, years[-1])), recs.get((i.key, years[-2]))
                row[f"Δ {years[-2]}→{years[-1]}"] = (f"{a['value'] / b['value'] - 1:+.1%}"
                                                    if a is not None and b is not None and b["value"] > 0 else "NA")
            tags = sorted({recs[(i.key, y)]["tag"] for y in years if (i.key, y) in recs})
            row["XBRL-Tag(s)"] = ", ".join(tags)
            rows.append(row)
    bs = pd.DataFrame(rows)
    st.markdown(f"**Vollständige Bilanz** — {MARK_DERIVED} = berechnet, {MARK_RESTATED} = später korrigiert "
                "(Restatement; angezeigt wird der zuletzt gemeldete Wert)")
    st.dataframe(bs, width="stretch", hide_index=True, height=700)

    w = it.pivot_table(index="fiscal_year", columns="item_key", values="value", aggfunc="first").reset_index()
    left, right = st.columns([3, 2])
    with left:
        plot(balance_structure(w, ctx.mode, "Bilanzstruktur"), "prof_struct")
    with right:
        st.markdown("**Datenqualität**")
        core = [i.key for i in ITEMS if i.section in ("Aktiva", "Passiva", "Eigenkapital")]
        q = []
        for y in years:
            yi = it[it["fiscal_year"] == y]
            ta = yi[yi["item_key"] == "total_assets"]["value"]
            le = yi[yi["item_key"] == "liabilities_and_equity"]["value"]
            ident = ("✓" if abs(ta.iloc[0] - le.iloc[0]) <= max(0.005 * abs(ta.iloc[0]), 1) else "✗ Abweichung") \
                if len(ta) and len(le) else "NA"
            q.append({"GJ": y, "Abdeckung Standardpositionen": f"{yi['item_key'].isin(core).sum()}/{len(core)}",
                      "berechnet": int(yi["is_derived"].sum()), "Restatements": int(yi["restated"].fillna(False).sum()),
                      "Aktiva = Passiva": ident})
        st.dataframe(pd.DataFrame(q), hide_index=True, width="stretch")
        per = ctx.data["periods"]
        per = per[(per["cik"] == cik)]
        src = per.groupby("fiscal_year")["fy_source"].first()
        st.caption("Geschäftsjahr-Zuordnung: " + ", ".join(f"{y}: {src.get(y, 'NA')}" for y in years))

    st.markdown("**Kennzahlen (berechnet)**")
    met = ctx.data["metrics"]
    met = met[(met["cik"] == cik) & met["fiscal_year"].isin(years)]
    mrows = []
    for m in METRICS:
        row = {"Kennzahl": m.label}
        for y in years:
            r = met[(met["metric_key"] == m.key) & (met["fiscal_year"] == y)]
            if r.empty:
                row[str(y)] = "NA"
            else:
                r = r.iloc[0]
                row[str(y)] = fmt_metric(r["value"], m.kind) + (" (teilweise)" if r["status"] == "teilweise" else "")
        row["Formel"] = m.formula
        mrows.append(row)
    st.dataframe(pd.DataFrame(mrows), hide_index=True, width="stretch")
    with st.expander("Gründe für NA / teilweise berechnete Kennzahlen"):
        notes = met[met["status"] != "ok"][["fiscal_year", "metric_key", "status", "note"]].copy()
        notes["metric_key"] = notes["metric_key"].map(lambda k: METRIC_BY_KEY[k].label)
        st.dataframe(notes.sort_values(["metric_key", "fiscal_year"]), hide_index=True, width="stretch")

    st.markdown("**Originalquellen**")
    per = ctx.data["periods"]
    per = per[(per["cik"] == cik) & per["fiscal_year"].isin(years)].sort_values(["fiscal_year", "filed"])
    src_rows = [{"GJ": int(r["fiscal_year"]), "Stichtag": pd.Timestamp(r["period_end"]).date(), "Formular": r["form"],
                 "Eingereicht": pd.Timestamp(r["filed"]).date(), "Accession": r["accn"],
                 "Filing": filing_index_url(cik, r["accn"])} for _, r in per.iterrows()]
    st.dataframe(pd.DataFrame(src_rows), hide_index=True, width="stretch",
                 column_config={"Filing": st.column_config.LinkColumn("Filing (SEC-Archiv)", display_text="öffnen")})
    st.caption(f"Rohdaten: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10(cik)}.json · Werte, die nur als "
               "Vergleichsspalte eines späteren Berichts vorliegen, verweisen auf diesen späteren Bericht.")

    st.markdown("**Weitere XBRL-Instant-Positionen** (alle in Jahresberichten gemeldeten Stichtagswerte, die keiner "
                "Standardposition zugeordnet sind – Bilanz und Anhang)")
    facts = load_company_facts(db_version(), cik)
    if not facts.empty:
        facts = facts[facts["fiscal_year"].isin(years)]
        extra = facts[~facts["tag"].isin(TAG_TO_ITEM.keys()) | (facts["taxonomy"] != "us-gaap")]
        pv = extra.pivot_table(index=["taxonomy", "tag", "label", "unit"], columns="fiscal_year", values="value",
                               aggfunc="first").reset_index()
        q = st.text_input("Suche in Tags/Bezeichnungen", key="prof_tagsearch")
        if q:
            pv = pv[pv["tag"].str.contains(q, case=False) | pv["label"].str.contains(q, case=False)]
        st.dataframe(pv, hide_index=True, width="stretch", height=360)
        st.download_button("Alle Facts dieses Unternehmens (Excel)",
                           to_excel_bytes({"Standardpositionen": it.drop(columns=["name"], errors="ignore"),
                                           "Alle Instant-Facts": facts}),
                           f"{c['primary_ticker']}_bilanz.xlsx", key="prof_xlsx")
    export_buttons(bs, f"{c['primary_ticker']}_bilanz", "prof_exp")
