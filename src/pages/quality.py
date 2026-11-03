from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.data import item_label, item_options
from src.components.ui import export_buttons, plot, sidebar_context
from src.config import get_settings
from src.normalization.taxonomy import ITEMS
from src.visualization.charts import coverage_heatmap, fmt_usd, heatmap


def page() -> None:
    ctx = sidebar_context()
    st.title("Datenqualität und Abdeckung")
    comps = ctx.companies
    meta = ctx.data["meta"]

    st.markdown("#### Abrufstatus")
    counts = comps["fetch_status"].value_counts()
    cols = st.columns(4)
    for col, s in zip(cols, ["ok", "keine Daten", "Fehler", "nicht abgerufen"]):
        col.metric(s, int(counts.get(s, 0)))
    st.caption(f"Universum: {len(comps)} Unternehmen (eindeutige CIKs; Aktiengattungen wie GOOG/GOOGL zusammengefasst). "
               f"Indexzusammensetzung: {meta.get('membership_method', '–')}.")
    bad = comps[comps["fetch_status"] != "ok"][["primary_ticker", "name", "cik", "fetch_status", "fetch_error"]]
    if len(bad):
        with st.expander(f"{len(bad)} Unternehmen ohne Daten"):
            st.dataframe(bad, hide_index=True, width="stretch")

    st.markdown("#### Abdeckung je Position und Jahr")
    loaded = set(comps[comps["fetch_status"] == "ok"]["cik"])
    base = st.radio("Bezugsgröße", ["Unternehmen mit geladenen Daten", "Gesamtes Universum"], horizontal=True,
                    key="q_base")
    denom = len(loaded) if base.startswith("Unternehmen") else len(ctx.universe)
    it = ctx.items
    keys = item_options(["Aktiva", "Passiva", "Eigenkapital", "Abstimmung"])
    cnt = it[it["item_key"].isin(keys)].groupby(["item_key", "fiscal_year"])["cik"].nunique().unstack() \
        .reindex(index=keys, columns=ctx.years)
    pct = cnt / denom if denom else cnt * float("nan")
    text = pct.map(lambda v: f"{v:.0%}" if pd.notna(v) else "0%")
    pct = pct.fillna(0)
    pct.index = text.index = [item_label(k) for k in pct.index]
    pct.columns = text.columns = [str(c) for c in pct.columns]
    plot(heatmap(pct, text, ctx.mode, f"Anteil meldender Unternehmen (n={denom})", color_by_column_rank=False),
         "q_cov")
    st.caption("Viele Positionen sind branchenabhängig (Banken melden z. B. kein Umlaufvermögen). Eine niedrige "
               "Abdeckung bedeutet 'nicht gemeldet', nicht 'null'.")

    st.markdown("#### Abdeckung je Unternehmen")
    key = st.selectbox("Position", keys, format_func=item_label, key="q_item")
    sub = it[it["item_key"] == key]
    status = sub.assign(s=lambda d: (~d["is_derived"].astype(bool)).astype(int) + 1)
    mat = status.pivot_table(index="cik", columns="fiscal_year", values="s", aggfunc="max") \
        .reindex(index=sorted(ctx.universe), columns=ctx.years).fillna(0)
    lab = comps.set_index("cik")["primary_ticker"]
    mat.index = [lab.get(c, str(c)) for c in mat.index]
    only_gaps = st.toggle("Nur Unternehmen mit Lücken oder berechneten Werten", value=True, key="q_gaps")
    if only_gaps:
        mat = mat[(mat < 2).any(axis=1)]
    st.caption(f"{len(mat)} Unternehmen angezeigt")
    if len(mat):
        plot(coverage_heatmap(mat.head(150), ctx.mode, f"{item_label(key)}: gemeldet / berechnet / fehlt"
                              + (" (erste 150)" if len(mat) > 150 else "")), "q_comp")

    st.markdown("#### Plausibilitätsprüfung: Bilanzsumme = Summe Passiva")
    w = it[it["item_key"].isin(["total_assets", "liabilities_and_equity"])] \
        .pivot_table(index=["primary_ticker", "fiscal_year"], columns="item_key", values="value").dropna()
    if len(w):
        w["Abweichung"] = (w["total_assets"] - w["liabilities_and_equity"]).abs() / w["total_assets"].abs()
        badw = w[w["Abweichung"] > 0.005].reset_index()
        st.write(f"{len(w)} Unternehmensjahre geprüft, {len(badw)} mit Abweichung > 0,5 %.")
        if len(badw):
            st.dataframe(badw, hide_index=True, width="stretch")

    st.markdown("#### Restatements und Mehrfachmeldungen")
    r = it[it["restated"] == True][["primary_ticker", "fiscal_year", "item_key", "original_value", "value",  # noqa: E712
                                    "n_versions", "accn"]].copy()
    r["Position"] = r["item_key"].map(item_label)
    r["Erstmeldung"] = r["original_value"].map(fmt_usd)
    r["Zuletzt gemeldet (verwendet)"] = r["value"].map(fmt_usd)
    st.write(f"{len(r)} Werte wurden in späteren Berichten geändert; das Dashboard verwendet den zuletzt gemeldeten Wert.")
    st.dataframe(r[["primary_ticker", "fiscal_year", "Position", "Erstmeldung", "Zuletzt gemeldet (verwendet)",
                    "n_versions"]], hide_index=True, width="stretch", height=260)
    conf = it[it["conflict"] == True]  # noqa: E712
    if len(conf):
        st.warning(f"{len(conf)} Werte mit widersprüchlichen Angaben innerhalb desselben Berichts.")

    st.markdown("#### Hinweise aus der Normalisierung")
    iss = comps[comps["issues"].fillna("") != ""][["primary_ticker", "name", "issues"]]
    st.dataframe(iss, hide_index=True, width="stretch", height=220)
    per = ctx.data["periods"]
    if len(per):
        st.caption("Herkunft der Geschäftsjahr-Zuordnung: " +
                   ", ".join(f"{k}: {v}" for k, v in per["fy_source"].value_counts().items()))

    st.markdown("#### Stichproben-Verifikation gegen die SEC Company Concept API")
    rep = get_settings().data_dir / "verification_report.csv"
    if rep.exists():
        v = pd.read_csv(rep)
        st.write(v["result"].value_counts().to_dict())
        st.dataframe(v, hide_index=True, width="stretch",
                     column_config={"filing": st.column_config.LinkColumn(display_text="Originalbericht")})
    else:
        st.info("Noch keine Verifikation durchgeführt. Ausführen mit: `python -m src.verify --sample 15` "
                "(oder verify.bat). Der Bericht erscheint danach hier.")
    export_buttons(comps.drop(columns=["label"], errors="ignore"), "abdeckung_unternehmen", "q_exp")
