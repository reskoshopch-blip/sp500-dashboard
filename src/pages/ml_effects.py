from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ml_data import DISCLAIMER, flabel, pick, require_ml
from src.components.ui import plot, theme_mode
from src.ml.features import BS_FEATURES
from src.ml.model import TARGETS
from src.visualization.charts import REGIME_ORDER, bar_chart, heatmap, line_chart

QLAB = {0: "nicht gemeldet", 1: "Q1 tief", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5 hoch"}


def page() -> None:
    st.title("Wirkung von Bilanzentwicklungen")
    ml = require_ml()
    if ml is None:
        return
    mode = theme_mode()
    meta = ml["meta"]
    b = ml["buckets"]
    if b.empty:
        st.info("Keine Auswertung verfügbar.")
        return
    st.caption("Zwei Sichtweisen: (1) **Modellfrei** – Aktien werden je Jahr nach einem Merkmal in fünf Gruppen "
               "(Quintile) geteilt; gezeigt wird, was danach tatsächlich passierte. (2) **Modellsicht** – wie die "
               "Prognose des gelernten Modells auf das Merkmal reagiert (Partial Dependence), getrennt nach Regime.")

    st.markdown("#### Überblick: Unterschied Q5 − Q1 bei der 12-Monats-Überrendite")
    a = b[b["q"].isin([1, 5])].pivot_table(index=["feature", "regime"], columns="q", values="excess_mean")
    a = (a[5] - a[1]).unstack("regime").reindex(columns=["Alle", "Bull", "Bear"])
    a = a.reindex(a["Alle"].abs().sort_values(ascending=False).index).head(20)
    ranked = list(a.index)
    text = a.map(lambda v: f"{v * 100:+.1f} Pp." if pd.notna(v) else "NA")
    a.index = text.index = [flabel(meta, f) for f in a.index]
    plot(heatmap(a, text, mode, "Q5 (hoher Wert) minus Q1 (tiefer Wert): Ø Überrendite vs. SPY"), "fx_over")
    st.caption("Positiv = Aktien mit hohem Merkmalswert entwickelten sich besser. Unterschiedliche Werte in Bull und "
               "Bear zeigen, dass die Wirkung vom Marktregime abhängt. Farbe = Rang innerhalb der Spalte.")

    st.divider()
    feats = ranked + [f for f in BS_FEATURES if f in set(b["feature"]) and f not in ranked]
    f = st.selectbox("Merkmal im Detail (sortiert nach Stärke des Unterschieds)", feats,
                     format_func=lambda k: flabel(meta, k), key="fx_feat")
    sub = b[b["feature"] == f].copy()
    sub["Gruppe"] = sub["q"].astype(int).map(QLAB)
    sub["Regime"] = sub["regime"]
    sub = sub.sort_values(["regime", "q"])
    order = REGIME_ORDER
    left, right = st.columns(2)
    with left:
        plot(bar_chart(sub, "Gruppe", "excess_mean", "Regime", mode, "Ø Überrendite vs. SPY (12 Monate)", kind="pct",
                       stable_order=order), "fx_exc")
    with right:
        plot(bar_chart(sub, "Gruppe", "dd_rate", "Regime", mode, "Anteil Kursrückgang > 30 %", kind="pct",
                       stable_order=order), "fx_dd")
    tab = sub[sub["regime"] == "Alle"][["Gruppe", "n", "value_median", "excess_mean", "excess_median",
                                       "outperf_rate", "dd_rate"]]
    st.dataframe(tab.rename(columns={"value_median": "Median Merkmal", "excess_mean": "Ø Überrendite",
                                     "excess_median": "Median Überrendite", "outperf_rate": "Quote schlägt Index",
                                     "dd_rate": "Quote Rückgang > 30 %"}),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="percent")
                                for c in ["Ø Überrendite", "Median Überrendite", "Quote schlägt Index",
                                          "Quote Rückgang > 30 %"]})

    st.markdown("#### Modellsicht (Partial Dependence)")
    target = st.radio("Ziel", list(TARGETS), format_func=lambda k: TARGETS[k]["label"], horizontal=True, key="fx_t")
    pdp = pick(ml["pdp"], target, "bilanz")
    pf = pdp[pdp["feature"] == f]
    if pf.empty:
        st.info("Für dieses Merkmal wurde keine Modellkurve berechnet (nur die 12 wichtigsten Merkmale je Modell).")
    else:
        pf = pf.copy()
        pf["x_txt"] = pf["x"].map(lambda v: f"{v:.3g}")
        plot(line_chart(pf.sort_values("x"), "x_txt", "prob", "regime", mode,
                        f"Modell-Wahrscheinlichkeit „{TARGETS[target]['positive']}“ in Abhängigkeit von "
                        f"{flabel(meta, f)}", kind="pct", stable_order=order), "fx_pdp")
        st.caption("Bereich 5.–95. Perzentil des Merkmals. Die Kurven zeigen Zusammenhänge, die das Modell gelernt "
                   "hat – keine Ursache-Wirkung.")
    st.caption(DISCLAIMER)
