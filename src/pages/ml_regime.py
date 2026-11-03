from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ml_data import DISCLAIMER, load_top, require_ml, top_version
from src.components.ui import plot, theme_mode
from src.visualization.charts import REGIME_ORDER, bar_chart, regime_chart


def page() -> None:
    st.title("Marktregime")
    ml = require_ml()
    if ml is None:
        return
    mode = theme_mode()
    mf = ml["regime"].dropna(subset=["close"])
    mf["date"] = pd.to_datetime(mf["date"])
    known = mf[mf["regime"] != "unbekannt"]
    cur = known.iloc[-1]
    runs = (known["regime"] != known["regime"].shift()).cumsum()
    since = known[runs == runs.iloc[-1]]["date"].iloc[0]
    c = st.columns(3)
    c[0].metric("Aktuelles Regime", "Bullish" if cur["regime"] == "Bull" else "Bearish",
                help="Bull = SPY schliesst über seinem 200-Tage-Durchschnitt, Bear = darunter.")
    c[1].metric("Seit", f"{since.date()}", f"{(cur['date'] - since).days} Tage", delta_color="off")
    c[2].metric("SPY vs. 200-Tage-Durchschnitt", f"{cur['vs_sma200']:+.1%}", delta_color="off")
    st.caption(f"Stand der Kurse: {cur['date'].date()}. Anteil Bear-Tage seit 2009: "
               f"{(known[known['date'] >= '2009-01-01']['regime'] == 'Bear').mean():.0%}.")
    top = load_top(top_version("SPY"), "SPY")
    warn = None
    if top is not None and len(top["daily"]) and (top["meta"].get("auc_oos") or 0) >= 0.55:
        td = top["daily"].set_index("date")
        lvl = 2 * top["meta"]["base_rate"]
        warn = {"Top-Warnung (ML)": ("ml", (td["prob"] >= lvl).astype(int).where(td["prob"].notna(), 0))}
    plot(regime_chart(mf[mf["date"] >= "2009-01-01"], mode, "S&P 500 (SPY) – Bull-Phasen grün, Bear-Phasen rot",
                      warn=warn), "rg_chart")
    if warn is None:
        st.caption("Tipp: Unter „Top-Warnung“ gibt es Frühwarnsignale vor Rückgängen – sie erscheinen dann auch hier "
                   "als oranger Streifen.")

    st.markdown("#### Wie entwickelten sich Aktien nach Bilanzveröffentlichung – je nach Regime?")
    ds = ml["dataset"]
    lab = ds[ds["excess_ret"].notna() & ds["regime"].isin(["Bull", "Bear"])]
    if lab.empty:
        st.info("Noch keine abgeschlossenen 12-Monats-Ergebnisse.")
        return
    g = lab.groupby("regime").agg(n=("excess_ret", "size"), outperf=("y_outperf", "mean"),
                                  excess=("excess_ret", "median"), ret=("ret", "median"),
                                  dd=("y_dd30", "mean")).reindex(["Bull", "Bear"]).reset_index()
    long = pd.concat([g.assign(M="Anteil schlägt S&P 500", v=g["outperf"]),
                      g.assign(M="Anteil Rückgang > 30 %", v=g["dd"])])
    long["Regime"] = long["regime"]
    left, right = st.columns(2)
    with left:
        plot(bar_chart(long, "M", "v", "Regime", mode, "Ausgang in den 12 Monaten nach t0", kind="pct",
                       stable_order=REGIME_ORDER), "rg_out")
        st.caption(" · ".join(f"{r.regime}: n={r.n}, Median-Überrendite {r.excess:+.1%}" for r in g.itertuples()))
    with right:
        yr = lab.groupby(["t0_year", "regime"]).size().unstack(fill_value=0) \
            .reindex(columns=["Bull", "Bear"], fill_value=0).reset_index()
        yl = yr.melt(id_vars="t0_year", var_name="Regime", value_name="n")
        yl["Jahr"] = yl["t0_year"].astype(str)
        plot(bar_chart(yl, "Jahr", "n", "Regime", mode, "Firmenjahre je Regime bei Veröffentlichung", kind="num",
                       stack=True, stable_order=REGIME_ORDER), "rg_years")
    st.caption("t0 = erster Handelstag nach dem Einreichen des Jahresberichts. Regime = Zustand des Gesamtmarkts an t0. "
               "Das Regime ist auch ein Merkmal der ML-Modelle – so lernen sie, ob Bilanzentwicklungen in Bull- und "
               "Bear-Phasen unterschiedlich wirken (Seite „Wirkung von Bilanzentwicklungen“).")
    st.caption(DISCLAIMER)
