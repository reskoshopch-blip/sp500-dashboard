from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.components.ml_data import DISCLAIMER, flabel, pick, require_ml, selectors
from src.components.ui import export_buttons, plot, theme_mode
from src.ml.model import FEATURE_SETS, TARGETS
from src.visualization.charts import bar_chart, line_chart


def _t(v) -> str:
    return "–" if v is None or pd.isna(v) else f"{v:.2f}"


def page() -> None:
    st.title("ML-Backtest (historischer Test)")
    ml = require_ml()
    if ml is None:
        return
    mode = theme_mode()
    meta = ml["meta"]
    target, fset = selectors("bt")
    s = meta["summaries"].get(f"{target}|{fset}", {})
    if not s or "auc_mean" not in s:
        st.warning(s.get("error", "Für diese Auswahl gibt es keine Backtest-Ergebnisse."))
        return
    n_txt = f"{s['n']:,}".replace(",", "'")
    st.caption(f"Walk-forward: Für jedes Testjahr wird ein neues Modell nur mit Daten trainiert, deren 12-Monats-Ergebnis "
               f"vor Beginn des Testjahres feststand. Testjahre {s['first_fold']}–{s['last_fold']}, "
               f"{n_txt} Out-of-sample-Prognosen.")
    pos = TARGETS[target]["positive"]
    c = st.columns(4)
    c[0].metric("Ø AUC je Testjahr", f"{s['auc_mean']:.3f}",
                help="Trennschärfe: 0.5 = Zufall, 1.0 = perfekt. Bei Aktienrenditen gelten 0.53–0.58 bereits als "
                     "nützlich, Werte über 0.65 sind verdächtig (Prüfen auf Datenfehler).")
    c[1].metric("Testjahre mit AUC > 0.5", f"{s['share_years_auc_gt_05']:.0%}")
    c[2].metric("Quote Top − Bottom", f"{s['rate_spread_mean'] * 100:+.1f} Pp.",
                f"t = {_t(s['rate_spread_t'])}", delta_color="off",
                help=f"Anteil „{pos}“: durchschnittlicher Unterschied je Testjahr zwischen den 20 % mit der höchsten "
                     "und der tiefsten Prognose (Quintile). |t| > 2 gilt als statistisch belastbar.")
    c[3].metric("Überrendite Top − Bottom", f"{s['excess_spread_mean'] * 100:+.1f} Pp.",
                f"t = {_t(s['excess_spread_t'])}", delta_color="off",
                help="Überrendite gegenüber SPY in den 12 Monaten nach t0, Differenz der Quintile.")

    fm = pick(ml["folds"], target, fset).copy()
    fm["Jahr"] = fm["fold"].astype(int).astype(str)
    left, right = st.columns(2)
    with left:
        long = pd.concat([fm.assign(S="Top-Quintil", v=fm["top_excess"]),
                          fm.assign(S="Alle", v=fm["all_excess"]),
                          fm.assign(S="Bottom-Quintil", v=fm["bottom_excess"])])
        plot(bar_chart(long, "Jahr", "v", "S", mode, "Ø 12-Monats-Überrendite vs. SPY je Testjahr", kind="pct"),
             "bt_years")
    with right:
        long = pd.concat([fm.assign(S="Modell (AUC)", v=fm["auc"]), fm.assign(S="Zufall", v=0.5)])
        plot(line_chart(long, "Jahr", "v", "S", mode, "Trennschärfe (AUC) je Testjahr", kind="num"), "bt_auc")

    oos = pick(ml["oos"], target, fset)
    left, right = st.columns(2)
    with left:
        qq = oos.groupby("q").agg(rate=("y", "mean"), excess=("excess_ret", "mean"), n=("y", "size")).reset_index()
        qq["Quintil"] = qq["q"].astype(int).map(lambda q: f"Q{q}" + (" (tief)" if q == 1 else " (hoch)" if q == 5 else ""))
        plot(bar_chart(qq, "Quintil", "rate", None, mode, f"Tatsächliche Quote „{pos}“ je Prognose-Quintil",
                       kind="pct"), "bt_q")
        st.caption(f"Basisquote über alle Testjahre: {s['base_rate']:.1%}. Steigen die Balken von Q1 nach Q5, "
                   "sortiert das Modell sinnvoll.")
    with right:
        cal = pick(ml["calibration"], target, fset)
        cl = pd.concat([cal.assign(S="Prognostiziert", v=cal["prob_mean"]),
                        cal.assign(S="Tatsächlich eingetreten", v=cal["observed"])])
        cl["Dezil"] = (cl["bin"].astype(int) + 1).astype(str)
        plot(line_chart(cl, "Dezil", "v", "S", mode, "Kalibrierung: Prognose vs. Realität je Dezil", kind="pct"),
             "bt_cal")

    st.markdown("#### Nach Marktregime bei Veröffentlichung")
    rg = pick(ml["regimes"], target, fset)
    st.dataframe(pd.DataFrame({
        "Regime": rg["regime"], "n": rg["n"], "AUC": rg["auc"].round(3),
        "Basisquote %": (rg["base_rate"] * 100).round(1), "Top-Quintil Quote %": (rg["top_rate"] * 100).round(1),
        "Bottom-Quintil Quote %": (rg["bottom_rate"] * 100).round(1),
        "Top Ø Überrendite %": (rg["top_excess"] * 100).round(1),
        "Bottom Ø Überrendite %": (rg["bottom_excess"] * 100).round(1)}), hide_index=True, width="stretch")

    st.markdown("#### Welche Merkmale waren im Test wichtig?")
    imp = pick(ml["importance"], target, fset).head(15).copy()
    imp["Merkmal"] = imp["feature"].map(lambda f: flabel(meta, f))
    plot(bar_chart(imp, "Merkmal", "importance", None, mode,
                   "Permutation Importance (AUC-Verlust beim Vertauschen, Ø über Testjahre)", kind="num",
                   horizontal=True, height=520), "bt_imp")

    st.markdown("#### Vergleich aller Modelle")
    rows = []
    for k, v in meta["summaries"].items():
        t, f = k.split("|")
        if "auc_mean" in v:
            rows.append({"Ziel": TARGETS[t]["label"], "Merkmale": FEATURE_SETS[f]["label"],
                         "Ø AUC": round(v["auc_mean"], 3), "AUC gepoolt": round(v["auc_pooled"], 3),
                         "Ø IC (Spearman)": round(v["ic_mean"], 3),
                         "Quote Top−Bottom (Pp.)": round(v["rate_spread_mean"] * 100, 1),
                         "t": round(v["rate_spread_t"], 2) if v["rate_spread_t"] == v["rate_spread_t"] else None,
                         "Überrendite Top−Bottom (Pp.)": round(v["excess_spread_mean"] * 100, 1)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    export_buttons(oos.drop(columns=["target", "fset"], errors="ignore"), f"backtest_{target}_{fset}", "bt_exp")
    with st.expander("Grenzen dieses Backtests"):
        st.markdown(
            "- **Survivorship-Bias:** Es werden nur heutige S&P-500-Mitglieder getestet. Firmen, die pleitegingen oder "
            "aus dem Index fielen, fehlen – Absturzquoten sind dadurch eher zu tief, Renditen eher zu hoch.\n"
            "- **Keine Transaktionskosten**, keine Steuern; Überrenditen sind Durchschnitte von 12-Monats-Haltedauern, "
            "keine handelbare Equity-Kurve.\n"
            "- **Point-in-time:** Bilanzwerte wie erstmals gemeldet, Prognosezeitpunkt erst nach dem Filing – "
            "dadurch kein Blick in die Zukunft.\n"
            "- **Wenige Jahre** (rund 11 Testjahre): Einzeljahre schwanken stark; der t-Wert zeigt, wie belastbar "
            "ein Unterschied ist.")
    st.caption(DISCLAIMER)
