from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ml_data import DISCLAIMER, flabel, load_model, ml_version, pick, require_ml, selectors
from src.components.ui import export_buttons, plot, theme_mode
from src.ml.model import TARGETS, prepare_X
from src.visualization.charts import driver_chart


PCT = {"equity_ratio", "liab_to_assets", "mkt_vs_sma200", "mkt_mom_12m", "mkt_vol_3m", "mom_12m", "mom_1m",
       "vol_12m", "dist_52w_high"}


def _fmt(v, f: str) -> str:
    if v is None or pd.isna(v):
        return "nicht gemeldet"
    if f == "d_current_ratio":
        return f"{v:+.2f}"
    if f.startswith(("g_", "d_")):
        return f"{v:+.1%}"
    if f.endswith("_to_assets") or f in PCT:
        return f"{v:.1%}"
    return f"{v:,.2f}"


def page() -> None:
    st.title("ML-Prognosen")
    ml = require_ml()
    if ml is None:
        return
    mode = theme_mode()
    meta = ml["meta"]
    target, fset = selectors("fc")
    up_good = target == "outperf"
    live = pick(ml["live"], target, fset).copy()
    s = meta["summaries"].get(f"{target}|{fset}", {})
    if live.empty:
        st.info("Für diese Auswahl gibt es keine offenen Prognosen (keine Bilanz der letzten 12 Monate mit Kursdaten).")
        return
    n_txt = f"{meta['n_labeled']:,}".replace(",", "'")
    st.caption(f"Prognose für die 12 Monate nach Veröffentlichung der jeweils neuesten Jahresbilanz. Modell trainiert auf "
               f"{n_txt} abgeschlossenen Firmenjahren; im Walk-forward-Backtest Ø AUC "
               f"{s.get('auc_mean', float('nan')):.3f} (0.5 = Zufall). Stand Kurse {meta['last_price']}.")

    c1, c2, c3 = st.columns([2, 2, 1])
    sectors = c1.multiselect("Branche", sorted(live["sector"].unique()), key="fc_sec", placeholder="Alle")
    q = c2.text_input("Suche Ticker / Name", key="fc_q")
    order_desc = c3.toggle("Höchste zuerst", value=True, key="fc_ord")
    v = live
    if sectors:
        v = v[v["sector"].isin(sectors)]
    if q:
        v = v[v["primary_ticker"].str.contains(q, case=False) | v["name"].str.contains(q, case=False)]
    v = v.sort_values("prob", ascending=not order_desc)
    pos = TARGETS[target]["positive"]
    table = pd.DataFrame({
        "Ticker": v["primary_ticker"], "Unternehmen": v["name"], "Branche": v["sector"],
        "Bilanz GJ": v["fiscal_year"].astype(int), "Prognose ab": pd.to_datetime(v["t0"]).dt.date,
        "Regime bei t0": v["regime"],
        f"Wahrscheinlichkeit „{pos}“": v["prob"] * 100,
        "Rang (Perzentil)": v["pct_rank"] * 100,
        "Historisch in diesem Bereich: Quote": v["hist_rate"] * 100,
        "Historisch: Ø Überrendite": v["hist_excess"] * 100,
        "Historisch: Ø max. Rückgang": v["hist_dd"] * 100,
        "Historisch: n": v["hist_n"],
    })
    pct = st.column_config.NumberColumn(format="%.1f %%")
    st.dataframe(table, hide_index=True, width="stretch", height=420, column_config={
        f"Wahrscheinlichkeit „{pos}“": st.column_config.ProgressColumn(format="%.1f %%", min_value=0, max_value=100),
        "Rang (Perzentil)": st.column_config.NumberColumn(format="%.0f"),
        "Historisch in diesem Bereich: Quote": pct, "Historisch: Ø Überrendite": pct,
        "Historisch: Ø max. Rückgang": pct, "Bilanz GJ": st.column_config.NumberColumn(format="%d")})
    st.caption("„Historisch in diesem Bereich“: Was im Backtest tatsächlich passierte, wenn das Modell eine ähnliche "
               "Wahrscheinlichkeit ausgab (gleiches Dezil der Out-of-sample-Prognosen). Das ist die ehrlichere Zahl als "
               "die Modellwahrscheinlichkeit selbst.")
    export_buttons(table, f"prognosen_{target}_{fset}", "fc_exp")

    st.divider()
    st.markdown("#### Warum? – Treiber der Prognose und Szenario")
    opts = v["primary_ticker"] + " · " + v["name"]
    if opts.empty:
        return
    sel = st.selectbox("Unternehmen", opts.tolist(), key="fc_comp")
    row = v[opts == sel].iloc[0]
    cik = int(row["cik"])
    dr = pick(ml["drivers"], target, fset)
    dr = dr[dr["cik"] == cik].copy()
    dr["label"] = dr["feature"].map(lambda f: flabel(meta, f))
    dr["value_txt"] = [_fmt(x, f) for x, f in zip(dr["value"], dr["feature"])]
    top = dr.reindex(dr["delta"].abs().sort_values(ascending=False).index).head(10)
    left, right = st.columns([3, 2])
    with left:
        plot(driver_chart(top, mode, f"{row['primary_ticker']}: Wahrscheinlichkeit {row['prob']:.1%} – grösste Treiber",
                          up_is_good=up_good), "fc_drv")
        st.caption("Beitrag = Prognose minus Prognose, wenn dieses Merkmal auf den typischen Wert (Median) gesetzt wird. "
                   "Blau = günstig, rot = ungünstig.")
    with right:
        st.markdown("**Szenario: Was wäre, wenn …?**")
        bundle = load_model(ml_version(), target, fset)
        ds = ml["dataset"]
        base = ds[(ds["cik"] == cik) & (pd.to_datetime(ds["t0"]) == pd.to_datetime(row["t0"]))]
        if bundle is None or base.empty:
            st.info("Modell oder Datensatzzeile nicht verfügbar.")
        else:
            x = base.iloc[[0]].copy()
            feats = [f for f in top["feature"].tolist() if f != "sector"][:6]
            for f in feats:
                cur = x[f].iloc[0]
                val = st.number_input(flabel(meta, f), value=float(cur) if pd.notna(cur) else 0.0,
                                      key=f"fc_sc_{cik}_{f}_{target}_{fset}", format="%.4f",
                                      help=f"Aktuell: {_fmt(cur, f)} (Quoten als Dezimalzahl, 0.10 = 10 %)")
                x[f] = val
            p_new = bundle["model"].predict_proba(prepare_X(x, bundle["features"], bundle["sectors"]))[:, 1][0]
            st.metric("Neue Wahrscheinlichkeit", f"{p_new:.1%}", f"{(p_new - row['prob']) * 100:+.1f} Pp.",
                      delta_color="normal" if up_good else "inverse")
            st.caption("Das Szenario ändert nur die Eingaben des Modells. Es zeigt, wie das Modell auf eine andere "
                       "Bilanzentwicklung reagiert hätte – keine Kausalaussage.")
    st.caption(DISCLAIMER)
