"""Modell-Konsens S&P 500: alle Modelle zusammen auswerten – Trefferquote, wenn alle (oder k) bullisch sind."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.components.ml_data import DISCLAIMER
from src.components.ui import plot, theme_mode
from src.config import get_settings
from src.consensus import DEFAULT, MODELS, by_count, learned_combo, load_frame, per_model, trade_stats, trades
from src.visualization.charts import PALETTE, STATUS, _runs, base_layout

HORIZONS = {5: "1 Woche", 10: "2 Wochen", 21: "1 Monat", 63: "3 Monate"}
PCT = st.column_config.NumberColumn(format="percent")


@st.cache_data(show_spinner=False)
def frame(version: float, data_dir: str) -> pd.DataFrame:
    return load_frame()


def _version() -> float:
    d = get_settings().data_dir
    ps = [d / "ml" / "health_SPY_daily.parquet", d / "fundamental" / "fund_SPY_daily.parquet",
          d / "cot" / "SPX_weekly.parquet", d / "ml" / "top_SPY_daily.parquet"]
    return sum(p.stat().st_mtime for p in ps if p.exists())


def page() -> None:
    st.title("Modell-Konsens S&P 500")
    d = frame(_version(), str(get_settings().data_dir))
    if d.empty:
        st.info("Zuerst im Markt-Terminal die Markt-Analysen berechnen (Seite „Daten laden“).")
        return
    avail = [m for m in MODELS if m in d and d[m].notna().sum() > 500]
    c1, c2, c3 = st.columns([3, 1, 2])
    models = c1.multiselect("Modelle", avail, default=[m for m in DEFAULT if m in avail], format_func=MODELS.get,
                            key="cons_models")
    th = c2.slider("Bullisch ab", 50, 80, 60, 5, key="cons_th")
    h = c3.radio("Horizont", list(HORIZONS), index=2, horizontal=True, format_func=HORIZONS.get, key="cons_h")
    if not models:
        st.warning("Mindestens ein Modell wählen.")
        return
    x = d.dropna(subset=models)
    last = x.iloc[-1]
    n_bull = int((last[models] >= th).sum())
    st.caption(f"Gemeinsame Historie aller gewählten Modelle: {x.index.min():%d.%m.%Y} – {x.index.max():%d.%m.%Y} "
               f"({len(x):,} Handelstage). Bullisch = Score ≥ {th}, bärisch = ≤ {100 - th}. Trefferquote = Anteil, in "
               f"dem der S&P 500 (SPY) nach {HORIZONS[h]} höher stand.".replace(f"{len(x):,}", f"{len(x):,}".replace(",", "'")))

    # ---------------------------------------------------------------- heute
    st.markdown(f"##### Heute ({x.index[-1]:%d.%m.%Y}): **{n_bull} von {len(models)}** Modellen bullisch")
    cols = st.columns(len(models))
    for col, m in zip(cols, models):
        v = last[m]
        icon = "🟢" if v >= th else ("🔴" if v <= 100 - th else "🟡")
        col.metric(MODELS[m], f"{v:.0f}", icon, delta_color="off")
    gex = _gex_now()
    if gex:
        st.caption(gex)

    # ---------------------------------------------------------------- Hauptauswertung
    cnt = by_count(d, models, th, h)
    allrow = cnt[cnt["k"] == len(models)].iloc[0]
    base = cnt[cnt["k"] == -1].iloc[0]
    best = cnt[(cnt["k"] >= 0) & (cnt["Tage"] >= 100)].sort_values("Trefferquote", ascending=False).head(1)
    cond_all = (x[models] >= th).all(axis=1)
    tr = trade_stats(trades(d, cond_all, h))
    m = st.columns(4)
    m[0].metric(f"Alle {len(models)} bullisch: Trefferquote",
                f"{allrow['Trefferquote']:.0%}" if pd.notna(allrow["Trefferquote"]) else "–",
                f"{(allrow['Trefferquote'] - base['Trefferquote']) * 100:+.0f} Pp. vs. Basis"
                if pd.notna(allrow["Trefferquote"]) else None)
    m[1].metric("Wie oft kam das vor?", f"{int(allrow['Tage'])} Tage",
                f"{tr['n']} getrennte Signale" if tr.get("n") else "keine Signale", delta_color="off")
    m[2].metric("Basis (irgendein Tag)", f"{base['Trefferquote']:.0%}", f"Ø {base['Ø Rendite']:+.2%}",
                delta_color="off")
    if len(best):
        b = best.iloc[0]
        m[3].metric("Beste Stufe", b["Bullische Modelle"], f"Trefferquote {b['Trefferquote']:.0%}", delta_color="off")
    if allrow["Tage"] < 100:
        st.warning(f"„Alle bullisch“ kam nur an {int(allrow['Tage'])} Tagen vor – zu selten für eine verlässliche "
                   "Aussage. Die Modelle messen teils dasselbe (Trend), und wenn wirklich alles bullisch ist, ist "
                   "der Markt meist schon gut gelaufen.")
    elif pd.notna(allrow["Trefferquote"]) and allrow["Trefferquote"] < base["Trefferquote"]:
        st.warning("Wenn alle Modelle bullisch waren, lag die Trefferquote **unter** der Basis – volle Einigkeit war "
                   "historisch eher ein Zeichen für einen heiss gelaufenen Markt als für weiteres Plus.")

    left, right = st.columns([3, 2])
    with left:
        plot(count_chart(cnt, len(models), base["Trefferquote"], theme_mode()), "cons_cnt")
    with right:
        st.dataframe(cnt.drop(columns="k"), hide_index=True, width="stretch",
                     column_config={"Trefferquote": PCT, "Ø Rendite": PCT})
        if tr.get("n"):
            st.caption(f"Als Trades (Einstieg am ersten Tag „alle bullisch“, {h} Tage halten, nicht überlappend): "
                       f"{tr['n']} Trades, Winrate {tr['winrate']:.0%}, Ø {tr['avg']:+.2%}, schlechtester "
                       f"{tr['worst']:+.1%}.")

    plot(history_chart(x, models, th, theme_mode()), "cons_hist")

    t1, t2, t3 = st.tabs(["Jedes Modell einzeln", "Gelernte Kombination (KI)", "Short-Seite"])
    with t1:
        st.dataframe(per_model(d, models, th, h), hide_index=True, width="stretch",
                     column_config={"Trefferquote": PCT, "Ø Rendite": PCT})
        st.caption("Trefferquote, wenn nur dieses Modell bullisch (≥) bzw. bärisch (≤) war. Vergleich mit der Basis.")
    with t2:
        lc = learned_combo(d, models, h)
        if not lc:
            st.info("Zu wenig gemeinsame Historie.")
        else:
            st.metric("Test-AUC der gelernten Kombination", f"{lc['auc']:.2f}",
                      help="0.5 = Zufall. Jährlich neu gelernt, nur mit abgeschlossenen Zeiträumen.")
            st.dataframe(lc["table"], hide_index=True, width="stretch",
                         column_config={"Trefferquote": PCT, "Ø Rendite": PCT})
            st.caption(f"Logistische Regression über alle gewählten Modelle, Test ab {lc['start']}. Liegt die AUC unter "
                       "0.5, hat die Kombination historisch eher in die falsche Richtung gezeigt.")
    with t3:
        cond_bear = (x[models] <= 100 - th).all(axis=1)
        r = (d["close"].shift(-h) / d["close"] - 1).reindex(x.index)
        g = r[cond_bear].dropna()
        st.metric(f"Alle ≤ {100 - th}: Anteil fallender Zeiträume", f"{(g < 0).mean():.0%}" if len(g) else "–",
                  f"{len(g)} Tage", delta_color="off")
        st.caption("Short-Trefferquote = Anteil, in dem der Index nach dem Horizont tiefer stand.")
    with st.expander("Wie wird gerechnet?"):
        st.markdown(
            "- Jeder Modell-Score stammt aus den bestehenden Seiten und nutzt je Tag nur damals bekannte Daten "
            "(COT ab dem Montag nach Veröffentlichung, Fundamental-ML aus dem Walk-forward-Test).\n"
            "- „k von n bullisch“ zählt, wie viele Modelle an einem Tag über der Schwelle lagen; Trefferquote und "
            "Ø Rendite über alle solchen Tage (überlappende Zeiträume – die Tage sind nicht unabhängig).\n"
            "- Trades: nicht überlappend, 0.05 % Kosten je Seite, kein Stop-Loss.\n"
            "- GEX fliesst noch nicht ein, weil es erst ab heute gesammelt wird.")
    st.caption(DISCLAIMER)


def _gex_now() -> str | None:
    p = get_settings().data_dir / "gex" / "model.json"
    try:
        m = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None
    cur = (m or {}).get("current", {})
    parts = [f"{k}: {v.get('regime')}e Gamma" for k, v in cur.items() if v.get("regime")]
    return ("GEX aktuell – " + ", ".join(parts) + " (noch ohne Historie, daher nicht im Test).") if parts else None


def count_chart(cnt: pd.DataFrame, n: int, base: float, mode: str) -> go.Figure:
    p = PALETTE[mode]
    c = cnt[cnt["k"] >= 0]
    colors = [STATUS["ok"] if v >= base else STATUS["NA"] for v in c["Trefferquote"].fillna(0)]
    fig = go.Figure(go.Bar(x=c["Bullische Modelle"], y=c["Trefferquote"], marker=dict(color=colors, cornerradius=4),
                           customdata=c["Tage"], text=[f"{v:.0%}" if pd.notna(v) else "" for v in c["Trefferquote"]],
                           textposition="outside",
                           hovertemplate="%{x}: %{y:.1%} (%{customdata} Tage)<extra></extra>"))
    fig.add_hline(y=base, line=dict(color=p["text2"], dash="dash"), annotation_text=f"Basis {base:.0%}",
                  annotation_position="top left")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return base_layout(fig, mode, "Trefferquote nach Anzahl bullischer Modelle (grün = besser als Basis)", height=360)


def history_chart(x: pd.DataFrame, models: list[str], th: float, mode: str) -> go.Figure:
    from plotly.subplots import make_subplots
    p = PALETTE[mode]
    k = (x[models] >= th).sum(axis=1)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=x.index, y=x["close"], name="SPY", line=dict(color=p["text"], width=1.6),
                             hovertemplate="SPY %{y:,.2f}<extra></extra>"), row=1, col=1)
    for a, b in _runs((k == len(models)).values, x.index.values):
        fig.add_vrect(x0=a, x1=b + np.timedelta64(1, "D"), fillcolor=STATUS["ok"], opacity=0.25, line_width=0,
                      row=1, col=1)
    fig.add_trace(go.Bar(x=x.index, y=k, name="Anzahl bullisch", marker_color=p["series"][0],
                         hovertemplate="%{y} bullisch<extra></extra>"), row=2, col=1)
    lo, hi = float(x["close"].min()), float(x["close"].max())
    ticks = [v for v in (10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 1000) if lo * 0.9 <= v <= hi * 1.1]
    fig.update_yaxes(type="log", title_text="SPY (log.)", tickvals=ticks, ticktext=[str(v) for v in ticks], row=1, col=1)
    fig.update_yaxes(title_text="bullisch", range=[0, len(models)], dtick=1, row=2, col=1)
    fig = base_layout(fig, mode, "Verlauf: grün hinterlegt = alle gewählten Modelle bullisch", height=520)
    fig.update_layout(showlegend=False, bargap=0)
    return fig
