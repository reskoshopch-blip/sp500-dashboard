"""Champagner-Index: Stimmung/Luxuskonsum über die Aktien der Champagnerhäuser, getestet gegen den S&P 500."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.cloud.pull import is_cloud
from src.components.ml_data import DISCLAIMER
from src.components.ui import plot, theme_mode
from src.config import get_settings
from src.visualization.charts import PALETTE, base_layout

PCT = st.column_config.NumberColumn(format="percent")


def _gdir():
    return get_settings().data_dir / "geo"


@st.cache_data(show_spinner=False)
def daily(version: float, data_dir: str) -> pd.DataFrame:
    df = pd.read_parquet(_gdir() / "champagne_daily.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def page() -> None:
    st.title("Champagner-Index")
    st.caption("Wird viel Champagner gekauft, sind Stimmung und Luxuskonsum hoch. Täglicher Ersatz: Korb der "
               "börsenkotierten Champagnerhäuser (Laurent-Perrier, Vranken-Pommery, Lanson-BCC, LVMH/Moët) im "
               "Vergleich zum europäischen Markt (STOXX 600), über 3 Monate.")
    p = _gdir() / "champagne_daily.parquet"
    j = _gdir() / "champagne.json"
    if not p.exists() or not j.exists():
        msg = "Noch nicht berechnet. Auf dem PC `champagne.bat` ausführen (läuft danach auch mit `pizza.bat` mit)."
        st.info(msg)
        return
    info = json.loads(j.read_text(encoding="utf-8"))
    df = daily(p.stat().st_mtime, str(get_settings().data_dir))
    sl = info.get("score_learned")
    c = st.columns(4)
    c[0].metric("Champagner-Score (roh)", f"{info['score']:.0f}", f"rel. Stärke 3 M. {info['rel_mom']:+.1%}",
                delta_color="off", help="Perzentil der relativen Stärke in den letzten 5 Jahren (100 = stärkster Wert).")
    c[1].metric("Signal für S&P 500", f"{sl:.0f}" if sl is not None else "–",
                ("bullisch" if sl >= 60 else "bärisch" if sl <= 40 else "neutral") if sl is not None else None,
                delta_color="off", help="Gelernte Richtung: hoch = bullisch für den S&P 500.")
    c[2].metric("Gelernte Richtung", info.get("direction") or "–")
    c[3].metric("Rangkorrelation (3 M.)", f"{info['ic']:+.2f}" if info.get("ic") is not None else "–",
                help="Spearman zwischen gelerntem Score und S&P-Rendite in 3 Monaten. |0.05| ist schon viel.")
    if info.get("missing"):
        st.caption("Keine Kurse für: " + ", ".join(info["missing"]))

    plot(chart(df, theme_mode()), "champ_chart")
    st.markdown("##### Test: S&P 500 nach Champagner-Signal")
    st.dataframe(pd.DataFrame(info["stats"]), hide_index=True, width="stretch",
                 column_config={"Trefferquote": PCT, "Ø Rendite": PCT})
    st.caption("Überlappende Zeiträume – die Tage sind nicht unabhängig. Nur wenn „bullisch“ klar über der Basis "
               "und „bärisch“ klar darunter liegt, taugt der Index als Zusatzfilter. Er ist auch im Modell-Konsens "
               "wählbar.")
    with st.expander("Wie wird gerechnet?"):
        st.markdown(
            "- Korb: gleichgewichtete Tagesrenditen der Häuser; Häuser kommen dazu, sobald sie Kurse haben.\n"
            "- Relative Stärke = Korb ÷ STOXX 600, Veränderung über 63 Handelstage; Score = Perzentil über 5 Jahre.\n"
            "- Richtung jedes Jahr neu gelernt, nur mit Jahren, deren 3-Monats-Ergebnis schon feststand.\n"
            "- Echte Absatzzahlen (Comité Champagne) gibt es nur jährlich – zu grob für tägliche Signale.")
    if not is_cloud():
        st.caption("Aktualisierung: `champagne.bat` oder automatisch mit `pizza.bat` / dem täglichen Update.")
    st.caption(DISCLAIMER)


def chart(df: pd.DataFrame, mode: str) -> go.Figure:
    from plotly.subplots import make_subplots
    p = PALETTE[mode]
    x = df.dropna(subset=["score"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=x.index, y=x["spy"], name="S&P 500 (SPY)", line=dict(color=p["text"], width=1.4),
                             hovertemplate="SPY %{y:,.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x.index, y=x["score"], name="Champagner-Score (roh)",
                             line=dict(color=p["series"][2], width=1),
                             hovertemplate="roh %{y:.0f}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x.index, y=x["score_learned"], name="Signal für S&P 500 (gelernt)",
                             line=dict(color=p["series"][0], width=1.6),
                             hovertemplate="Signal %{y:.0f}<extra></extra>"), row=2, col=1)
    for y in (30, 70):
        fig.add_hline(y=y, line=dict(color=p["text2"], dash="dash", width=1), row=2, col=1)
    fig.update_yaxes(type="log", title_text="SPY (log.)", row=1, col=1)
    fig.update_yaxes(title_text="Score", range=[0, 100], row=2, col=1)
    return base_layout(fig, mode, "S&P 500 und Champagner-Index", height=520)
