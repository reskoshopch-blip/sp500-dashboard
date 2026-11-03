"""Pentagon-Pizza-Index: Frühwarnung für Militäreinsätze (DOUGHCON + Auslastung der Pizzerien beim Pentagon)."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.cloud.pull import is_cloud
from src.components.ui import plot, theme_mode
from src.config import get_settings
from src.geo.pizza import LEVELS, SPIKE_PCT, load_history
from src.visualization.charts import PALETTE, STATUS, base_layout

COLORS = {5: STATUS["ok"], 4: STATUS["ok"], 3: STATUS["teilweise"], 2: STATUS["NA"], 1: STATUS["NA"]}


def _latest() -> dict | None:
    p = get_settings().data_dir / "geo" / "pizza_latest.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None


def page() -> None:
    st.title("Pizza-Index (Konflikt-Warnung)")
    st.caption("Vor grösseren Militäreinsätzen arbeitet das Pentagon nachts durch – die Pizzerien in der Nähe werden "
               "ungewöhnlich voll. Quelle: pizzint.watch (Google-Maps-Auslastung, DOUGHCON 5 = ruhig … 1 = maximal).")
    cur = _latest()
    if not cur:
        msg = "Noch keine Messung. Auf dem PC `pizza.bat` (oder `pizza_zeitplan.bat` für alle 30 Min.) ausführen."
        st.info(msg if is_cloud() else msg + " Danach `cloud_sync.bat`, damit es auch auf dem iPad erscheint.")
        return
    lvl = cur.get("level")
    t = pd.Timestamp(cur["taken_at"]).tz_convert("Europe/Zurich")
    c = st.columns(4)
    c[0].metric("DOUGHCON", f"{lvl}" if lvl else "–", LEVELS.get(lvl, "unbekannt"), delta_color="off")
    c[1].metric("Pizzerien mit Ausschlag", f"{cur.get('n_spikes', 0)} / {cur.get('n_places') or '–'}",
                f"≥ {SPIKE_PCT} % über normal", delta_color="off")
    avg = cur.get("avg_pct_above")
    c[2].metric("Ø Auslastung vs. normal", f"{avg:+.0f} %" if avg is not None else "–")
    c[3].metric("Gemessen", f"{t:%d.%m. %H:%M}", "Schweizer Zeit", delta_color="off")
    if lvl and lvl <= 2:
        st.error(f"DOUGHCON {lvl}: deutlich erhöhte Aktivität rund ums Pentagon – mögliche Militäraktion. "
                 "Geopolitisches Risiko: Öl, Gold und Volatilität beobachten.")
    elif lvl == 3 or (cur.get("n_spikes") or 0) >= 2:
        st.warning("Erhöhte Aktivität – noch kein klares Signal.")
    else:
        st.success("Ruhig – keine Auffälligkeiten.")
    if cur.get("alert"):
        st.caption(f"Hinweis der Quelle: {cur['alert']}")

    places = pd.DataFrame(cur.get("places") or [])
    if len(places):
        places = places.rename(columns={"name": "Pizzeria", "distance": "Distanz", "current": "Jetzt",
                                        "normal": "Normal", "pct_above": "vs. normal %", "is_open": "Offen"})
        st.dataframe(places, hide_index=True, width="stretch",
                     column_config={"vs. normal %": st.column_config.NumberColumn(format="%+.0f %%")})
    elif cur.get("source") == "html":
        st.caption("Einzelwerte der Pizzerien waren nicht abrufbar – nur die DOUGHCON-Stufe der Startseite.")

    h = load_history()
    if len(h) >= 2:
        plot(history_chart(h, theme_mode()), "pizza_hist")
    else:
        st.caption("Der Verlauf entsteht mit jeder Messung (alle 30 Minuten, solange der PC läuft).")
    with st.expander("Wie verlässlich ist das?"):
        st.markdown(
            "- Bekannte Beispiele: Golfkrieg 1991, Grenada 1983, Juni 2025 (Iran) – Anekdoten, keine Statistik.\n"
            "- PizzINT selbst bezeichnet sich als Unterhaltungs-/OSINT-Projekt; Korrelation ≠ Kausalität.\n"
            "- Es gibt keine öffentliche Historie zum Testen. Sobald genug eigene Messungen gesammelt sind, lässt sich "
            "prüfen, ob Ausschläge vor Ölpreis-/Gold-Sprüngen kamen.\n"
            "- Nur als Zusatz-Warnlampe nutzen, nicht als Handelssignal.")


def history_chart(h: pd.DataFrame, mode: str) -> go.Figure:
    from plotly.subplots import make_subplots
    p = PALETTE[mode]
    h = h.sort_values("taken_at")
    x = h["taken_at"].dt.tz_convert("Europe/Zurich")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=x, y=h["level"], mode="lines+markers", line=dict(shape="hv", color=p["series"][0]),
                             marker=dict(color=[COLORS.get(v, p["text2"]) for v in h["level"]], size=6),
                             name="DOUGHCON", hovertemplate="DOUGHCON %{y}<extra></extra>"), row=1, col=1)
    if h["avg_pct_above"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=h["avg_pct_above"], name="Ø vs. normal", line=dict(color=p["series"][1]),
                                 hovertemplate="%{y:+.0f} %<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=h["max_pct_above"], name="Max vs. normal",
                                 line=dict(color=p["series"][2], dash="dot"),
                                 hovertemplate="max %{y:+.0f} %<extra></extra>"), row=2, col=1)
        fig.add_hline(y=SPIKE_PCT, line=dict(color=STATUS["NA"], dash="dash"), row=2, col=1)
    fig.update_yaxes(title_text="DOUGHCON", range=[5.5, 0.5], dtick=1, row=1, col=1)
    fig.update_yaxes(title_text="% vs. normal", row=2, col=1)
    return base_layout(fig, mode, "Verlauf (oben: 1 = höchste Alarmstufe)", height=460)
