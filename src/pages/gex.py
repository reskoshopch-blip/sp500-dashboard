"""Gamma Exposure (GEX): aktuelles Profil, Levels, Regime und die GEX-KI, die mit jedem Tag dazulernt."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.components.ml_data import DISCLAIMER
from src.components.ui import plot, theme_mode
from src.config import get_settings
from src.gex.client import api_key, symbols
from src.gex.features import profile
from src.gex.learn import FEAT_LABEL, MIN_SAMPLES
from src.pages.data_admin import start_module
from src.visualization.charts import PALETTE, STATUS, base_layout, date_lines


def gdir():
    return get_settings().data_dir / "gex"


def _read(name):
    p = gdir() / name
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None


def latest_raw(sym: str) -> dict | None:
    d = gdir() / "raw" / sym
    if not d.exists():
        return None
    files = sorted(f for f in d.glob("*.json") if not f.stem.endswith("_zeta"))
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


@st.cache_data(show_spinner=False)
def dataset(v: float) -> pd.DataFrame:
    p = gdir() / "dataset.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def fmt(v, d=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:,.{d}f}".replace(",", "'")


def page() -> None:
    st.title("Gamma Exposure (GEX)")
    st.caption("Daten: AnalyticsFX-API (Optionsketten → Gamma der Market Maker je Strike). Die API liefert nur den "
               "aktuellen Stand – das Dashboard speichert deshalb jeden Abruf und die GEX-KI lernt mit jedem "
               "abgeschlossenen Tag dazu.")
    from src.cloud.pull import is_cloud
    cloud = is_cloud()
    if not api_key() and not cloud:
        st.warning("Noch kein API-Schlüssel eingetragen.")
        st.markdown("1. Im Projektordner die Datei **.env** mit Notepad öffnen.\n"
                    "2. Eine neue Zeile hinzufügen: `ANALYTICSFX_API_KEY=afx_live_...` (deinen Schlüssel aus dem "
                    "AnalyticsFX-Dashboard einfügen).\n"
                    "3. Optional: `GEX_SYMBOLS=SPY,QQQ,SPX` (Standard).\n"
                    "4. Speichern, das Dashboard neu starten (`start.bat nodata`).\n\n"
                    "Der Schlüssel bleibt nur in .env auf deinem PC – er wird nicht angezeigt oder gespeichert.")
        st.caption(DISCLAIMER)
        return
    stt = _read("status.json") or {}
    c1, c2 = st.columns([3, 2])
    with c1:
        if cloud:
            st.caption("☁️ Cloud-Ansicht – neue GEX-Abrufe laufen auf dem PC (gex_zeitplan.bat).")
        else:
            force = st.checkbox("Auch abrufen, wenn der letzte Abruf weniger als 6 Stunden her ist", key="gex_force")
            n = len(symbols())
            if st.button(f"GEX jetzt abrufen ({', '.join(symbols())} – ca. {n * 2 + 2} Credits)", type="primary"):
                start_module("src.gex.collect", ["--force"] if force else [], "gex.log")
                st.info("Abruf gestartet – in ca. 30 Sekunden die Seite neu laden (F5).")
    with c2:
        if stt:
            st.caption(f"Letzter Lauf: {stt.get('updated_at', '')} UTC · abgerufen: {', '.join(stt.get('done', [])) or '–'}"
                       f" · Credits übrig: {stt.get('credits_remaining', '–')}")
            for k, v in (stt.get("errors") or {}).items():
                st.error(f"{k}: {v}")
    model = _read("model.json")
    if not model or not model.get("current"):
        st.info("Noch keine Daten. Oben „GEX jetzt abrufen“ klicken. Für das tägliche Lernen am besten einmal "
                "**gex_zeitplan.bat** ausführen – dann sammelt Windows jeden Börsentag um 22:15 automatisch.")
        st.caption(DISCLAIMER)
        return
    cur = model["current"]
    sym = st.radio("Symbol", list(cur), horizontal=True, key="gex_sym")
    c = cur[sym]
    mode = theme_mode()
    spot = c.get("spot")
    reg = c.get("regime")
    st.caption(f"Stand {pd.Timestamp(c['taken_at']).tz_convert('Europe/Zurich'):%d.%m.%Y %H:%M} (Schweizer Zeit)")
    m = st.columns(5)
    m[0].metric("Kurs (Spot)", fmt(spot))
    m[1].metric("Gamma-Flip", fmt(c.get("flip")), _dist(spot, c.get("flip")), delta_color="off")
    m[2].metric("Call-Wall", fmt(c.get("call_wall")), _dist(spot, c.get("call_wall")), delta_color="off")
    m[3].metric("Put-Wall", fmt(c.get("put_wall")), _dist(spot, c.get("put_wall")), delta_color="off")
    m[4].metric("Max Gamma", fmt(c.get("max_gamma")), _dist(spot, c.get("max_gamma")), delta_color="off")
    if reg == "positiv":
        st.success("**Positive Gamma** – Kurs über dem Flip. Market Maker sichern *gegen* die Bewegung ab: Rückgänge "
                   "werden gekauft, Anstiege verkauft. Typisch: ruhigere Tage, Mean-Reversion, Kurs zieht Richtung "
                   "Max Gamma, Call-Wall bremst oben, Put-Wall stützt unten.")
    elif reg == "negativ":
        st.error("**Negative Gamma** – Kurs unter dem Flip. Market Maker sichern *mit* der Bewegung ab: Abgaben "
                 "verstärken Abgaben, Anstiege werden gejagt. Typisch: grössere, trendige Tage und höhere "
                 "Volatilität. Ein Zurück über den Flip beruhigt den Markt oft schnell.")

    left, right = st.columns([3, 2])
    with left:
        raw = latest_raw(sym)
        if raw:
            plot(profile_chart(profile(raw), c, mode), f"gex_prof_{sym}")
    with right:
        st.markdown("##### Nächster Handelstag")
        rv = c.get("rv20")
        rows = [{"Modell": "Regel (Marktlogik)", "Erwartete Bewegung": c.get("rule_exp_move"),
                 "Chance Plus": np.nan, "Zwischen Walls": np.nan}]
        if c.get("ml_exp_move") is not None or c.get("ml_up") is not None:
            rows.append({"Modell": "GEX-KI (gelernt)", "Erwartete Bewegung": c.get("ml_exp_move"),
                         "Chance Plus": c.get("ml_up"), "Zwischen Walls": c.get("ml_inside")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Erwartete Bewegung": st.column_config.NumberColumn(format="percent", help="± typische Tagesbewegung"),
            "Chance Plus": st.column_config.NumberColumn(format="percent"),
            "Zwischen Walls": st.column_config.NumberColumn(format="percent",
                                                            help="Wahrscheinlichkeit, dass der nächste Schluss "
                                                                 "zwischen Put- und Call-Wall liegt")})
        if rv and spot:
            em = c.get("ml_exp_move") or c.get("rule_exp_move")
            if em:
                st.caption(f"Erwartete Spanne bis Schluss: ca. {fmt(spot * (1 - em))} – {fmt(spot * (1 + em))} "
                           f"(normale Tagesschwankung {rv:.2%}).")
        done, need = model.get("n_done", 0), model.get("min_samples", MIN_SAMPLES)
        st.markdown("##### Lernstand der KI")
        st.progress(min(done / need, 1.0), text=f"{done} abgeschlossene Tage (alle Symbole) – lernt ab {need}")
        ev = model.get("eval", {})
        if done < need:
            st.caption("Bis genug Tage gesammelt sind, gilt das Regelmodell. Jeder Abruf nach Börsenschluss ist ein "
                       "neuer Lerntag – mit 3 Symbolen reichen gut zwei Wochen.")
        elif ev:
            txt = []
            if "ml_move_corr" in ev:
                txt.append(f"Bewegungsgrösse: KI {ev['ml_move_corr']:+.2f} vs. Regel {ev.get('ml_move_corr_rule_same', 0):+.2f} "
                           "(Rangkorrelation mit der echten Bewegung, nur echte Vorhersagen)")
            if "ml_up_auc" in ev:
                txt.append(f"Richtung: AUC {ev['ml_up_auc']:.2f} (0.5 = Zufall)")
            if "ml_inside_auc" in ev:
                txt.append(f"Range zwischen Walls: Trefferquote {ev['ml_inside_hit']:.0%} (Basis {ev['inside_base']:.0%})")
            st.caption(" · ".join(txt))
        drv = c.get("move_drivers")
        if drv:
            st.caption("Was die KI für die Bewegungsgrösse gelernt hat (+ = grössere Bewegung): " + ", ".join(
                f"{FEAT_LABEL.get(x['key'], x['key'])} {x['coef']:+.2f}" for x in drv[:5]))

    reg_stats = model.get("regimes") or []
    if reg_stats:
        st.markdown("##### Was die gesammelten Tage zeigen")
        st.dataframe(pd.DataFrame(reg_stats), hide_index=True, width="stretch", column_config={
            "Bewegung vs. normal": st.column_config.NumberColumn(format="%.2fx", help="Median |Bewegung| / normale "
                                                                 "Tagesschwankung. Unter 1 = ruhiger als normal."),
            **{k: st.column_config.NumberColumn(format="percent") for k in ("Ø |Bewegung|", "Anteil Plus-Tage",
                                                                             "Schluss zwischen den Walls",
                                                                             "Richtung Max Gamma")}})
    d = dataset(_mtime())
    if not d.empty:
        g = d[d["symbol"] == sym].copy()
        if len(g) >= 2:
            g["date"] = pd.to_datetime(g["taken_at"]).dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
            plot(date_lines(g, "date", {"spot": "Kurs", "flip": "Gamma-Flip", "call_wall": "Call-Wall",
                                        "put_wall": "Put-Wall"}, mode, f"{sym}: Kurs und GEX-Levels über die Zeit",
                            kind="num", height=320), f"gex_hist_{sym}")
    with st.expander("Wie lernt die GEX-KI?"):
        st.markdown(METHOD)
    st.caption(DISCLAIMER)


def _mtime() -> float:
    p = gdir() / "dataset.parquet"
    return p.stat().st_mtime if p.exists() else 0.0


def _dist(spot, lvl):
    if not spot or lvl is None or not np.isfinite(lvl):
        return None
    return f"{lvl / spot - 1:+.2%} vom Kurs"


def profile_chart(pr: pd.DataFrame, c: dict, mode: str) -> go.Figure:
    p = PALETTE[mode]
    spot = c.get("spot") or pr["strike"].median()
    v = pr[(pr["strike"] >= spot * 0.92) & (pr["strike"] <= spot * 1.08)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=v["strike"], y=v["call"], name="Call-Gamma", marker_color=STATUS["ok"], opacity=0.8,
                         hovertemplate="Strike %{x}: %{y:,.0f}<extra>Calls</extra>"))
    fig.add_trace(go.Bar(x=v["strike"], y=v["put"], name="Put-Gamma", marker_color=STATUS["NA"], opacity=0.8,
                         hovertemplate="Strike %{x}: %{y:,.0f}<extra>Puts</extra>"))
    fig.add_trace(go.Scatter(x=v["strike"], y=v["total"], name="Netto", mode="lines",
                             line=dict(color=p["text"], width=2), hovertemplate="Netto %{y:,.0f}<extra></extra>"))
    lo = float(np.nanmin(np.minimum(v["put"].fillna(0), v["total"].fillna(0)))) if len(v) else 0
    hi = float(np.nanmax(np.maximum(v["call"].fillna(0), v["total"].fillna(0)))) if len(v) else 1
    for lvl, nm, colr, dash in ((spot, "Kurs", p["series"][0], "solid"), (c.get("flip"), "Gamma-Flip", p["series"][3], "dash"),
                                (c.get("call_wall"), "Call-Wall", STATUS["ok"], "dot"),
                                (c.get("put_wall"), "Put-Wall", STATUS["NA"], "dot"),
                                (c.get("max_gamma"), "Max Gamma", p["series"][6], "dashdot")):
        if lvl is not None and np.isfinite(lvl):
            fig.add_trace(go.Scatter(x=[lvl, lvl], y=[lo * 1.05, hi * 1.05], mode="lines", name=f"{nm} {lvl:,.0f}",
                                     line=dict(color=colr, width=2.5, dash=dash),
                                     hovertemplate=f"{nm}: {lvl:,.2f}<extra></extra>"))
    fig.update_layout(barmode="relative")
    fig.update_xaxes(title_text="Strike")
    fig.update_yaxes(title_text="Gamma Exposure")
    return base_layout(fig, mode, "GEX-Profil (±8 % um den Kurs)", height=440)


METHOD = f"""
**Daten:** Bei jedem Abruf speichert das Dashboard die komplette Antwort der API (Profil je Strike, Flip, Max Gamma)
und berechnet daraus Merkmale: Regime (über/unter Flip), Abstände zu Flip, Walls und Max Gamma, Netto-Gamma-Anteil,
Konzentration um den Kurs, Lage zwischen den Walls, Zeta-Regime (SPY/QQQ), normale Tagesschwankung und 5-Tage-Trend.

**Lernziel:** Mit dem nächsten Börsenschluss nach dem Abruf (Yahoo-Kurse): wie gross war die Bewegung im Verhältnis
zur normalen Schwankung, ging es hoch oder runter, schloss der Kurs zwischen Put- und Call-Wall?

**Ohne Blick in die Zukunft:** Jeder Tag wird mit einem Modell vorhergesagt, das nur Tage kannte, deren Ergebnis
schon feststand. Erst diese echten Vorhersagen ergeben die angezeigte Treffsicherheit.

**Modelle:** Regelmodell (positive Gamma 0.8 × normale Bewegung, negative 1.25 ×) und ab {MIN_SAMPLES}
abgeschlossenen Tagen regularisierte Regressionen über alle Symbole (Ridge für die Bewegungsgrösse, logistische
Regression für Richtung und Range). Einfach gehalten, weil die Historie anfangs kurz ist – komplexe Modelle würden
sich hier nur Zufall merken.

**Kosten:** GEX 2 Credits, Zeta 1 Credit je Symbol. Täglich SPY, QQQ, SPX ≈ 8 Credits, rund 180 pro Monat.
"""
