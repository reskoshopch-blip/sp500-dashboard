from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ml_data import DISCLAIMER, load_health, load_top, top_markets, top_version
from src.components.ui import plot, theme_mode
from src.ml.health import (COMP_BY_KEY, COMPONENTS, ENTRY_MODES, HORIZONS, LEVELS, base_stats, level, rule_trades,
                           trade_stats)
from src.ml.market_top import AMPEL
from src.ml.markets import MARKETS
from src.visualization.charts import bar_chart, date_lines, gauge, terminal_chart


def _fmt(v, kind: str) -> str:
    if v is None or pd.isna(v):
        return "–"
    if kind == "pct":
        return f"{v:+.1%}" if abs(v) < 5 else f"{v:,.0f}"
    if kind == "pp":
        return f"{v:+.2f} Pp."
    return f"{v:,.2f}"


def _light(score) -> str:
    if score is None or pd.isna(score):
        return "⚪ keine Daten"
    if score >= 60:
        return "🟢 " + level(score)
    if score >= 40:
        return "🟡 " + level(score)
    return "🔴 " + level(score)


PERIODS = {"6 Monate": 182, "1 Jahr": 365, "3 Jahre": 365 * 3, "5 Jahre": 365 * 5, "10 Jahre": 3652,
           "20 Jahre": 7305, "Alles": None}


def regime_tab(d: pd.DataFrame, key: str, mode: str) -> None:
    d = d.copy()
    d["sma50"] = d["close"].rolling(50, min_periods=50).mean()
    d["sma200"] = d["close"].rolling(200, min_periods=200).mean()
    d = d[d["H_kurz"].notna() | d["H_lang"].notna()]
    c1, c2 = st.columns([3, 2])
    per = c1.radio("Zeitraum", list(PERIODS), index=2, horizontal=True, key="rg_per")
    bg = c2.radio("Hintergrund nach", ["lang", "mittel", "kurz", "keiner"], horizontal=True, key="rg_bg",
                  format_func=lambda h: HORIZONS[h][0] if h in HORIZONS else "kein Hintergrund")
    with st.expander("Signal-Regel (Long) einstellen – die Dreiecke im Chart sind die historischen Trades", expanded=True):
        a, b, c, e, f = st.columns([3, 2, 2, 2, 2])
        entry = a.radio("Einstieg", list(ENTRY_MODES), format_func=ENTRY_MODES.get, key="rg_entry")
        kth = b.slider("Schwelle Kurzfristig", 5, 50, 20, 5, key="rg_k")
        lmin = c.slider("Langfristig mindestens", 0, 70, 40, 5, key="rg_l",
                        help="Filter: nur kaufen, wenn der langfristige Trend noch intakt ist. 0 = kein Filter.")
        hold = e.select_slider("Haltedauer (Handelstage)", [5, 10, 15, 21, 42, 63], 21, key="rg_hold")
        cost = f.number_input("Kosten je Seite (%)", 0.0, 1.0, 0.05, 0.05, key="rg_cost",
                              help="Spread/Kommission pro Ein- und Ausstieg. CFD-Finanzierung ist nicht enthalten.") / 100
        show_tr = st.toggle("Signale im Chart anzeigen", True, key="rg_show")
    trades = rule_trades(d, entry, kth, lmin, hold, cost)
    days = PERIODS[per]
    view = d if days is None else d[d.index >= d.index.max() - pd.Timedelta(days=days)]
    tv = trades[trades["Einstieg"] >= view.index.min()] if not trades.empty else trades
    plot(terminal_chart(view, mode, MARKETS[key]["short"], tv if show_tr else None,
                        None if bg == "keiner" else bg, {"sma50": "50-Tage-Schnitt", "sma200": "200-Tage-Schnitt"}),
         "rg_chart")
    st.caption("Oben: Kurs (Schlusskurse) mit 50/200-Tage-Schnitt; Hintergrund rot = Score < 40, grün = ≥ 60 "
               "(gelb/neutral ohne Färbung). Darunter die drei Scores als Farbband: dunkelrot = sehr schwach … "
               "dunkelgrün = sehr gesund. ▲ = Einstieg der Regel, ✕ = Ausstieg nach der Haltedauer, "
               "hohles Dreieck = Trade läuft noch.")

    st.markdown("##### Ergebnis der Regel")
    for label, tt in (("Ganze Historie", trades), (f"Nur im gewählten Zeitraum ({per})", tv)):
        s = trade_stats(tt)
        bs = base_stats(d, hold, start=tt["Einstieg"].min() if len(tt) else None)
        st.markdown(f"**{label}**")
        if s["n"] == 0:
            st.caption("Keine abgeschlossenen Trades.")
            continue
        m = st.columns(6)
        m[0].metric("Trades", s["n"])
        m[1].metric("Winrate", f"{s['winrate']:.0%}",
                    f"{(s['winrate'] - bs['winrate']) * 100:+.0f} Pp. vs. alle Tage" if bs else None,
                    help="Vergleich: Anteil positiver Zeiträume gleicher Länge, wenn man an irgendeinem Tag kauft.")
        m[2].metric("Profit-Faktor", "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}",
                    help="Summe Gewinne / Summe Verluste. Über 1.5 ist gut, um 1 heisst kein Vorteil.")
        m[3].metric("Ø Trade", f"{s['avg']:+.2%}", f"{(s['avg'] - bs['avg']) * 100:+.2f} Pp. vs. alle Tage" if bs else None)
        m[4].metric("Ø Gewinn / Ø Verlust", f"{s['avg_win']:+.1%} / {s['avg_loss']:+.1%}"
                    if pd.notna(s["avg_loss"]) else f"{s['avg_win']:+.1%} / –")
        m[5].metric("Schlechtester Trade", f"{s['worst']:+.1%}",
                    f"tiefster Zwischenstand {s['worst_dd']:.1%}", delta_color="off",
                    help="Tiefster Zwischenstand = grösster Buchverlust während eines Trades (für den Stop-Loss).")
    if not trades.empty:
        op = trades[trades["Status"] == "offen"]
        if len(op):
            r = op.iloc[-1]
            st.info(f"**Aktiver Trade** seit {r['Einstieg'].date()} (Kurz {r['Kurz']:.0f}, Lang {r['Lang']:.0f}) – "
                    f"bisher {r['Rendite']:+.1%}.")
        with st.expander(f"Alle {len(trades)} Trades"):
            show = trades.sort_values("Einstieg", ascending=False).copy()
            for col in ("Einstieg", "Ausstieg"):
                show[col] = show[col].dt.date
            st.dataframe(show, hide_index=True, width="stretch", column_config={
                "Rendite": st.column_config.NumberColumn(format="percent"),
                "Max. Rückgang": st.column_config.NumberColumn(format="percent"),
                "Kurs Einstieg": st.column_config.NumberColumn(format="%.0f"),
                "Kurs Ausstieg": st.column_config.NumberColumn(format="%.0f"),
                "Kurz": st.column_config.NumberColumn(format="%.0f"),
                "Lang": st.column_config.NumberColumn(format="%.0f")})
            yr = trades[trades["Status"] == "abgeschlossen"].assign(Jahr=lambda t: t["Einstieg"].dt.year)
            if len(yr):
                st.markdown("Nach Jahr")
                g = yr.groupby("Jahr")["Rendite"].agg(Trades="count", Summe="sum",
                                                      Winrate=lambda r: (r > 0).mean()).reset_index()
                st.dataframe(g, hide_index=True, width="stretch", column_config={
                    "Summe": st.column_config.NumberColumn("Summe Renditen", format="percent"),
                    "Winrate": st.column_config.NumberColumn(format="percent")})
    st.caption("Einstieg zum Schlusskurs des Signaltags, nur mit damals bekannten Scores. Trades überschneiden sich "
               "nicht. Kein Stop-Loss, keine Finanzierungskosten. Je mehr Schwellen man auf denselben Daten "
               "ausprobiert, desto eher findet man Zufallstreffer – die Standardregel (Kurz < 20, Lang ≥ 40, 21 Tage) "
               "wurde vorab festgelegt.")


def page() -> None:
    st.title("Markt-Terminal")
    keys = [k for k in top_markets() if load_health(top_version(k), k) is not None]
    if not keys:
        st.info("Noch keine Marktanalyse berechnet. Auf der Seite **Daten laden** unten „Markt-Analysen berechnen“ "
                "klicken.")
        return
    key = st.radio("Markt", keys, format_func=lambda k: MARKETS[k]["label"], horizontal=True, key="term_mkt")
    mode = theme_mode()
    h = load_health(top_version(key), key)
    top = load_top(top_version(key), key)
    d = h["daily"].set_index("date")
    d = d[d["close"].notna()]
    last = d.iloc[-1]
    prev = d.iloc[-6] if len(d) > 6 else None
    st.caption(f"Stand {last.name.date()} · {MARKETS[key]['label']} · Scores 0–100 = Perzentil jedes Indikators "
               "gegenüber den letzten 5 Jahren (nur Vergangenheit), Durchschnitt je Horizont. Pfeil = Veränderung "
               "gegenüber vor einer Woche.")

    cols = st.columns(4)
    for col, (hk, (name, sub, _)) in zip(cols, list(HORIZONS.items()) + [("gesamt", ("Gesamt", "Durchschnitt", 0))]):
        v = last.get(f"H_{hk}")
        ref = prev.get(f"H_{hk}") if prev is not None else None
        with col:
            plot(gauge(v, name, sub, mode, ref), f"g_{hk}")
            st.markdown(f"<div style='text-align:center;margin-top:-8px'>{_light(v)}</div>", unsafe_allow_html=True)

    # Kurzfazit
    parts = [f"{HORIZONS[hk][0]}: **{level(last.get(f'H_{hk}'))}**" for hk in HORIZONS]
    st.markdown(" · ".join(parts))
    if top is not None:
        td = top["daily"].set_index("date")
        tl = td.iloc[-1]
        signs = top["meta"].get("ampel_signs", list(AMPEL))
        active = [AMPEL[k] for k in signs if tl.get(k, 0) == 1]
        n_act = len(active)
        (st.warning if n_act >= 2 else st.info)(
            f"**Ampel {n_act}/{len(signs)}**: " + (", ".join(active) if active else "keine Warnzeichen aktiv")
            + " – Details auf der Seite „Top-Warnung“.")

    tab0, tab1, tab2, tab3, tab4 = st.tabs(["Chart mit Regimen", "Indikatoren", "Verlauf", "Einzelindikator",
                                            "Historischer Test"])
    with tab0:
        regime_tab(d, key, mode)
    with tab1:
        rows = []
        for c in COMPONENTS:
            raw = last.get(f"raw_{c.key}")
            sc = last.get(f"s_{c.key}")
            if pd.isna(raw) and pd.isna(sc):
                continue
            sc_prev = prev.get(f"s_{c.key}") if prev is not None else None
            rows.append({"Horizont": HORIZONS[c.horizon][0], "Bereich": c.category, "Indikator": c.label,
                         "Wert": _fmt(raw, c.fmt), "Score": sc,
                         "Δ Woche": (sc - sc_prev) if sc_prev is not None and pd.notna(sc) and pd.notna(sc_prev) else None,
                         "Signal": _light(sc), "Bedeutung": c.explain})
        tab = pd.DataFrame(rows)
        st.dataframe(tab, hide_index=True, width="stretch", height=min(900, 40 + 35 * len(tab)),
                     column_config={"Score": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
                                    "Δ Woche": st.column_config.NumberColumn(format="%+.0f")})
        missing = [c.label for c in COMPONENTS if f"s_{c.key}" not in d.columns or pd.isna(last.get(f"s_{c.key}"))]
        if missing:
            st.caption("Für diesen Markt nicht verfügbar: " + ", ".join(missing))
    with tab2:
        start = st.select_slider("Ab Jahr", options=sorted(set(d.index.year)), value=max(d.index.year.min(),
                                                                                         d.index.year.max() - 5),
                                 key="term_from")
        dd = d[d.index >= f"{start}-01-01"].reset_index()
        pf = date_lines(dd, "date", {"close": MARKETS[key]["short"]}, mode, f"{MARKETS[key]['short']} (log. Skala)",
                        kind="num", height=300)
        pf.update_yaxes(type="log")
        plot(pf, "term_px")
        plot(date_lines(dd, "date", {"H_kurz": "Kurzfristig", "H_mittel": "Mittelfristig", "H_lang": "Langfristig"},
                        mode, "Gesundheits-Scores (0–100)", kind="num", height=320, hline=50, hline_label="neutral"),
             "term_hist")
    with tab3:
        avail = [c.key for c in COMPONENTS if f"s_{c.key}" in d.columns]
        ck = st.selectbox("Indikator", avail, format_func=lambda k: f"{HORIZONS[COMP_BY_KEY[k].horizon][0]} · "
                                                                   f"{COMP_BY_KEY[k].label}", key="term_comp")
        c = COMP_BY_KEY[ck]
        st.caption(c.explain + (" · Hohe Werte sind günstig." if c.good_high else " · Hohe Werte sind ungünstig."))
        dd = d[d.index >= d.index.max() - pd.Timedelta(days=365 * 6)].reset_index()
        left, right = st.columns(2)
        with left:
            plot(date_lines(dd, "date", {f"raw_{ck}": c.label}, mode, "Rohwert", kind="pct" if c.fmt == "pct" else "num",
                            height=300), "term_raw")
        with right:
            plot(date_lines(dd, "date", {f"s_{ck}": "Score"}, mode, "Score (Perzentil, 0–100)", kind="num", height=300,
                            hline=50, hline_label="neutral"), "term_sc")
    with tab4:
        bt = h["bt"]
        if bt.empty:
            st.info("Zu wenig Historie für einen Test.")
        else:
            hk = st.radio("Horizont", list(HORIZONS), format_func=lambda k: f"{HORIZONS[k][0]} "
                                                                          f"({HORIZONS[k][2]} Handelstage)",
                          horizontal=True, key="term_bt_h")
            b = bt[bt["horizon"] == hk].copy()
            if b.empty:
                st.info("Für diesen Horizont gibt es zu wenig Daten.")
            else:
                ic = b["Rangkorrelation"].iloc[0]
                bb = b.dropna(subset=["Rückgang ≥ 10 % zwischendurch"])
                bb = bb[bb["Tage"] >= 100]
                weak, strong = bb.iloc[0], bb.iloc[-1]
                c1, c2, c3 = st.columns(3)
                c1.metric("Rückgangsrisiko: schwächster vs. stärkster Bereich",
                          f"{weak['Rückgang ≥ 10 % zwischendurch']:.0%} vs. {strong['Rückgang ≥ 10 % zwischendurch']:.0%}",
                          help="Wie oft fiel der Index zwischendurch um ≥ 10 %? Ein grosser Unterschied heisst: Der Score "
                               "trennt riskante von ruhigen Phasen.")
                c2.metric("Rangkorrelation Score ↔ Folgerendite", f"{ic:+.2f}",
                          help="Positiv = höherer Score ging mit besserer Folgerendite einher. Nahe 0 = der Score sagt die "
                               "Rendite nicht voraus (häufig: schwache Phasen erholen sich, sind aber schwankungsreicher).")
                c3.metric("Ø Folgerendite / Rückgang ≥ 10 % (alle Tage)",
                          f"{b['Basis Ø Rendite'].iloc[0]:+.1%} / {b['Basis ≥10%'].iloc[0]:.0%}")
                st.caption("Lesart: Der Gesundheitsindex misst vor allem das **Risiko** (wie wahrscheinlich ein grösserer "
                           "Rückgang ist), weniger die Rendite. Schwache Phasen waren oft auch Einstiegsgelegenheiten – "
                           "aber mit deutlich höherem Rückschlagrisiko.")
                plot(bar_chart(b.dropna(subset=["Ø Rendite"]), "Bereich", "Ø Rendite", None, mode,
                               f"Ø Rendite der nächsten {HORIZONS[hk][2]} Handelstage je Score-Bereich", kind="pct"),
                     "term_bt")
                show = b[["Bereich", "Tage", "Ø Rendite", "Median Rendite", "Anteil positiv",
                          "Rückgang ≥ 10 % zwischendurch"]].copy()
                for col in ("Ø Rendite", "Median Rendite"):
                    show[col] = show[col].map(lambda v: f"{v:+.1%}" if pd.notna(v) else "–")
                for col in ("Anteil positiv", "Rückgang ≥ 10 % zwischendurch"):
                    show[col] = show[col].map(lambda v: f"{v:.0%}" if pd.notna(v) else "–")
                st.dataframe(show, hide_index=True, width="stretch")
                now = last.get(f"H_{hk}")
                if pd.notna(now):
                    cur = next(lab for lo, hi, lab in LEVELS if lo <= now < hi) if now < 100 else "sehr gesund"
                    st.caption(f"Heute: Score {now:.0f} ({cur}). Die Zeile dieses Bereichs zeigt, was historisch danach "
                               "passierte. Überlappende Zeiträume – die Tage sind nicht unabhängig.")
    with st.expander("Wie wird der Index berechnet?"):
        st.markdown(
            "- Jeder Indikator wird mit seiner eigenen Geschichte der letzten 5 Jahre verglichen (Perzentil). "
            "50 = typisch, 90 = besser als an 90 % der Tage. So lassen sich ganz verschiedene Grössen (Prozent, "
            "Anteile, Zinsen) zusammenfassen.\n"
            "- Kurzfristig: Trend 20 Tage, 10-Tage-Momentum, Breite 20 Tage, McClellan, Tages-Divergenzen, "
            "Volatilität, VIX-Terminstruktur.\n"
            "- Mittelfristig: Trend 50 Tage, 3-Monats-Momentum, Breite 50 Tage, Advance-Decline-Linie, Hochs/Tiefs, "
            "gleichgewichtet vs. Index (RSP/SPY), Small vs. Large Caps, zyklisch vs. defensiv, Kreditmarkt, "
            "(Japan: USD/JPY).\n"
            "- Langfristig: Trend 200 Tage, 12-Monats-Momentum, Breite 200 Tage, Anteil Aktien ≥ 20 % unter Hoch, "
            "Abstand zum Hoch, Kreditmarkt 12 Monate, Zinskurve, Kupfer/Gold.\n"
            "- US-spezifische Vergleiche (RSP, IWM, XLY/XLP, VIX) nur beim S&P 500. Marktbreite aus heutigen "
            "Indexmitgliedern (frühe Jahre leicht verzerrt). Zinsen und Rohstoffe von Yahoo (Futures-Fortschreibung).")
    st.caption(DISCLAIMER)
