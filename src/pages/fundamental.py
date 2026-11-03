"""Fundamental-Terminal: Makro, Notenbanken, Liquidität, Konjunktur, Gewinne, Bewertung und Positionierung –
mit gelernten Richtungen/Gewichten, ML-Prognose und der Frage „Was ist eingepreist?“."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.components.fund_data import fund_version, load_fund
from src.components.ml_data import DISCLAIMER, load_health, top_version
from src.components.ui import plot, theme_mode
from src.fundamental.catalog import CATEGORIES, IND_BY_KEY
from src.fundamental.learn import CAT_COL, DIV_BUCKETS, level
from src.ml.health import rule_trades, trade_stats
from src.ml.markets import MARKETS
import plotly.graph_objects as go

from src.visualization.charts import PALETTE, bar_chart, base_layout, date_lines, driver_chart, gauge, terminal_chart

PERIODS = {"1 Jahr": 365, "3 Jahre": 365 * 3, "5 Jahre": 365 * 5, "10 Jahre": 3652, "20 Jahre": 7305, "Alles": None}


def fmt(v, kind: str) -> str:
    if v is None or pd.isna(v):
        return "–"
    if kind == "pct":
        return f"{v:+.1%}"
    if kind == "pp":
        return f"{v:+.2f} Pp."
    if kind == "x":
        return f"{v:.1f}x"
    return f"{v:,.2f}"


def light(v) -> str:
    if v is None or pd.isna(v):
        return "⚪ keine Daten"
    return ("🟢 " if v >= 60 else "🟡 " if v >= 40 else "🔴 ") + level(v)


def div_text(v) -> str:
    if v is None or pd.isna(v):
        return "keine Daten"
    for lo, hi, name in DIV_BUCKETS:
        if lo <= v < hi:
            return name
    return ""


def page() -> None:
    st.title("Fundamental-Terminal")
    data = load_fund(fund_version())
    if not data or not data["markets"]:
        st.info("Noch keine Fundamentalanalyse berechnet. Auf der Seite **Daten laden** unten "
                "„Fundamentalanalyse berechnen“ klicken (lädt Makrodaten von FRED, COT-Daten der CFTC und wertet die "
                "SEC-Gewinne aus; erster Lauf ca. 3–8 Minuten).")
        return
    keys = list(data["markets"])
    key = st.radio("Markt", keys, format_func=lambda k: MARKETS[k]["label"], horizontal=True, key="fund_mkt")
    mode = theme_mode()
    M = data["markets"][key]
    meta = M["meta"]
    d = M["daily"].set_index("date")
    d = d[d["close"].notna()]
    last = d.iloc[-1]
    prev = d.iloc[-64] if len(d) > 64 else d.iloc[0]
    ml = meta.get("ml", {})
    st.caption(f"Stand {last.name.date()} · {MARKETS[key]['label']} · Scores 0–100: 50 = neutral, hoch = bullisch. "
               "Richtung und Gewicht jedes Indikators werden jedes Jahr aus der Vergangenheit neu gelernt. "
               "Pfeil = Veränderung über 3 Monate (bei den ML-Tachos: Abstand zur historischen Basisrate).")

    # ------------------------------------------------------------------ Kopfzeile
    c = st.columns(4)
    with c[0]:
        plot(gauge(last.get("F_gesamt"), "Fundamental gesamt", "gelernter Score", mode, prev.get("F_gesamt")), "fg")
        st.markdown(f"<div style='text-align:center;margin-top:-8px'>{light(last.get('F_gesamt'))}</div>",
                    unsafe_allow_html=True)
    for col, (h, title) in zip(c[1:3], (("3M", "Chance 3 Monate"), ("12M", "Chance 12 Monate"))):
        r = ml.get(h, {})
        p = r.get("p_now")
        with col:
            plot(gauge(None if p is None else p * 100, title, "ML: Wahrsch. positive Rendite", mode,
                       None if r.get("base") is None else r["base"] * 100), f"fp{h}")
            auc = r.get("auc")
            q = ("🟢 brauchbar" if auc and auc >= 0.58 else "🟡 schwach" if auc and auc >= 0.53 else "🔴 kein Vorteil")
            st.markdown(f"<div style='text-align:center;margin-top:-8px'>Test-AUC {auc:.2f} · {q}</div>" if auc else
                        "<div style='text-align:center'>noch kein Test</div>", unsafe_allow_html=True)
    with c[3]:
        dv = last.get("divergence")
        plot(gauge(None if pd.isna(dv) else float(np.clip(50 + dv / 2, 0, 100)), "Eingepreist?",
                   "50 = im Einklang", mode), "fdiv")
        st.markdown(f"<div style='text-align:center;margin-top:-8px'>{div_text(dv)}</div>", unsafe_allow_html=True)

    # Kategorien
    cats = [(cat, col) for cat, col in CAT_COL.items() if col in d.columns and d[col].notna().any()]
    rows = [{"Bereich": cat, "Score": last.get(col), "Δ 3 M.": last.get(col) - prev.get(col)
             if pd.notna(last.get(col)) and pd.notna(prev.get(col)) else None, "Signal": light(last.get(col))}
            for cat, col in cats]
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Score": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
            "Δ 3 M.": st.column_config.NumberColumn(format="%+.0f")})
    with right:
        st.markdown(summary(last, prev, ml, key, d))

    tabs = st.tabs(["Indikatoren", "Chart & Verlauf", "Prognose & Treiber", "Gelernte Wirkung", "Eingepreist?",
                    "Fundamental + Technik", "Historischer Test"])
    with tabs[0]:
        indicator_tab(d, last, prev, meta)
    with tabs[1]:
        chart_tab(d, key, mode, cats)
    with tabs[2]:
        forecast_tab(ml, mode)
    with tabs[3]:
        effects_tab(M, d, mode)
    with tabs[4]:
        priced_tab(d, M, key, mode, meta, data)
    with tabs[5]:
        combo_tab(d, key)
    with tabs[6]:
        test_tab(M, last, mode)
    with st.expander("Wie funktioniert das Fundamental-Terminal?"):
        st.markdown(METHOD)
    if meta.get("missing"):
        st.caption("Für diesen Markt nicht verfügbar (Quelle fehlt oder zu kurze Historie): " + ", ".join(meta["missing"]))
    gm = data["meta"]
    if gm.get("fred_failed"):
        st.caption("Nicht geladene FRED-Reihen: " + ", ".join(gm["fred_failed"]))
    st.caption(DISCLAIMER)


def summary(last, prev, ml, key, d) -> str:
    g, gp = last.get("F_gesamt"), prev.get("F_gesamt")
    lines = []
    if pd.notna(g):
        trend = ""
        if pd.notna(gp):
            delta = g - gp
            trend = (" und **verbessert sich**" if delta >= 5 else " und **verschlechtert sich**" if delta <= -5
                     else ", Tendenz seitwärts")
            trend += f" ({delta:+.0f} in 3 M.)"
        lines.append(f"**Fundamental:** {level(g)} ({g:.0f}){trend}.")
    cats = [(c, last.get(col)) for c, col in CAT_COL.items() if pd.notna(last.get(col))]
    if cats:
        best = max(cats, key=lambda x: x[1])
        worst = min(cats, key=lambda x: x[1])
        lines.append(f"Stärkster Bereich: **{best[0]}** ({best[1]:.0f}), schwächster: **{worst[0]}** ({worst[1]:.0f}).")
    r3, r12 = ml.get("3M", {}), ml.get("12M", {})
    if r3.get("p_now") is not None:
        lines.append(f"**ML:** {r3['p_now']:.0%} Chance auf Plus in 3 Monaten (historisch {r3.get('base', 0):.0%}), "
                     f"{r12.get('p_now', float('nan')):.0%} in 12 Monaten (historisch {r12.get('base', 0):.0%}).")
    dv = last.get("divergence")
    if pd.notna(dv):
        lines.append(f"**Eingepreist:** {div_text(dv)} (Kursmomentum {last.get('mom_pct'):.0f} vs. Fundamental "
                     f"{g:.0f}).")
    return "\n\n".join(lines)


def indicator_tab(d, last, prev, meta) -> None:
    rows = []
    for k in meta.get("available", []):
        ind = IND_BY_KEY.get(k)
        if ind is None:
            continue
        o, w = last.get(f"o_{k}"), last.get(f"w_{k}")
        sc = last.get(f"s_{k}")
        if pd.isna(sc) and pd.isna(last.get(f"raw_{k}")):
            continue
        sign = None
        if pd.notna(o) and pd.notna(sc):
            sign = "hoch = gut" if abs(o - sc) < 1e-9 else "tief = gut"
        rows.append({"Bereich": ind.category, "Indikator": ind.label, "Wert": fmt(last.get(f"raw_{k}"), ind.fmt),
                     "Perzentil": sc, "Gelernt": sign or "–", "Gewicht": w if pd.notna(w) else 0.0,
                     "Bullisch-Score": o, "Δ 3 M.": (o - prev.get(f"o_{k}")) if pd.notna(o) and pd.notna(prev.get(f"o_{k}"))
                     else None, "Signal": light(o) if pd.notna(w) and w > 0 else "⚪ ohne Gewicht",
                     "Bedeutung": ind.explain})
    tab = pd.DataFrame(rows)
    if tab.empty:
        st.info("Keine Indikatoren verfügbar.")
        return
    tab["_c"] = tab["Bereich"].map({c: i for i, c in enumerate(CATEGORIES)})
    tab = tab.sort_values(["_c", "Gewicht"], ascending=[True, False]).drop(columns="_c")
    st.dataframe(tab, hide_index=True, width="stretch", height=min(1100, 40 + 35 * len(tab)), column_config={
        "Perzentil": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100,
                                                     help="Wert im Vergleich zu den letzten 10 Jahren"),
        "Bullisch-Score": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100,
                                                          help="Perzentil in die gelernte Richtung gedreht"),
        "Gewicht": st.column_config.NumberColumn(format="%.1f", help="Statistische Stärke (t-Wert, max. 3). "
                                                 "0.5 = nur ökonomische Grundrichtung, 0 = ohne Einfluss"),
        "Δ 3 M.": st.column_config.NumberColumn(format="%+.0f")})
    st.caption("„Gelernt“: welche Richtung historisch günstig war (aus abgeschlossenen 6-Monats-Zeiträumen vor dem "
               "jeweiligen Jahr). Ein Indikator kann gegen die Lehrbuchmeinung wirken – z. B. tiefe Konsumentenstimmung "
               "als Kaufgelegenheit. Sortiert nach Bereich und Gewicht.")


def chart_tab(d, key, mode, cats) -> None:
    c1, c2 = st.columns([3, 2])
    per = c1.radio("Zeitraum", list(PERIODS), index=2, horizontal=True, key="fund_per")
    opts = ["F_gesamt"] + [col for _, col in cats]
    names = {"F_gesamt": "Fundamental gesamt", **{col: cat for cat, col in cats}}
    bg = c2.selectbox("Hintergrund nach", opts, format_func=names.get, key="fund_bg")
    days = PERIODS[per]
    v = d if days is None else d[d.index >= d.index.max() - pd.Timedelta(days=days)]
    v = v[v["F_gesamt"].notna()] if v["F_gesamt"].notna().any() else v
    v = v.copy()
    v["sma200"] = d["close"].rolling(200, min_periods=200).mean().reindex(v.index)
    bands = {"Gesamt": "F_gesamt", **{cat.split(" (")[0]: col for cat, col in cats}}
    if "p_3M" in v and v["p_3M"].notna().any():
        v["p3"] = v["p_3M"] * 100
        bands["ML 3 Monate"] = "p3"
    plot(terminal_chart(v, mode, MARKETS[key]["short"], None, bg, {"sma200": "200-Tage-Schnitt"}, bands=bands,
                        bg_label=names[bg]), "fund_chart")
    st.caption("Hintergrund rot = Score < 40, grün = ≥ 60. Die Bänder darunter zeigen jeden Bereich von dunkelrot "
               "(sehr bärisch) bis dunkelgrün (sehr bullisch). „ML 3 Monate“ = Wahrscheinlichkeit einer positiven "
               "3-Monats-Rendite in %, jeweils aus einem Modell, das nur die Jahre davor kannte.")
    hist = d[d.index >= d.index.max() - pd.Timedelta(days=PERIODS[per] or 365 * 40)].reset_index()
    plot(date_lines(hist, "date", {"F_gesamt": "Fundamental gesamt", "mom_pct": "Kursmomentum (Perzentil)"}, mode,
                    "Fundamentaldaten vs. Kurs", kind="num", height=300, hline=50, hline_label="neutral"), "fund_vs")


def forecast_tab(ml, mode) -> None:
    for h, title in (("3M", "3 Monate"), ("12M", "12 Monate")):
        r = ml.get(h, {})
        st.markdown(f"##### Prognose {title}")
        if "auc" not in r:
            st.info("Zu wenig Historie für einen Test.")
            continue
        c = st.columns(4)
        c[0].metric("Heute", f"{r.get('p_now', float('nan')):.0%}", f"{(r.get('p_now', 0) - r['base']) * 100:+.0f} Pp. "
                    "vs. Basis", help="Wahrscheinlichkeit einer positiven Rendite, kalibriert an den echten Testergebnissen "
                    f"(Modell roh: {r.get('p_raw', float('nan')):.0%}). Ohne Vorhersagekraft landet sie bei der Basisrate.")
        c[1].metric("Test-AUC (alle Jahre)", f"{r['auc']:.2f}", help="0.5 = Zufall, 0.6 = brauchbar, 0.7 = stark. "
                    "Nur echte Vorhersagen: jedes Jahr mit einem Modell, das die Zukunft nicht kannte.")
        c[2].metric("Trefferquote Richtung", f"{r['hit']:.0%}", f"Basis {r['base']:.0%} positiv", delta_color="off")
        c[3].metric("Testzeitraum", f"{len(r.get('per_year', []))} Jahre", f"{r['n']} Wochen", delta_color="off")
        if r["auc"] < 0.53:
            st.warning("Für diesen Horizont hat das Modell historisch **keinen verlässlichen Vorteil** gezeigt – die "
                       "Prognose nur als Zusatzinfo lesen.")
        left, right = st.columns(2)
        with left:
            if r.get("drivers"):
                dr = pd.DataFrame(r["drivers"])
                dr["label"] = dr["key"].map(_label)
                dr["value_txt"] = ""
                plot(driver_chart(dr, mode, "Was die heutige Prognose treibt (Pp.)"), f"drv{h}")
                st.caption("Balken = um wie viele Prozentpunkte die Prognose sinken/steigen würde, wenn dieser Wert "
                           "durchschnittlich wäre.")
        with right:
            if r.get("calibration"):
                cal = pd.DataFrame(r["calibration"])
                st.markdown("Was nach ähnlichen Prognosen passierte")
                st.dataframe(cal, hide_index=True, width="stretch", column_config={
                    "Ø Prognose": st.column_config.NumberColumn(format="percent"),
                    "Trefferquote": st.column_config.NumberColumn(format="percent"),
                    "Ø Rendite": st.column_config.NumberColumn(format="percent")})
            py = pd.DataFrame(r.get("per_year", []))
            if not py.empty:
                with st.expander("Test Jahr für Jahr"):
                    st.dataframe(py, hide_index=True, width="stretch", column_config={
                        "AUC": st.column_config.NumberColumn(format="%.2f"),
                        "Ø Prognose": st.column_config.NumberColumn(format="percent"),
                        "Anteil positiv": st.column_config.NumberColumn(format="percent"),
                        "Ø Rendite": st.column_config.NumberColumn(format="percent")})


def _label(k: str) -> str:
    base = k[2:] if k.startswith("s_") else k
    if base in IND_BY_KEY:
        return IND_BY_KEY[base].label
    for cat, col in CAT_COL.items():
        if base == col:
            return f"Bereich {cat}"
        if base == f"{col}_chg":
            return f"Bereich {cat} – Veränderung 3 M."
    return {"F_gesamt": "Fundamental gesamt", "F_gesamt_chg": "Fundamental gesamt – Veränderung 3 M.",
            "mom_pct": "Kursmomentum"}.get(base, base)


def effects_tab(M, d, mode) -> None:
    eff, learned = M["effects"], M["learned"]
    if eff.empty:
        st.info("Zu wenig Historie.")
        return
    keys = [k for k in eff["key"].unique() if k in IND_BY_KEY]
    k = st.selectbox("Indikator", keys, format_func=lambda x: f"{IND_BY_KEY[x].category} · {IND_BY_KEY[x].label}",
                     key="fund_eff")
    ind = IND_BY_KEY[k]
    st.caption(ind.explain)
    e = eff[eff["key"] == k]
    left, right = st.columns(2)
    with left:
        plot(bar_chart(e, "Quintil", "Ø 6 M.", None, mode, "Ø Rendite der nächsten 6 Monate je Perzentil-Bereich",
                       kind="pct"), "eff_bar")
        st.caption("0–20 = Indikator so tief wie selten in 10 Jahren, 80–100 = so hoch wie selten. Modellfrei, "
                   "überlappende Zeiträume.")
    with right:
        show = e[["Quintil", "Tage", "Ø 6 M.", "Ø 12 M.", "Anteil positiv 6 M.", "Rückgang ≥ 10 % (6 M.)"]]
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="percent") for c in show.columns[2:]})
        ly = learned[learned["key"] == k].sort_values("year")
        if not ly.empty:
            l = ly.iloc[-1]
            st.markdown(f"**Gelernt (Stand {int(l['year'])}):** {'hoch = gut' if l['sign'] > 0 else 'tief = gut'}, "
                        f"Gewicht {l['weight']:.1f} ({l['source']}), Rangkorrelation {l['ic']:+.2f}, t = {l['t']:+.1f}"
                        if pd.notna(l["ic"]) else f"**Gelernt:** {l['source']}")
    ly = learned[learned["key"] == k].sort_values("year")
    if len(ly) > 1:
        ly = ly.assign(Gewicht=ly["weight"] * ly["sign"], date=pd.to_datetime(ly["year"].astype(str) + "-01-01"))
        plot(date_lines(ly, "date", {"Gewicht": "Gewicht × Richtung"}, mode,
                        "Wie sich das Gelernte über die Jahre verändert hat (positiv = hoch ist gut)", kind="num",
                        height=260, hline=0), "eff_hist")
    st.markdown("##### Alle Indikatoren – aktuell gelernt")
    last_year = learned["year"].max()
    lt = learned[learned["year"] == last_year].copy()
    lt = lt[lt["key"].isin(IND_BY_KEY)]
    lt["Indikator"] = lt["key"].map(lambda x: IND_BY_KEY[x].label)
    lt["Bereich"] = lt["key"].map(lambda x: IND_BY_KEY[x].category)
    lt["Richtung"] = np.where(lt["sign"] > 0, "hoch = gut", "tief = gut")
    lt["Lehrbuch"] = lt["prior"].map({1: "hoch = gut", -1: "tief = gut", 0: "offen"})
    lt = lt.sort_values("weight", ascending=False)
    st.dataframe(lt[["Bereich", "Indikator", "Richtung", "Lehrbuch", "weight", "ic", "t", "source"]].rename(
        columns={"weight": "Gewicht", "ic": "Rangkorr. 6 M.", "t": "t-Wert", "source": "Quelle"}),
        hide_index=True, width="stretch", column_config={
            "Gewicht": st.column_config.NumberColumn(format="%.1f"),
            "Rangkorr. 6 M.": st.column_config.NumberColumn(format="%+.2f"),
            "t-Wert": st.column_config.NumberColumn(format="%+.1f")})
    st.caption("„gelernt“ = |t| ≥ 1 (statistisch erkennbar), „Grundrichtung“ = zu schwach, ökonomische Richtung mit "
               "kleinem Gewicht, „kein Signal“ = ohne Einfluss. Bei überlappenden Renditen sind t-Werte mit Vorsicht "
               "zu lesen (effektive Stichprobe berücksichtigt).")


def priced_tab(d, M, key, mode, meta, data) -> None:
    last = d.iloc[-1]
    st.markdown("Ein Markt bewegt sich vor allem, wenn sich Erwartungen ändern. Das Terminal misst deshalb, ob der Kurs "
                "den Fundamentaldaten **vorausgelaufen** ist (viel Optimismus eingepreist) oder **hinterherhinkt** "
                "(Fundamentaldaten besser als die Stimmung).")
    c = st.columns(4)
    c[0].metric("Kursmomentum (6 M., Perzentil)", f"{last.get('mom_pct', np.nan):.0f}")
    c[1].metric("Fundamental gesamt", f"{last.get('F_gesamt', np.nan):.0f}")
    c[2].metric("Differenz", f"{last.get('divergence', np.nan):+.0f}", div_text(last.get("divergence")),
                delta_color="off")
    pe = last.get("raw_pe")
    if pd.notna(pe):
        c[3].metric("KGV S&P 500 (Näherung)", f"{pe:.1f}x", f"Perzentil 10 J.: {last.get('s_pe', np.nan):.0f}",
                    delta_color="off")
    bt = M["btdiv"]
    if not bt.empty:
        h = st.radio("Horizont", [63, 126, 252], index=1, horizontal=True, key="fund_div_h",
                     format_func=lambda x: {63: "3 Monate", 126: "6 Monate", 252: "12 Monate"}[x])
        b = bt[bt["h"] == h]
        show = b[["Lage", "Tage", "Ø Rendite", "Anteil positiv", "Rückgang ≥ 10 %"]]
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="percent") for c in ("Ø Rendite", "Anteil positiv",
                                                                         "Rückgang ≥ 10 %")})
        st.caption(f"Historisch (ab {M['meta'].get('start', '')[:4]}), alle Tage: Ø {b['Basis Ø Rendite'].iloc[0]:+.1%}, "
                   f"positiv {b['Basis positiv'].iloc[0]:.0%}, Rückgang ≥ 10 % {b['Basis ≥10%'].iloc[0]:.0%}.")
    st.markdown("##### Was der Markt sonst einpreist")
    rows = []
    for k, txt in (("policy_exp", "Zinspfad: 2-jährige Rendite minus Leitzins (negativ = Senkungen eingepreist)"),
                   ("breakeven", "Inflation, die der Anleihemarkt für 10 Jahre erwartet"),
                   ("erp", "Risikoprämie: Gewinnrendite minus Realzins (tief = Aktien teuer vs. Anleihen)"),
                   ("hy_spread", "Kreditrisiko, das Hochzinsanleihen einpreisen"),
                   ("cot_index", "Positionierung der Spekulanten (über 80 = überfüllt long, unter 20 = überfüllt short)")):
        if f"raw_{k}" in d.columns and pd.notna(last.get(f"raw_{k}")):
            ind = IND_BY_KEY[k]
            rows.append({"Grösse": ind.label, "Heute": fmt(last.get(f"raw_{k}"), ind.fmt),
                         "Perzentil 10 J.": last.get(f"s_{k}"), "Bedeutung": txt})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Perzentil 10 J.": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100)})
    earn = data.get("earnings")
    if key == "SPY" and earn is not None and not earn.empty:
        e = earn.copy()
        e["date"] = pd.to_datetime(e["date"])
        e = e[e["ni_growth"].notna()]
        plot(date_lines(e, "date", {"ni_growth": "Gewinnwachstum (TTM)", "rev_growth": "Umsatzwachstum (TTM)"}, mode,
                        "S&P 500: Gewinn- und Umsatzwachstum aus SEC-Daten (like-for-like, GAAP)", height=300, hline=0),
             "fund_earn")
        anc = data["meta"].get("pe_anchor", {})
        st.caption(f"Monatlich zum Monatsende, nur mit damals veröffentlichten Berichten. GAAP-Gewinne (inkl. "
                   f"Sondereffekte, z. B. Bewertungsgewinne bei Berkshire) – schwankender als „operative“ Gewinne. "
                   f"KGV-Anker: {anc.get('n', 0)} Firmen mit einer Aktienklasse.")


def combo_tab(d, key) -> None:
    h = load_health(top_version(key), key)
    if h is None:
        st.info("Für diese Kombination zuerst im Markt-Terminal die Markt-Analysen berechnen.")
        return
    t = h["daily"].set_index("date")[["H_kurz", "H_mittel", "H_lang"]]
    j = d[["close", "F_gesamt"]].join(t, how="inner")
    if j["F_gesamt"].notna().sum() < 500:
        st.info("Zu wenig gemeinsame Historie.")
        return
    st.markdown("Fundamentaldaten geben die **Richtung**, die Technik das **Timing**. Hier siehst du, wie sich der "
                "Index entwickelte, je nachdem wie beide zusammenstanden.")
    hz = st.radio("Horizont", [21, 63], horizontal=True, key="combo_h",
                  format_func=lambda x: f"{x} Handelstage (~{x // 21} Monat{'e' if x > 21 else ''})")
    fwd = j["close"].shift(-hz) / j["close"] - 1
    fb = pd.cut(j["F_gesamt"], [-1, 40, 60, 101], labels=["Fundamental bärisch (< 40)", "neutral (40–60)",
                                                            "Fundamental bullisch (≥ 60)"])
    tb = pd.cut(j["H_kurz"], [-1, 20, 50, 101], labels=["Technik: Dip (Kurz < 20)", "Kurz 20–50", "Kurz ≥ 50"])
    df = pd.DataFrame({"F": fb, "T": tb, "r": fwd}).dropna()
    piv = df.groupby(["F", "T"], observed=True)["r"].agg(["mean", lambda r: (r > 0).mean(), "count"])
    piv.columns = ["Ø Rendite", "Anteil positiv", "Tage"]
    piv = piv.reset_index()
    mode = theme_mode()
    zr = piv.pivot(index="F", columns="T", values="Ø Rendite")
    zp = piv.pivot(index="F", columns="T", values="Anteil positiv")
    zn = piv.pivot(index="F", columns="T", values="Tage")
    left, right = st.columns(2)
    with left:
        plot(matrix_fig(zr, zr.map(lambda v: f"{v:+.2%}") + zn.map(lambda n: f"<br>{n:.0f} Tage"), mode,
                        "Ø Rendite", 0), "combo_r")
    with right:
        base = (df["r"] > 0).mean()
        plot(matrix_fig(zp, zp.map(lambda v: f"{v:.0%}"), mode, f"Anteil positiv (alle Tage: {base:.0%})", base),
             "combo_p")
    with st.expander("Anzahl Tage je Feld"):
        st.dataframe(piv.pivot(index="F", columns="T", values="Tage"), width="stretch")
    lj = j.iloc[-1]
    st.caption(f"Heute: Fundamental {lj['F_gesamt']:.0f}, Kurzfrist-Technik {lj['H_kurz']:.0f}. Überlappende "
               "Zeiträume – die Tage sind nicht unabhängig.")
    st.markdown("##### Dip-Regel mit Fundamental-Filter")
    dd = j.rename(columns={"F_gesamt": "F"})
    tr = rule_trades(dd, "dreh", 20, 40, 21, 0.0005)
    if tr.empty:
        st.caption("Keine Trades.")
        return
    tr["F"] = dd["F"].reindex(tr["Einstieg"]).values
    rows = []
    for name, g in (("Alle Signale", tr), ("nur Fundamental ≥ 50", tr[tr["F"] >= 50]),
                    ("nur Fundamental < 50", tr[tr["F"] < 50])):
        s = trade_stats(g)
        if s["n"]:
            rows.append({"Variante": name, "Trades": s["n"], "Winrate": s["winrate"], "Profit-Faktor": s["pf"],
                         "Ø Trade": s["avg"], "Schlechtester": s["worst"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Winrate": st.column_config.NumberColumn(format="percent"),
        "Profit-Faktor": st.column_config.NumberColumn(format="%.2f"),
        "Ø Trade": st.column_config.NumberColumn(format="percent"),
        "Schlechtester": st.column_config.NumberColumn(format="percent")})
    st.caption("Regel „Drehung“ aus dem Markt-Terminal: Kurzfrist-Score steigt wieder über 20, Langfrist ≥ 40, 21 Tage "
               "halten, 0.05 % Kosten je Seite. Fundamental-Score am Einstiegstag (nur damals bekannte Daten).")


def matrix_fig(z: pd.DataFrame, text: pd.DataFrame, mode: str, title: str, mid: float) -> go.Figure:
    """Divergierende Farbmatrix (rot = schlechter als mid, blau = besser) mit Werten als Text."""
    p = PALETTE[mode]
    span = float(np.nanmax(np.abs(z.values - mid))) or 1.0
    fig = go.Figure(go.Heatmap(z=z.values, x=[str(c) for c in z.columns], y=[str(i) for i in z.index],
                               text=text.values, texttemplate="%{text}", textfont=dict(size=12), showscale=False,
                               colorscale=[[0, p["div"][0]], [0.5, p["div"][1]], [1, p["div"][2]]],
                               zmin=mid - span, zmax=mid + span, xgap=3, ygap=3,
                               hovertemplate="%{y} · %{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(side="bottom", type="category")
    return base_layout(fig, mode, title, height=320)


def test_tab(M, last, mode) -> None:
    bt = M["bt"]
    if bt.empty:
        st.info("Zu wenig Historie.")
        return
    h = st.radio("Horizont", [63, 126, 252], index=1, horizontal=True, key="fund_bt_h",
                 format_func=lambda x: {63: "3 Monate", 126: "6 Monate", 252: "12 Monate"}[x])
    b = bt[bt["h"] == h]
    bb = b[b["Tage"] >= 100]
    c = st.columns(3)
    if len(bb) >= 2:
        c[0].metric("Ø Rendite: bärischster vs. bullischster Bereich",
                    f"{bb['Ø Rendite'].iloc[0]:+.1%} vs. {bb['Ø Rendite'].iloc[-1]:+.1%}")
        c[1].metric("Rückgang ≥ 10 %: bärisch vs. bullisch",
                    f"{bb['Rückgang ≥ 10 %'].iloc[0]:.0%} vs. {bb['Rückgang ≥ 10 %'].iloc[-1]:.0%}")
    c[2].metric("Rangkorrelation Score ↔ Rendite", f"{b['IC'].iloc[0]:+.2f}",
                help="Über 0.1 ist für Fundamentaldaten bereits bemerkenswert.")
    plot(bar_chart(b.dropna(subset=["Ø Rendite"]), "Bereich", "Ø Rendite", None, mode,
                   "Ø Rendite je Bereich des Fundamental-Scores", kind="pct"), "fund_bt")
    show = b[["Bereich", "Tage", "Ø Rendite", "Anteil positiv", "Rückgang ≥ 10 %"]]
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        c: st.column_config.NumberColumn(format="percent") for c in ("Ø Rendite", "Anteil positiv", "Rückgang ≥ 10 %")})
    st.caption(f"Nur Tage ab {M['meta'].get('start', '')[:4]} – der Score nutzte an jedem Tag nur Gewichte, die aus "
               "früheren Jahren gelernt waren. Überlappende Zeiträume.")


METHOD = """
**Daten (alles gratis, ohne Schlüssel):** FRED (US-Notenbank St. Louis) für Zinsen, Notenbankbilanzen, Kreditspreads,
Konjunktur, Inflation, Wechselkurse und OECD-Daten zu Japan; Yahoo für Öl, Kupfer und Gold (vor 2000 Öl/Kupfer von FRED); CFTC Commitments of Traders für die Positionierung in
S&P-500- und Nikkei-Futures; die SEC-Berichte aller S&P-500-Firmen für Gewinne, Umsätze und Margen.

**Kein Blick in die Zukunft:** Jede Reihe gilt erst ab ihrer üblichen Veröffentlichung (z. B. Inflation ~45 Tage nach
Monatsbeginn, COT 3 Tage nach Stichtag, Firmengewinne ab Einreichung des Berichts). Spätere Revisionen der Behörden
werden nicht nachgebildet (FRED zeigt den heutigen Datenstand) – bei Konjunkturdaten ein kleiner Vorteil, den es
damals nicht gab.

**Lernen:** Jedes Jahr am 1. Januar misst das Terminal für jeden Indikator, wie sein Perzentil mit der Rendite der
folgenden 6 Monate zusammenhing (nur abgeschlossene Zeiträume). Daraus folgen Richtung und Gewicht. So passt sich das
Terminal an – z. B. wenn ein schwacher Yen für den Nikkei wichtiger wird – ohne heutiges Wissen zu verwenden.

**ML-Prognose:** Gradient Boosting auf allen Scores, jährlich neu trainiert, getestet Jahr für Jahr. Die Test-AUC zeigt
ehrlich, ob es einen Vorteil gab (0.5 = Zufall).

**Eingepreist:** Kursmomentum (6 Monate, als Perzentil) minus Fundamental-Score. Zusätzlich die Erwartungen, die
Anleihen (Zinspfad, Inflation), Kreditmarkt und Positionierung direkt einpreisen.

**Grenzen:** Fundamentaldaten wirken langsam (Monate). Für das Timing ist die Technik zuständig – siehe Tab
„Fundamental + Technik“. Gewinndaten der SEC erst ab ca. 2011, Japan-Gewinne sind nicht frei verfügbar (der
S&P-Gewinnzyklus dient als globaler Gewinnzyklus).
"""
