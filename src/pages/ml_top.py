from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ml_data import DISCLAIMER, load_top, top_markets, top_version
from src.components.ui import plot, theme_mode
from src.ml.market_top import AMPEL, DD_THRESHOLD, HORIZON, TOP_FEATURES, drawdown_episodes, evaluate_warnings
from src.ml.markets import MARKETS
from src.ml.signals import exit_signals, perf_stats, signal_stats, simulate
from src.visualization.charts import bar_chart, date_lines, regime_chart


def _pct(v, signed=False) -> str:
    if v is None or pd.isna(v):
        return "–"
    return f"{v:+.1%}" if signed else f"{v:.1%}"


def page() -> None:
    st.title("Top-Warnung")
    keys = top_markets()
    if not keys:
        st.info("Noch keine Top-Warnung berechnet. Auf der Seite **Daten laden** unten „Nur Top-Warnung berechnen“ "
                "klicken (S&P 500, JP225 und EWJ; beim ersten Mal 5–10 Minuten wegen der japanischen Einzelaktien).")
        return
    key = st.radio("Markt", keys, format_func=lambda k: MARKETS[k]["label"], horizontal=True, key="top_mkt")
    top = load_top(top_version(key), key)
    mode = theme_mode()
    meta, d = top["meta"], top["daily"].set_index("date")
    name = meta.get("short", key)
    fty = int(meta.get("first_test_year", 2014))
    start = pd.Timestamp(fty, 1, 1)
    signs = meta.get("ampel_signs", list(AMPEL))
    src = [f"Kurse ab {meta.get('first_date', '?')}",
           f"Volatilität: {'Index' if meta.get('vol_source') == 'index' else 'realisierte Vol. (kein Vol-Index)'}",
           f"Marktbreite: {meta.get('n_stocks_breadth', 0)} Aktien" if meta.get("n_stocks_breadth") else "keine Marktbreite",
           "USD/JPY" if meta.get("has_fx") else None, "US-Kreditmarkt" if meta.get("has_credit") else None]
    st.caption(f"Ziel: Phasen erkennen, in denen der {name} in den nächsten ~3 Monaten ({HORIZON} Handelstage) mit "
               f"erhöhter Wahrscheinlichkeit um mindestens {abs(DD_THRESHOLD):.0%} fällt. Stand: {meta['last_date']}. "
               "Daten: " + " · ".join(x for x in src if x) + ".")
    if meta.get("nikkei_source"):
        st.caption(f"Nikkei-225-Liste: {meta['nikkei_source']}")

    base = meta["base_rate"]
    ml_ok = pd.notna(meta.get("auc_oos")) and meta["auc_oos"] >= 0.55
    if not ml_ok:
        st.error(f"**Das ML-Top-Risiko ist für {name} nicht verlässlich** (AUC {meta['auc_oos']:.2f} im historischen "
                 "Test – 0.5 wäre Zufall). Nutze es nicht als Signal; der orange Streifen ist ausgeblendet.")
    else:
        st.success(f"ML-Top-Risiko für {name}: AUC {meta['auc_oos']:.2f} im historischen Test ab {fty} – besser als Zufall.")
    thr = st.slider("Warnschwelle ML (Vielfaches des normalen Risikos)", 1.2, 3.0, 2.0, 0.1, key="top_thr",
                    help=f"Normales Risiko (historischer Durchschnitt): {base:.0%}.")
    level = thr * base
    amp_n = st.slider("Ampel warnt ab … Zeichen", 1, max(2, len(signs)), 2 if len(signs) <= 5 else 3, key="top_ampn",
                      help="Mit mehr verfügbaren Warnzeichen ist eine höhere Schwelle sinnvoll (weniger Fehlalarme).")
    warn_ml = (d["prob"] >= level).astype(int).where(d["prob"].notna(), 0)
    amp_score = d[[k for k in signs if k in d]].sum(axis=1)
    warn_amp = (amp_score >= amp_n).astype(int)
    last = d.dropna(subset=["close"]).iloc[-1]

    c = st.columns(4)
    p_now = last["prob"]
    c[0].metric("Top-Risiko heute (ML)" + ("" if ml_ok else " – unzuverlässig"), _pct(p_now),
                f"{p_now / base - 1:+.0%} vs. normal ({base:.0%})" if pd.notna(p_now) else None, delta_color="inverse")
    active = [AMPEL[k] for k in signs if last.get(k, 0) == 1]
    c[1].metric("Ampel", f"{int(sum(last.get(k, 0) for k in signs))} / {len(signs)}", ", ".join(active) if active else "keine Warnzeichen",
                delta_color="off", help="Verfügbare Warnzeichen: " + ", ".join(AMPEL[k] for k in signs)
                + f". Ab {amp_n} aktiven Zeichen erscheint der gelbe Streifen.")
    c[2].metric("Marktbreite (über 200-Tage-Schnitt)", _pct(last["breadth_200"]),
                f"{last['d_breadth_200_63'] * 100:+.0f} Pp. in 3 Monaten" if pd.notna(last["d_breadth_200_63"]) else None)
    if meta.get("has_fx"):
        c[3].metric("USD/JPY 1 Monat", _pct(last["fx_mom_20"], True),
                    help="Fällt USD/JPY stark (Yen wird stärker), gerieten japanische Aktien oft unter Druck.",
                    delta_color="off")
    else:
        c[3].metric("Abstand zum 52-Wochen-Hoch", _pct(last["spy_dist_high"], True), delta_color="off")

    show_from = st.select_slider("Chart ab", options=sorted({y for y in range(1966, 2027) if y >= d.index.min().year}),
                                 value=max(fty, d.index.min().year), key="top_from")
    mf = d.reset_index()[["date", "close", "sma200", "regime"]]
    mf = mf[mf["date"] >= f"{show_from}-01-01"]
    warn = ({"Top-Warnung (ML)": ("ml", warn_ml)} if ml_ok else {}) | {f"Ampel ≥ {amp_n}": ("ampel", warn_amp)}
    labels = {"SPY": ("SPY (inkl. Dividenden)", "Kurs in USD (log. Skala)"),
              "N225": ("JP225 / Nikkei 225", "Punkte (log. Skala)"),
              "EWJ": ("EWJ (inkl. Dividenden)", "Kurs in USD (log. Skala)")}
    ml_, yt_ = labels.get(key, (name, "Kurs (log. Skala)"))
    plot(regime_chart(mf, mode, f"{name} – Regime (grün/rot) und Warnstreifen am unteren Rand", warn=warn,
                      main_label=ml_, y_title=yt_), "top_main")
    left, right = st.columns(2)
    dd = d.reset_index()
    dd = dd[dd["date"] >= f"{show_from}-01-01"]
    with left:
        plot(date_lines(dd, "date", {"prob": "Top-Risiko (ML)"}, mode, "Top-Risiko im Zeitverlauf", hline=level,
                        hline_label=f"Warnschwelle {level:.0%}"), "top_prob")
    with right:
        if dd["breadth_200"].notna().any():
            plot(date_lines(dd, "date", {"breadth_200": "über 200-Tage-Schnitt", "breadth_50": "über 50-Tage-Schnitt"},
                            mode, "Marktbreite: Anteil Aktien über ihrem Durchschnitt"), "top_breadth")
        elif meta.get("has_fx"):
            plot(date_lines(dd, "date", {"fx_mom_20": "USD/JPY 1-Monats-Veränderung"}, mode, "USD/JPY (Yen-Stärke < 0)",
                            hline=-0.03, hline_label="Warnzeichen −3 %"), "top_fx")

    # ------------------------------------------------------------ Historischer Test
    st.markdown(f"#### Historischer Test (ab {fty}, ohne Blick in die Zukunft)")
    eps = drawdown_episodes(d["close"].dropna())
    tab_ml, st_ml = evaluate_warnings(d, warn_ml, eps, start)
    tab_amp, st_amp = evaluate_warnings(d, warn_amp, eps, start)
    c = st.columns(4)
    c[0].metric("Einbrüche ≥ 10 % seit " + str(fty), st_amp["episodes"])
    c[1].metric(f"Ampel ≥ {amp_n}: erkannt", f"{st_amp['detected']} / {st_amp['episodes']}",
                help="Warnung innerhalb von 3 Monaten vor dem Hoch oder spätestens bevor −10 % erreicht waren.")
    c[2].metric("Ampel: Treffer nach Warntagen", _pct(st_amp["precision"]), f"normal {_pct(st_amp['base_rate'])}",
                delta_color="off")
    c[3].metric("ML: erkannt / AUC", f"{st_ml['detected']} / {st_ml['episodes']}", f"AUC {meta['auc_oos']:.2f}",
                delta_color="off")
    if len(tab_ml):
        t = tab_ml.merge(tab_amp[["Hoch", "Gewarnt", "Vorlauf zum Hoch (Handelstage)"]].rename(
            columns={"Gewarnt": "Ampel gewarnt", "Vorlauf zum Hoch (Handelstage)": "Ampel Vorlauf"}), on="Hoch")
        t = t.rename(columns={"Gewarnt": "ML gewarnt", "Vorlauf zum Hoch (Handelstage)": "ML Vorlauf"})
        t["Max. Rückgang"] = t["Max. Rückgang"].map(lambda v: f"{v:.1%}")
        for col in ("ML Vorlauf", "Ampel Vorlauf"):
            t[col] = t[col].map(lambda v: f"{int(v)} Tage" if pd.notna(v) else "–")
        t = t.astype(object).where(t.notna(), "–")
        st.dataframe(t, hide_index=True, width="stretch", height=min(420, 40 + 35 * len(t)))
        st.caption("Einbruch = Rückgang ≥ 10 % von einem Hoch, bis der Kurs vom Tief wieder ≥ 10 % gestiegen ist "
                   "(so zählen auch Einbrüche innerhalb langer Bärenmärkte einzeln).")

    # ------------------------------------------------------------ Technische Bestätigung
    st.markdown("#### Warnung + technische Bestätigung")
    st.caption("Die Ampel ist nur ein Filter. Ausgelöst wird erst, wenn der Kurs unter den gleitenden Durchschnitt "
               "schliesst. Wiedereinstieg, sobald er wieder darüber schliesst.")
    c1, c2, c3, c4 = st.columns(4)
    ma_len = c1.selectbox("Durchschnitt (Tage)", [20, 50, 100, 200], index=1, key="sig_ma")
    amp_min = c2.selectbox("Ampel mindestens", list(range(1, len(signs) + 1)), index=min(amp_n, len(signs)) - 1,
                           key="sig_amp")
    amp_win = c3.selectbox("Ampel innerhalb der letzten … Tage", [5, 10, 20, 40], index=2, key="sig_win")
    cost = c4.number_input("Kosten je Wechsel (%)", 0.0, 1.0, 0.10, 0.05, key="sig_cost") / 100
    sigs = exit_signals(d, ma_len, amp_min, amp_win)
    rows = []
    for nm, sg in sigs.items():
        s = signal_stats(d, sg, eps, start)
        rows.append({"Regel": nm, "Signale": s["signals"], "Einbrüche erwischt": f"{s['caught']}/{s['episodes']}",
                     "Ø Rückgang nach Signal (bei Einbrüchen)": _pct(s["remaining_drop"], True),
                     "Danach ≥ 10 % gefallen": _pct(s["hit_rate"]), "Median 3M-Rendite danach": _pct(s["median_ret_3m"], True)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    sims = {nm: simulate(d, sg, ma_len if "200" not in nm else 200, cost, start) for nm, sg in sigs.items()}
    first = next(iter(sims.values()))
    eq = pd.DataFrame({"date": first.index, "Kaufen und Halten": first["buy_hold"].values})
    for nm, sm in sims.items():
        eq[nm] = sm["strategy"].values
    cols = {c: c for c in eq.columns if c != "date"}
    fig = date_lines(eq, "date", cols, mode, f"Simulation {name} ab {first.index[0].date()}: Wert von 1 Einheit",
                     kind="num", height=380)
    fig.update_yaxes(type="log")
    plot(fig, "sim_eq")
    stats = [{"Strategie": "Kaufen und Halten", **perf_stats(first["buy_hold"])}]
    for nm, sm in sims.items():
        stats.append({"Strategie": nm, **perf_stats(sm["strategy"], sm["invested"], sm["trade"])})
    stt = pd.DataFrame(stats)
    for col in ("Rendite p. a.", "Max. Rückgang", "Volatilität p. a.", "Zeit investiert"):
        if col in stt:
            stt[col] = stt[col].map(lambda v: _pct(v, col == "Rendite p. a."))
    for col in ("Endwert (Start = 1)", "Rendite/Risiko"):
        stt[col] = stt[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "–")
    stt["Wechsel"] = stt["Wechsel"].map(lambda v: "–" if pd.isna(v) else f"{int(v)}")
    st.dataframe(stt, hide_index=True, width="stretch")
    st.caption("Handel zum Schlusskurs, ausserhalb des Marktes ohne Zins. "
               + ("^N225 ist ein Kursindex ohne Dividenden (wie der JP225-CFD). " if key == "N225" else "")
               + "Keine Steuern, keine Finanzierungskosten von CFDs. Vergangene Ergebnisse sagen nichts über die Zukunft.")

    # ------------------------------------------------------------ Treiber und Rot = Boden
    with st.expander("Was treibt das ML-Top-Risiko?"):
        coef = top["coef"].copy()
        coef["Merkmal"] = coef["feature"].map(lambda f: TOP_FEATURES.get(f, f))
        coef = coef.sort_values("coef")
        plot(bar_chart(coef, "Merkmal", "coef", None, mode,
                       "Gewicht im Modell (standardisiert): positiv = erhöht das Top-Risiko", kind="num",
                       horizontal=True, height=520), "top_coef")
    st.markdown("#### „Rot = Boden?“ – Was passierte nach Beginn einer roten Phase?")
    red = top["red"].copy()
    if len(red):
        red["Beginn Rot"] = pd.to_datetime(red["Beginn Rot"])
        red = red[red["Beginn Rot"] >= f"{show_from}-01-01"]
        cols = ["Schon vom Hoch gefallen", "Rendite 1 Monat", "Rendite 3 Monate", "Rendite 6 Monate",
                "Rendite 12 Monate", "Weiterer max. Rückgang (12M)"]
        c = st.columns(4)
        c[0].metric(f"Rote Phasen seit {show_from}", len(red))
        c[1].metric("Median: schon vom Hoch gefallen", _pct(red["Schon vom Hoch gefallen"].median()))
        c[2].metric("Median-Rendite 6 Monate danach", _pct(red["Rendite 6 Monate"].median(), True))
        c[3].metric("12 Monate später im Plus", _pct((red["Rendite 12 Monate"] > 0).mean()))
        show = red.copy()
        show["Beginn Rot"] = show["Beginn Rot"].dt.date
        for col in cols:
            show[col] = show[col].map(lambda v: _pct(v, True))
        show["Kurs"] = show["Kurs"].round(2)
        st.dataframe(show, hide_index=True, width="stretch", height=360)
    with st.expander("Wie funktioniert das?"):
        st.markdown(
            "- **Ampel**: Breiten-Divergenz (Index < 3 % unter Hoch, Anteil Aktien über 200-Tage-Schnitt in 3 Monaten "
            "≥ 10 Pp. gefallen) · Volatilität ≥ 30 % über 3-Monats-Tief (ohne Vol-Index: realisierte Vol. ≥ 50 %) bei "
            "Index nahe Hoch · US-Kreditmarkt (HYG/IEF unter 50-Tage-Schnitt) bei Index nahe Hoch · Überdehnung "
            "(> 12 % über 200-Tage-Schnitt) · nur Japan: Yen-Stärke (USD/JPY fällt > 3 % in einem Monat).\n"
            "- **ML-Top-Risiko**: logistische Regression, jedes Jahr neu trainiert nur mit Daten, deren Ergebnis "
            "vorher feststand. Fehlende Datenquellen in frühen Jahren werden als „fehlt“ markiert.\n"
            "- **Marktbreite** aus den heutigen Indexmitgliedern (frühe Jahre leicht verzerrt).")
    st.caption(DISCLAIMER)
