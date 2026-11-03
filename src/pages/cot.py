"""COT-Signale: Gold, Öl, Kupfer, S&P 500, Nasdaq 100 – COT-Index 0–100 und Long / Neutral / Short."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from src.components.ml_data import DISCLAIMER
from src.components.ui import plot, theme_mode
from src.config import get_settings
from src.cot.model import FEATURES, MARKETS, SIG_NAME, backtest, by_signal, signal
from src.visualization.charts import cot_chart, date_lines, gauge

ICON = {1: "🟢 Long", 0: "🟡 Neutral", -1: "🔴 Short"}
PERIODS = {"2 Jahre": 730, "5 Jahre": 1826, "10 Jahre": 3652, "Alles": None}


def cot_dir():
    return get_settings().data_dir / "cot"


def version() -> float:
    p = cot_dir() / "meta.json"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def load(v: float) -> dict | None:
    d = cot_dir()
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    out = {"meta": meta, "weekly": {}, "learned": {}}
    for k in MARKETS:
        p = d / f"{k}_weekly.parquet"
        if p.exists():
            w = pd.read_parquet(p)
            w["date"] = pd.to_datetime(w["date"])
            out["weekly"][k] = w.set_index("date")
            q = d / f"{k}_learned.parquet"
            out["learned"][k] = pd.read_parquet(q) if q.exists() else pd.DataFrame()
    return out


def page() -> None:
    st.title("COT-Signale")
    data = load(version())
    if not data or not data["weekly"]:
        st.info("Noch kein COT-Modell berechnet. Auf der Seite **Daten laden** „COT-Modell berechnen“ klicken "
                "(lädt die CFTC-Daten und Kurse, ca. 1–3 Minuten).")
        return
    st.caption("Commitments of Traders der CFTC (wöchentlich, Stichtag Dienstag, veröffentlicht Freitag) – dieselben "
               "Zahlen wie „CFTC … speculative net positions“ in Wirtschaftskalendern (z. B. Tradays), ergänzt um "
               "Commercials und kleine Spekulanten. Das Modell lernt je Markt, welche Positionierung historisch "
               "bullisch war, und fasst sie im **COT-Score 0–100** zusammen. Gehandelt wird erst zum Montagsschluss "
               "nach der Veröffentlichung.")
    c1, c2, c3 = st.columns([2, 2, 1])
    method = c1.radio("Index", ["score", "classic"], horizontal=True, key="cot_method",
                      format_func={"score": "Gelernter COT-Score", "classic": "Klassisch: Commercials-Index 3 J."}.get)
    default = (40, 60) if method == "score" else (20, 80)
    lo, hi = c2.slider("Short bis / Long ab", 0, 100, default, 5, key=f"cot_th_{method}",
                       help="Score ≥ Long-Schwelle = Long, ≤ Short-Schwelle = Short, dazwischen Neutral.")
    allow_short = c3.toggle("Short handeln", True, key="cot_short")

    rows, results = [], {}
    for k, w in data["weekly"].items():
        idx = w[method]
        sig = signal(idx, hi, lo)
        start = data["meta"]["markets"].get(k, {}).get("start")
        bt = backtest(w, sig, start=pd.Timestamp(start) if start else None, allow_short=allow_short)
        results[k] = (sig, bt, start)
        last = w.iloc[-1]
        prev = w.iloc[-2] if len(w) > 1 else last
        s_now = sig.iloc[-1]
        rows.append({"Markt": MARKETS[k]["label"], "Stichtag": w.index[-1].date(), "Index": last[method],
                     "Signal": ICON.get(int(s_now), "–") if pd.notna(s_now) else "–",
                     "Seit": _since(sig), "Spekulanten netto (Kontrakte)": last["nc_contracts"],
                     "Δ Woche": last["nc_contracts"] - prev["nc_contracts"],
                     "Modell p.a.": bt.get("cagr"), "Kaufen & Halten p.a.": bt.get("cagr_bh"),
                     "Winrate": bt.get("winrate"), "Trades": bt.get("n_trades")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Index": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=100),
        "Spekulanten netto (Kontrakte)": st.column_config.NumberColumn(format="%,.0f"),
        "Δ Woche": st.column_config.NumberColumn(format="%+,.0f"),
        "Modell p.a.": st.column_config.NumberColumn(format="percent"),
        "Kaufen & Halten p.a.": st.column_config.NumberColumn(format="percent"),
        "Winrate": st.column_config.NumberColumn(format="percent")})
    st.caption("„Spekulanten netto“ = Non-Commercials long minus short (die Zahl aus dem Wirtschaftskalender). "
               "Modell p.a. / Winrate: historischer Test des Signals mit den gewählten Schwellen (nur Wochen, in denen "
               "das Modell bereits aus der Vergangenheit gelernt hatte; 0.05 % Kosten je Wechsel).")

    key = st.radio("Markt im Detail", list(data["weekly"]), horizontal=True, key="cot_mkt",
                   format_func=lambda k: MARKETS[k]["label"])
    detail(key, data, results[key], method, lo, hi, allow_short)
    with st.expander("Wie funktioniert das COT-Modell?"):
        st.markdown(METHOD)
    st.caption(DISCLAIMER)


def _since(sig: pd.Series) -> str:
    s = sig.dropna()
    if s.empty:
        return "–"
    cur = s.iloc[-1]
    n = 0
    for v in s.values[::-1]:
        if v != cur:
            break
        n += 1
    return f"{n} Wo."


def detail(key, data, res, method, lo, hi, allow_short) -> None:
    mode = theme_mode()
    w = data["weekly"][key].copy()
    sig, bt, start = res
    last = w.iloc[-1]
    c = st.columns([1, 1, 2])
    with c[0]:
        plot(gauge(last[method], "COT-Score" if method == "score" else "Commercials-Index", MARKETS[key]["label"], mode,
                   w[method].iloc[-2] if len(w) > 1 else None), f"cot_g_{key}")
        s_now = sig.iloc[-1]
        st.markdown(f"<div style='text-align:center;margin-top:-8px'><b>{ICON.get(int(s_now), '–')}</b> "
                    f"seit {_since(sig)}</div>" if pd.notna(s_now) else "", unsafe_allow_html=True)
    with c[1]:
        st.metric("Spekulanten netto", f"{last['nc_contracts']:,.0f}".replace(",", "'"),
                  f"{last['nc_net']:+.1%} des Open Interest", delta_color="off")
        st.metric("Commercials-Index 3 J.", f"{last['c_idx156']:.0f}", help="0 = Absicherer so short wie seit 3 Jahren "
                  "nicht, 100 = so long wie nie (klassisch bullisch)")
        st.metric("Spekulanten-Index 3 J.", f"{last['nc_idx156']:.0f}", help="Über 80 = Spekulanten stark long "
                  "(überfüllt), unter 20 = stark short")
    with c[2]:
        if bt:
            m = st.columns(3)
            m[0].metric("Modell p.a.", f"{bt['cagr']:+.1%}", f"B&H {bt['cagr_bh']:+.1%}", delta_color="off",
                        help="B&H = Kaufen & Halten im selben Zeitraum")
            m[1].metric("Max. Rückgang", f"{bt['mdd']:.0%}", f"B&H {bt['mdd_bh']:.0%}", delta_color="off")
            m[2].metric("Winrate Trades", f"{bt['winrate']:.0%}" if pd.notna(bt["winrate"]) else "–",
                        f"{bt['n_trades']} · PF {bt['pf']:.2f}" if np.isfinite(bt["pf"]) else f"{bt['n_trades']} Trades",
                        delta_color="off")
            st.caption(f"Test ab {bt['start']}: {bt['time_long']:.0%} der Zeit long, {bt['time_short']:.0%} short. "
                       f"Ø Trade {bt['avg_trade']:+.1%}.")
        else:
            st.info("Für diesen Markt gibt es noch zu wenig Kursdaten für einen Test.")

    per = st.radio("Zeitraum", list(PERIODS), index=1, horizontal=True, key="cot_per")
    v = w[w["price"].notna()].copy()
    v.index = v["trade_date"]
    v["close"] = v["price"]
    v["sig100"] = sig.reindex(w.index).map({1: 100.0, 0: 50.0, -1: 0.0}).values[w["price"].notna().values]
    days = PERIODS[per]
    if days:
        v = v[v.index >= v.index.max() - pd.Timedelta(days=days)]
    v["c_contracts"] = v["c_net"] * v["oi"]
    v["nr_contracts"] = v["nr_net"] * v["oi"]
    opts = {"COT-Score (gelernt)": "score", "Commercials-Index 3 J.": "c_idx156", "Spekulanten-Index 3 J.": "nc_idx156",
            "Commercials-Index 52 W.": "c_idx52", "Spekulanten-Index 52 W.": "nc_idx52",
            "Commercials-Index 26 W.": "c_idx26", "Spekulanten-Index 26 W.": "nc_idx26"}
    main = "COT-Score (gelernt)" if method == "score" else "Commercials-Index 3 J."
    pick = st.multiselect("Linien", list(opts), default=[main, *[o for o in ("Commercials-Index 3 J.",
                          "Spekulanten-Index 3 J.") if o != main]], key=f"cot_lines_{key}")
    show_net = st.toggle("Netto-Positionen in Kontrakten anzeigen (wie Tradays)", True, key="cot_net")
    if len(v) > 5 and pick:
        if main in pick:
            pick = [main] + [p for p in pick if p != main]
        net = {"Spekulanten netto": "nc_contracts", "Commercials netto": "c_contracts",
               "Kleine Spekulanten netto": "nr_contracts"} if show_net else None
        plot(cot_chart(v, mode, MARKETS[key]["label"], {p: opts[p] for p in pick}, lo, hi, net), f"cot_chart_{key}")
        st.caption("Oben: Kurs, grün hinterlegt = Long-Signal, rot = Short-Signal, ohne Farbe = Neutral. Mitte: die "
                   "Indizes als Linien (0 = Tiefstwert, 100 = Höchstwert im Zeitraum), gestrichelt deine Schwellen – "
                   "sie gelten für die dicke Linie (" + main + "). Faustregel: Commercials hoch = bullisch, "
                   "Spekulanten hoch = überfüllt (eher bärisch). Unten: Netto-Positionen (long minus short) in "
                   "Kontrakten; „Spekulanten netto“ ist die Zahl aus Tradays.")

    t1, t2, t3 = st.tabs(["Was nach den Signalen passierte", "Kapitalkurve & Trades", "Gelernte Gewichte"])
    with t1:
        start_ts = pd.Timestamp(start) if start else None
        bs = by_signal(w, sig, start_ts)
        cols = [c for c in bs.columns if c.startswith("Ø") or c.startswith("positiv")]
        st.dataframe(bs, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="percent") for c in cols})
        st.caption("Ø Kursveränderung 4/13/26 Wochen nach jedem Wochen-Signal (ab Testbeginn). Gut ist, wenn Long "
                   "deutlich über „Alle Wochen“ liegt und Short darunter. Überlappende Zeiträume.")
    with t2:
        if bt:
            eq = bt["equity"].reset_index().rename(columns={"index": "date"})
            eq["date"] = eq["date"] + pd.Timedelta(days=6)
            fe = date_lines(eq, "date", {"Modell": "COT-Modell", "Kaufen & Halten": "Kaufen & Halten"}, mode,
                            "Wachstum von 1 (log. Skala)", kind="num", height=320)
            fe.update_yaxes(type="log")
            plot(fe, f"cot_eq_{key}")
            tr = bt["trades"].sort_values("Einstieg", ascending=False).copy()
            for col in ("Einstieg", "Ausstieg"):
                tr[col] = tr[col].dt.date
            st.dataframe(tr, hide_index=True, width="stretch", column_config={
                "Rendite": st.column_config.NumberColumn(format="percent")})
    with t3:
        L = data["learned"].get(key, pd.DataFrame())
        if not L.empty:
            ly = L[L["year"] == L["year"].max()].copy()
            ly["Merkmal"] = ly["key"].map(lambda k: FEATURES[k][0])
            ly["Richtung"] = np.where(ly["sign"] > 0, "hoch = bullisch", "hoch = bärisch")
            ly["Lehrbuch"] = ly["key"].map(lambda k: {1: "hoch = bullisch", -1: "hoch = bärisch", 0: "offen"}[FEATURES[k][1]])
            st.dataframe(ly.sort_values("weight", ascending=False)[["Merkmal", "Richtung", "Lehrbuch", "weight", "ic",
                                                                    "t", "source"]].rename(columns={
                "weight": "Gewicht", "ic": "Rangkorr. 13 W.", "t": "t-Wert", "source": "Quelle"}),
                hide_index=True, width="stretch", column_config={
                    "Gewicht": st.column_config.NumberColumn(format="%.1f"),
                    "Rangkorr. 13 W.": st.column_config.NumberColumn(format="%+.2f"),
                    "t-Wert": st.column_config.NumberColumn(format="%+.1f")})
            st.caption(f"Stand Lernjahr {int(L['year'].max())}. Gelernt aus der Rendite der 13 Wochen nach jeder "
                       "Veröffentlichung, nur abgeschlossene Zeiträume.")


METHOD = """
**Daten:** CFTC Commitments of Traders, Legacy-Bericht (nur Futures): Gold COMEX (088691), Rohöl WTI NYMEX (067651),
Kupfer COMEX (085692), E-mini S&P 500 (13874A), E-mini Nasdaq 100 (209742). Kurse: Yahoo (GC=F, CL=F, HG=F – Futures-
Fortschreibung; ^GSPC, ^NDX), Öl vor 2000 von FRED.

**Gruppen:** *Non-Commercials* = grosse Spekulanten (Fonds) – das ist die „speculative net position“. *Commercials* =
Produzenten/Absicherer, oft auf der Gegenseite. *Non-Reportables* = kleine Händler.

**COT-Index:** Wo steht die Netto-Position innerhalb der letzten 26 / 52 / 156 Wochen (0 = Tiefstwert, 100 =
Höchstwert)? Klassisch (Larry Williams): Commercials-Index über 80 = bullisch, unter 20 = bärisch.

**Gelernter COT-Score:** Alle Merkmale als Perzentil; jedes Jahr lernt das Modell aus der Vergangenheit, welche
Richtung und welches Gewicht jedes Merkmal für die folgenden 13 Wochen hatte. So kann es z. B. lernen, dass bei Aktien-
Futures die Spekulanten eher dem Trend folgen, bei Rohstoffen aber an Extremen falsch liegen.

**Signal & Test:** Long ab der oberen, Short ab der unteren Schwelle. Position ab Montagsschluss nach der
Veröffentlichung, gehalten bis zum nächsten Signalwechsel, 0.05 % Kosten je Wechsel. Kein Stop-Loss, keine
Finanzierungs-/Rollkosten. COT ist ein langsamer Indikator (Wochen bis Monate) – das Timing kommt vom Chart.
"""
