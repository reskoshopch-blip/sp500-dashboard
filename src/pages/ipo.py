"""Nasdaq-IPOs: anstehende Börsengänge und wie sich frühere IPOs entwickelt haben."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.cloud.pull import is_cloud
from src.components.ml_data import DISCLAIMER
from src.components.ui import plot, theme_mode
from src.config import get_settings
from src.ipo.nasdaq import HORIZONS
from src.visualization.charts import PALETTE, base_layout

PCT = st.column_config.NumberColumn(format="percent")
USD = st.column_config.NumberColumn(format="compact")


def _dir():
    return get_settings().data_dir / "ipo"


@st.cache_data(show_spinner=False)
def load(version: float, data_dir: str):
    d = _dir()
    deals = pd.read_parquet(d / "deals.parquet")
    perf = pd.read_parquet(d / "performance.parquet")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return deals, perf, meta


def page() -> None:
    st.title("Nasdaq-IPOs")
    st.caption("Anstehende Börsengänge an der Nasdaq und Auswertung der bisherigen: Wie liefen IPOs ab Ausgabepreis "
               "und ab dem ersten Schlusskurs (dort kaufen Privatanleger realistisch ein) – auch gegenüber dem "
               "Nasdaq 100 (QQQ). Quelle: nasdaq.com IPO-Kalender, Kurse Yahoo Finance.")
    p = _dir() / "meta.json"
    if not p.exists():
        st.info("Noch keine Daten. Auf dem PC `ipo.bat` ausführen (holt 3 Jahre Historie, danach täglich mit dem "
                "Update)." + ("" if is_cloud() else " Danach `cloud_sync.bat`."))
        return
    deals, perf, meta = load(p.stat().st_mtime, str(get_settings().data_dir))
    upd = pd.Timestamp(meta["updated_at"]).tz_convert("Europe/Zurich")
    c = st.columns(4)
    c[0].metric("Anstehend", meta["n_upcoming"])
    c[1].metric("Eingereicht (Filing)", meta["n_filed"])
    c[2].metric("Erfolgt (Auswertung)", f"{meta['n_with_prices']} / {meta['n_priced']}", "mit Kursdaten",
                delta_color="off")
    c[3].metric("Stand", f"{upd:%d.%m.%Y}", f"{upd:%H:%M}", delta_color="off")
    if meta.get("fetch_errors"):
        st.caption("Nicht abrufbare Monate: " + ", ".join(meta["fetch_errors"]))

    t1, t2, t3, t4 = st.tabs(["Anstehend", "Bisherige IPOs", "Statistik", "Eingereicht"])
    with t1:
        up = deals[deals["status"] == "upcoming"].sort_values("date")
        if up.empty:
            st.info("Zurzeit keine angekündigten Nasdaq-IPOs im Kalender.")
        else:
            st.dataframe(_deal_table(up, "Erwartet am"), hide_index=True, width="stretch",
                         column_config={"Volumen $": USD})
            _expect(up, meta)
    with t2:
        _history(perf)
    with t3:
        _stats(meta, perf)
    with t4:
        fi = deals[deals["status"] == "filed"].sort_values("date", ascending=False)
        st.dataframe(_deal_table(fi, "Eingereicht am"), hide_index=True, width="stretch",
                     column_config={"Volumen $": USD})
        st.caption("Unternehmen, die den Börsengang bei der SEC beantragt haben – Termin und Preis noch offen.")
    st.caption(DISCLAIMER)


def _deal_table(d: pd.DataFrame, date_label: str) -> pd.DataFrame:
    rng = [f"{lo:.2f}–{hi:.2f}" if lo is not None and hi is not None and lo != hi else
           (f"{hi:.2f}" if hi is not None and not pd.isna(hi) else "–") for lo, hi in zip(d["price_lo"], d["price_hi"])]
    return pd.DataFrame({date_label: pd.to_datetime(d["date"]).dt.strftime("%d.%m.%Y"), "Ticker": d["symbol"],
                         "Unternehmen": d["company"], "Preisspanne $": rng, "Aktien": d["shares"],
                         "Volumen $": d["volume_usd"], "Börse": d["exchange"]})


def _expect(up: pd.DataFrame, meta: dict) -> None:
    """Was die Historie für anstehende IPOs dieser Grösse erwarten lässt."""
    size = {r["Volumen"]: r for r in meta["stats"].get("size", [])}
    if not size:
        return
    def bucket(v):
        if v is None or pd.isna(v):
            return None
        return "< 50 Mio." if v < 50e6 else "50–200 Mio." if v < 200e6 else "200 Mio.–1 Mrd." if v < 1e9 else "> 1 Mrd."
    rows = []
    for _, d in up.iterrows():
        b = size.get(bucket(d["volume_usd"]))
        if b:
            rows.append({"Ticker": d["symbol"], "Unternehmen": d["company"], "Grössenklasse": b["Volumen"],
                         "historisch Ø 1. Tag": b["Ø 1. Tag"], "Ø 3 M. ab 1. Schluss": b["Ø nach 3 M. ab 1. Schluss"],
                         "Anteil im Plus (3 M.)": b["Anteil im Plus"], "Vergleichs-IPOs": b["IPOs"]})
    if rows:
        st.markdown("###### Was frühere Nasdaq-IPOs gleicher Grösse gemacht haben")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={k: PCT for k in ("historisch Ø 1. Tag", "Ø 3 M. ab 1. Schluss",
                                                     "Anteil im Plus (3 M.)")})


def _history(perf: pd.DataFrame) -> None:
    p = perf[perf["status"] == "priced"].sort_values("date", ascending=False)
    if p.empty:
        st.info("Keine erfolgten IPOs im Zeitraum.")
        return
    q = st.text_input("Suchen (Ticker oder Name)", key="ipo_q").strip().lower()
    if q:
        p = p[p["symbol"].fillna("").str.lower().str.contains(q) | p["company"].str.lower().str.contains(q)]
    cols = {"date": "Datum", "symbol": "Ticker", "company": "Unternehmen", "ipo_price": "Ausgabepreis $",
            "volume_usd": "Volumen $", "r_1 Tag": "1. Tag", "r_1 Monat": "1 Monat", "r_3 Monate": "3 Monate",
            "r_12 Monate": "12 Monate", "r_heute": "Heute", "x_3 Monate": "3 M. vs. QQQ"}
    t = p[[c for c in cols if c in p]].rename(columns=cols)
    t["Datum"] = pd.to_datetime(t["Datum"]).dt.strftime("%d.%m.%Y")
    st.dataframe(t, hide_index=True, width="stretch", height=520,
                 column_config={"Volumen $": USD, "Ausgabepreis $": st.column_config.NumberColumn(format="%.2f"),
                                **{k: PCT for k in ("1. Tag", "1 Monat", "3 Monate", "12 Monate", "Heute",
                                                    "3 M. vs. QQQ")}})
    st.caption("Renditen ab Ausgabepreis; „3 M. vs. QQQ“ ab dem ersten Schlusskurs gegenüber dem Nasdaq 100.")
    plot(monthly_chart(perf, theme_mode()), "ipo_month")


def _stats(meta: dict, perf: pd.DataFrame) -> None:
    hz = pd.DataFrame(meta["stats"]["horizons"])
    if hz.empty or hz["IPOs"].sum() == 0:
        st.info("Noch keine Kursdaten für eine Auswertung.")
        return
    one = hz.set_index("Horizont")
    m = st.columns(4)
    m[0].metric("Ø 1. Tag ab Ausgabepreis", f"{one.loc['1 Tag', 'Ø ab Ausgabepreis']:+.0%}",
                f"Median {one.loc['1 Tag', 'Median ab Ausgabepreis']:+.0%}", delta_color="off")
    r3 = one.loc["3 Monate"]
    m[1].metric("3 M. ab 1. Schluss im Plus", f"{r3['Anteil im Plus ab 1. Schluss']:.0%}" if r3["IPOs"] else "–")
    m[2].metric("3 M. besser als QQQ", f"{r3['Anteil besser als QQQ']:.0%}" if r3["IPOs"] else "–",
                f"Median {r3['Median vs. QQQ']:+.1%}" if r3["IPOs"] else None, delta_color="off")
    r12 = one.loc["12 Monate"]
    m[3].metric("12 M. besser als QQQ", f"{r12['Anteil besser als QQQ']:.0%}" if r12["IPOs"] else "–",
                f"{int(r12['IPOs'])} IPOs", delta_color="off")
    plot(horizon_chart(hz, theme_mode()), "ipo_hz")
    st.dataframe(hz, hide_index=True, width="stretch",
                 column_config={c: PCT for c in hz.columns if c not in ("Horizont", "IPOs")})
    left, right = st.columns(2)
    with left:
        st.markdown("###### Hilft ein starker 1. Tag?")
        pop = pd.DataFrame(meta["stats"]["pop"])
        if len(pop):
            st.dataframe(pop, hide_index=True, width="stretch",
                         column_config={c: PCT for c in pop.columns if c not in ("1. Tag", "IPOs")})
    with right:
        st.markdown("###### Nach Grösse des Börsengangs")
        sz = pd.DataFrame(meta["stats"]["size"])
        if len(sz):
            st.dataframe(sz, hide_index=True, width="stretch",
                         column_config={c: PCT for c in sz.columns if c not in ("Volumen", "IPOs")})
    st.caption("Kleine Nasdaq-IPOs (< 50 Mio.) sind oft sehr volatil, teils mit extremen Ausreissern – deshalb "
               "Median und Anteil im Plus beachten, nicht nur den Durchschnitt. IPOs, die später dekotiert wurden, "
               "fehlen bei Yahoo oft (Überlebens-Verzerrung → Ergebnisse eher zu positiv).")


def horizon_chart(hz: pd.DataFrame, mode: str) -> go.Figure:
    p = PALETTE[mode]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hz["Horizont"], y=hz["Median ab Ausgabepreis"], name="Median ab Ausgabepreis",
                         marker=dict(color=p["series"][0], cornerradius=4),
                         hovertemplate="%{x}: %{y:+.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=hz["Horizont"], y=hz["Anteil besser als QQQ"], name="Anteil besser als QQQ",
                             mode="lines+markers", line=dict(color=p["series"][2]), yaxis="y2",
                             hovertemplate="%{x}: %{y:.0%} besser als QQQ<extra></extra>"))
    fig.update_layout(yaxis=dict(tickformat="+.0%", title="Median-Rendite"),
                      yaxis2=dict(tickformat=".0%", range=[0, 1], overlaying="y", side="right",
                                  title="besser als QQQ"))
    fig.add_hline(y=0, line=dict(color=p["text2"], width=1))
    return base_layout(fig, mode, "Nasdaq-IPOs: Median-Rendite und Anteil besser als der Nasdaq 100", height=380)


def monthly_chart(perf: pd.DataFrame, mode: str) -> go.Figure:
    p = PALETTE[mode]
    x = perf[perf["status"] == "priced"].copy()
    x["m"] = pd.to_datetime(x["date"]).dt.to_period("M").dt.to_timestamp()
    g = x.groupby("m").agg(n=("company", "size"), pop=("r_1 Tag", "median")).reset_index()
    fig = go.Figure(go.Bar(x=g["m"], y=g["n"], name="IPOs je Monat", marker_color=p["series"][0],
                           hovertemplate="%{x|%b %Y}: %{y} IPOs<extra></extra>"))
    fig.add_trace(go.Scatter(x=g["m"], y=g["pop"], name="Median 1. Tag", yaxis="y2",
                             line=dict(color=p["series"][2]), hovertemplate="Median 1. Tag %{y:+.0%}<extra></extra>"))
    fig.update_layout(yaxis=dict(title="Anzahl"), yaxis2=dict(tickformat="+.0%", overlaying="y", side="right",
                                                              title="Median 1. Tag"))
    return base_layout(fig, mode, "IPO-Aktivität an der Nasdaq (viele IPOs + hohe Erstzeichnungsgewinne = heisser "
                                  "Markt)", height=360)
