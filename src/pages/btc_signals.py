"""BTC-Signale: Spot-Absorption (Preis fällt, Spot-Käufer kaufen) mit Live-Status, Chart und Backtest."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.components.ui import plot, theme_mode
from src.crypto import btc_signal
from src.crypto.btc_signal import (Rule, backtest, compute, multi_window, regime_runs, stats, stats_by_regime,
                                   status)
from src.crypto.hyblock_strategies import scan_frame, strategies_frame
from src.visualization.charts import PALETTE, STATUS, base_layout


@st.cache_data(ttl=60, show_spinner="Lade BTC-Kerzen von Binance …")
def load(interval: str, days: int) -> pd.DataFrame:
    return btc_signal.fetch_klines("BTCUSDT", days=days, interval=interval)


@st.cache_data(ttl=3600, show_spinner=False)
def load_daily(days: int) -> pd.DataFrame:
    return btc_signal.fetch_klines("BTCUSDT", days=days, interval="1d")


def _mio(v: float) -> str:
    return f"{v / 1e6:+,.1f} Mio. $".replace(",", "'")


def _win(h: int) -> str:
    return f"{h // 24} T" if h >= 48 and h % 24 == 0 else f"{h} h"


def _days(d: int) -> str:
    return f"{d // 365} Jahr{'e' if d >= 730 else ''}" if d >= 365 else f"{d} Tage"


def page() -> None:
    st.title("BTC-Signale (Spot-Absorption)")
    tf = st.segmented_control("Zeiteinheit", list(btc_signal.PRESETS), default="1h", key="btc_tf",
                              format_func=lambda k: btc_signal.PRESETS[k]["label"]) or "1h"
    pr = btc_signal.PRESETS[tf]
    st.caption("Long-Signal, wenn BTC im Fenster deutlich fällt, das Binance-Spot-CVD im selben Fenster aber positiv "
               "ist – Spot-Käufer nehmen den Abverkauf auf. Daten: Binance BTCUSDT Spot, jede Minute aktualisiert.")
    if tf != "1h":
        st.warning("Getestet ist nur «1 Std» mit 24h-Fenster. Die Startwerte für 15 Min und 4 Std sind hochgerechnet – "
                   "prüf sie im Backtest unten, bevor du danach handelst. Kurzfristig fressen Gebühren und "
                   "Rauschen einen grossen Teil der Bewegung.")

    with st.expander("Einstellungen", expanded=False):
        c = st.columns(5)
        win_h = c[0].number_input("Fenster (Stunden)", 1, 240, pr["window_h"], 1, key=f"btc_win_{tf}")
        drop = c[1].number_input("Preisrückgang grösser als (%)", 0.3, 15.0, pr["drop"], 0.1, key=f"btc_drop_{tf}")
        tp = c[2].number_input("Take Profit (%)", 0.2, 20.0, pr["tp"], 0.1, key=f"btc_tp_{tf}")
        sl = c[3].number_input("Stop Loss (%)", 0.2, 20.0, pr["sl"], 0.1, key=f"btc_sl_{tf}")
        days = c[4].selectbox("Backtest-Zeitraum", pr["days_opts"], index=pr["days_opts"].index(pr["days"]),
                              key=f"btc_days_{tf}", format_func=_days)
        c = st.columns([2, 1])
        cvd_mode = c[0].radio("CVD-Bedingung", ["zero", "pct"], horizontal=True, key="btc_cvd_mode",
                              format_func=lambda m: "CVD > 0 (streng)" if m == "zero" else
                              "CVD über eigenem Normalwert (Perzentil, reagiert früher)")
        cvd_pct = c[1].number_input("Perzentil-Schwelle (90 Tage)", 50, 99, 80, 5, key="btc_cvd_pct",
                                    disabled=cvd_mode == "zero")
        c = st.columns([3, 1])
        bull_only = c[0].toggle("Nur Signale im Bull-Regime handeln (BTC über dem 200-Tage-Schnitt)", value=False,
                                key="btc_bull_only")
        fee = c[1].number_input("Gebühr pro Seite (%)", 0.0, 0.5, 0.045, 0.005, format="%.3f", key="btc_fee")
        st.caption("1 Std Standard (TP 2 % / SL 6 %): hohe Trefferquote, aber ein Verlust kostet drei Gewinne. "
                   "Alternative mit besserem Chance-Risiko-Verhältnis: TP 6 % / SL 4 %.")
    rule = Rule(drop_pct=drop, window_h=int(win_h), timeframe=tf, tp_pct=tp, sl_pct=sl, fee_pct=fee,
                bull_only=bull_only, cvd_mode=cvd_mode, cvd_pct=float(cvd_pct))
    w = _win(rule.window_h)

    try:
        raw = load(tf, days + 3)
        daily = load_daily(days + rule.regime_days + 10)   # Tageskerzen für das 200-Tage-Regime
    except Exception as exc:  # noqa: BLE001
        st.error(f"Binance-Daten konnten nicht geladen werden: {exc}")
        return
    if len(raw) < rule.window + 2:
        st.info("Noch zu wenige Kerzen für eine Auswertung.")
        return
    d = compute(raw, rule, daily)
    d = d[d["ts"] >= d["ts"].iloc[-1] - pd.Timedelta(days=days)].reset_index(drop=True)
    s = status(d, rule)
    trades, open_trade = backtest(d, rule)
    st_ = stats(trades)

    # --- Live-Status ---
    t = s["ts"].tz_convert("Europe/Zurich") + pd.Timedelta(minutes=rule.bar_min)
    c = st.columns(4)
    c[0].metric(f"BTC (Schluss letzte {'Kerze' if tf != '1h' else 'Stunde'})", f"{s['close']:,.0f} $".replace(",", "'"),
                f"Stand {t:%d.%m. %H:%M}", delta_color="off")
    c[1].metric(f"Preisänderung {w}", f"{s['chg_pct']:+.2f} %",
                "Bedingung erfüllt" if s["price_ok"] else f"fehlt noch {s['missing_drop']:.2f} %",
                delta_color="normal" if s["price_ok"] else "off")
    c[2].metric(f"Spot-CVD {w} (Binance)", _mio(s["cvd"]),
                (f"Perzentil {s['cvd_rank']:.0f} (90 T)" if rule.cvd_mode == "pct" and s.get("cvd_rank") == s.get("cvd_rank")
                 else "Spot kauft netto" if s["cvd"] > 0 else "Spot verkauft netto"),
                delta_color="normal" if s["cvd_ok"] else "inverse")
    c[3].metric("Signal", "AKTIV" if s["active"] else "kein Signal",
                "beide Bedingungen erfüllt" if s["active"] else
                f"{int(s['price_ok']) + int(s['cvd_ok'])} von 2 Bedingungen", delta_color="off")

    reg = s.get("regime")
    if isinstance(reg, str):
        sma = s.get("sma_regime")
        txt = (f"**Marktregime: {'🟢 Bullish' if reg == 'Bull' else '🔴 Bärisch'}** – BTC liegt "
               f"{'über' if reg == 'Bull' else 'unter'} dem 200-Tage-Schnitt"
               + (f" ({sma:,.0f} $)".replace(",", "'") if sma == sma and sma is not None else "") + ".")
        (st.success if reg == "Bull" else st.error)(txt + " Wie das Signal je Regime abgeschnitten hat, zeigt die Tabelle unter dem Chart.")

    if s["active"]:
        st.success(f"Long-Signal aktiv: Einstieg um {s['close']:,.0f} $, Take Profit "
                   f"{s['close'] * (1 + tp / 100):,.0f} $, Stop Loss {s['close'] * (1 - sl / 100):,.0f} $."
                   .replace(",", "'"))
    elif s["cvd_ok"]:
        st.info(f"CVD-Bedingung erfüllt – das Signal kommt, falls BTC im {w}-Fenster noch {s['missing_drop']:.2f} % "
                f"mehr fällt (unter ca. {s['close'] * (1 - s['missing_drop'] / 100):,.0f} $).".replace(",", "'"))
    else:
        st.caption("Kein Setup: Spot-Käufer fehlen. Ein Preisrückgang allein reicht nicht.")

    if open_trade:
        st.warning(f"Offener Signal-Trade seit {open_trade['Einstieg'].tz_convert('Europe/Zurich'):%d.%m. %H:%M}: "
                   f"Einstieg {open_trade['Einstiegspreis']:,.0f} $, aktuell {open_trade['Offen %']:+.2f} % · "
                   f"TP {open_trade['Take Profit']:,.0f} $ · SL {open_trade['Stop Loss']:,.0f} $".replace(",", "'"))

    # --- Mehrere Fenster ---
    st.subheader("Spot-CVD über mehrere Fenster")
    mw = multi_window(raw, rule)
    if len(mw):
        pc = [c for c in mw.columns if c.startswith("CVD-Perzentil")][0]
        st.dataframe(mw, hide_index=True, width="stretch",
                     column_config={"Preisänderung %": st.column_config.NumberColumn(format="%+.2f %%"),
                                    "Spot-CVD Mio. $": st.column_config.NumberColumn(format="%+,.1f"),
                                    pc: st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")})
        st.caption("Perzentil = wo das CVD heute im Vergleich zu den letzten 90 Tagen steht (100 = stärkster "
                   "Spot-Kaufdruck). Fallender Preis bei hohem Perzentil = Spot-Käufer nehmen den Abverkauf auf.")

    # --- Backtest-Kennzahlen ---
    st.subheader("Backtest")
    c = st.columns(5)
    c[0].metric("Trades", st_["n"])
    c[1].metric("Trefferquote", f"{st_['winrate']:.0f} %" if st_["n"] else "–")
    c[2].metric("Ø pro Trade", f"{st_['avg']:+.2f} %" if st_["n"] else "–", "nach Gebühren", delta_color="off")
    c[3].metric("Summe aller Trades", f"{st_['total']:+.1f} %" if st_["n"] else "–")
    c[4].metric("Längste Verlustserie", st_["max_losses"])

    plot(signal_chart(d, trades, theme_mode(), w, pr["view_days"]), f"btc_sig_chart_{tf}")

    if len(trades):
        st.markdown("**Ergebnis nach Marktregime beim Einstieg**")
        st.dataframe(stats_by_regime(trades), hide_index=True, width="stretch",
                     column_config={"Trefferquote %": st.column_config.NumberColumn(format="%.0f %%"),
                                    "Ø pro Trade %": st.column_config.NumberColumn(format="%+.2f %%"),
                                    "Summe %": st.column_config.NumberColumn(format="%+.1f %%")})

    if len(trades):
        show = trades.sort_values("Einstieg", ascending=False).copy()
        for col in ["Einstieg", "Ausstieg"]:
            show[col] = show[col].dt.tz_convert("Europe/Zurich").dt.strftime("%d.%m.%Y %H:%M")
        st.dataframe(show, hide_index=True, width="stretch",
                     column_config={"Einstiegspreis": st.column_config.NumberColumn(format="%.0f"),
                                    "Ausstiegspreis": st.column_config.NumberColumn(format="%.0f"),
                                    "PnL %": st.column_config.NumberColumn(format="%+.2f %%")})

    st.subheader("Hyblock-Strategien (Tier 3, getestet 23.11.2026)")
    st.caption("Mit Hyblock Capital über 2 Jahre gebacktestet (BTC, Binance-Perp, Gebühren 0,045 % pro Seite). "
               "Jahr 2 ist die Kontrolle auf neueren Daten. Diese Indikatoren gibt es nur bei Hyblock – live laufen die "
               "Strategien dort als Alarm, nicht hier im Dashboard.")
    st.dataframe(strategies_frame(), hide_index=True, width="stretch",
                 column_config={"Summe %": st.column_config.NumberColumn(format="%+.1f %%"),
                                "Treffer %": st.column_config.NumberColumn(format="%.0f %%"),
                                "Regel": st.column_config.TextColumn(width="large")})
    with st.expander("Alle getesteten Indikatoren (Signal-Statistik, 4 Std, 2 Jahre)"):
        st.dataframe(scan_frame(), hide_index=True, width="stretch",
                     column_config={"Treffer nach 3 T %": st.column_config.NumberColumn(format="%.1f %%")})
        st.caption("Vergleich: Über alle 4-Std-Kerzen stand BTC nach 3 Tagen in 49,6 % der Fälle höher. "
                   "Bei Shorts zählt ein tieferer Kurs als Treffer. Signale überlappen sich – die Zahlen zeigen "
                   "eine Tendenz, keine Trade-Statistik.")

    with st.expander("Wie verlässlich ist das?"):
        st.markdown(
            "- Hyblock-Backtest (2 Jahre, TP 2 % / SL 6 %): 17 Trades, 82 % Treffer, nur leicht im Plus – "
            "ein Verlust (−6 %) frisst drei Gewinne (+2 %).\n"
            "- Mit TP 6 % / SL 4 %: 47 % Treffer, ebenfalls nur leicht positiv. Die Kante ist dünn.\n"
            "- Seit Mitte 2026 (Abwärtstrend) lief das Signal deutlich schlechter – darum die Regime-Färbung "
            "und der Schalter «Nur Signale im Bull-Regime».\n"
            "- Regime = Tagesschluss über/unter dem 200-Tage-Schnitt, gilt ab dem Folgetag.\n"
            "- Rund 8 Signale pro Jahr (1 Std): zu wenige für eine verlässliche Statistik.\n"
            "- 15 Min / 4 Std: nicht mit Hyblock getestet – nur der Backtest auf dieser Seite (Binance-Daten). "
            "Kurzfristig zählen Gebühren und Slippage stark; der Backtest rechnet mit Einstieg zum Schlusskurs "
            "ohne Slippage.\n"
            "- Die Werte hier können leicht von Hyblock abweichen (nur BTCUSDT auf Binance Spot, Delta aus "
            "Taker-Volumen).\n"
            "- Als Bestätigung oder Alarm nutzen, nicht als automatisches Handelssystem. Keine Anlageberatung.")


def signal_chart(d: pd.DataFrame, trades: pd.DataFrame, mode: str, win: str = "24 h",
                 view_days: int = 90) -> go.Figure:
    p = PALETTE[mode]
    x = d["ts"].dt.tz_convert("Europe/Zurich")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=x, y=d["close"], name="BTC", line=dict(color=p["series"][0], width=1.5),
                             hovertemplate="%{y:,.0f} $<extra></extra>"), row=1, col=1)
    if "sma_regime" in d and d["sma_regime"].notna().any():
        fig.add_trace(go.Scatter(x=x, y=d["sma_regime"], name="200-Tage-Schnitt",
                                 line=dict(color=p["muted"], width=1, dash="dot"),
                                 hovertemplate="200T-Schnitt %{y:,.0f} $<extra></extra>"), row=1, col=1)
    for start, end, reg in regime_runs(d):
        fig.add_vrect(x0=start.tz_convert("Europe/Zurich"), x1=end.tz_convert("Europe/Zurich"),
                      fillcolor=STATUS["ok"] if reg == "Bull" else STATUS["NA"], opacity=0.08, line_width=0,
                      layer="below")
    sig = d[d["signal"]]
    fig.add_trace(go.Scatter(x=sig["ts"].dt.tz_convert("Europe/Zurich"), y=sig["close"], mode="markers",
                             name="Signal", marker=dict(symbol="triangle-up", size=10, color=STATUS["ok"]),
                             hovertemplate="Signal %{y:,.0f} $<extra></extra>"), row=1, col=1)
    if len(trades):
        for reason, color, sym in (("Take Profit", STATUS["ok"], "circle"), ("Stop Loss", STATUS["NA"], "x")):
            t = trades[trades["Grund"] == reason]
            if len(t):
                fig.add_trace(go.Scatter(x=t["Ausstieg"].dt.tz_convert("Europe/Zurich"), y=t["Ausstiegspreis"],
                                         mode="markers", name=reason,
                                         marker=dict(symbol=sym, size=8, color=color),
                                         hovertemplate=f"{reason} %{{y:,.0f}} $<extra></extra>"), row=1, col=1)
    cvd = d["cvd"] / 1e6
    fig.add_trace(go.Bar(x=x, y=cvd, name=f"Spot-CVD {win} (Mio. $)",
                         marker=dict(color=[STATUS["ok"] if v > 0 else STATUS["NA"] for v in cvd.fillna(0)],
                                     line=dict(width=0)),
                         hovertemplate="%{y:+,.1f} Mio. $<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="Mio. $", row=2, col=1)
    end = x.iloc[-1]
    fig.update_xaxes(range=[end - pd.Timedelta(days=view_days), end + pd.Timedelta(days=view_days / 60)],
                     rangeselector=dict(buttons=[dict(count=1, label="1T", step="day", stepmode="backward"),
                                                 dict(count=7, label="7T", step="day", stepmode="backward"),
                                                 dict(count=30, label="30T", step="day", stepmode="backward"),
                                                 dict(count=90, label="90T", step="day", stepmode="backward"),
                                                 dict(count=1, label="1J", step="year", stepmode="backward"),
                                                 dict(step="all", label="Alles")],
                                        bgcolor=p["surface"], font=dict(color=p["text2"]),
                                        x=0, xanchor="left", y=1.01, yanchor="bottom"), row=1, col=1)
    fig = base_layout(fig, mode, f"BTC mit Signalen – Hintergrund grün = Bull, rot = Bär (oben) · Spot-CVD {win} (unten)", height=560)
    # Legende unter den Chart, damit sie nicht mit den Zoom-Knöpfen oben kollidiert
    fig.update_layout(hovermode="x unified", bargap=0, margin=dict(t=70, b=60),
                      legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="left", x=0))
    return fig
