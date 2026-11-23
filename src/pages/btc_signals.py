"""BTC-Signale: Spot-Absorption (Preis fällt, Spot-Käufer kaufen) mit Live-Status, Chart und Backtest."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.components.ui import plot, theme_mode
from src.crypto import btc_signal
from src.crypto.btc_signal import Rule, backtest, compute, regime_runs, stats, stats_by_regime, status
from src.visualization.charts import PALETTE, STATUS, base_layout


@st.cache_data(ttl=300, show_spinner="Lade BTC-Kerzen von Binance …")
def load(days: int) -> pd.DataFrame:
    return btc_signal.fetch_klines("BTCUSDT", days=days)


def _mio(v: float) -> str:
    return f"{v / 1e6:+,.1f} Mio. $".replace(",", "'")


def page() -> None:
    st.title("BTC-Signale (Spot-Absorption)")
    st.caption("Long-Signal, wenn BTC in 24 Stunden um mehr als 3 % fällt, das Binance-Spot-CVD der letzten 24 Stunden "
               "aber positiv ist – Spot-Käufer nehmen den Abverkauf auf. Daten: Binance BTCUSDT Spot, 1h, "
               "alle 5 Minuten aktualisiert.")

    with st.expander("Einstellungen", expanded=False):
        c = st.columns(4)
        drop = c[0].number_input("Preisrückgang 24h grösser als (%)", 1.0, 10.0, 3.0, 0.5, key="btc_drop")
        tp = c[1].number_input("Take Profit (%)", 0.5, 15.0, 2.0, 0.5, key="btc_tp")
        sl = c[2].number_input("Stop Loss (%)", 0.5, 15.0, 6.0, 0.5, key="btc_sl")
        days = c[3].selectbox("Backtest-Zeitraum", [365, 730], index=1, key="btc_days",
                              format_func=lambda d: f"{d // 365} Jahr{'e' if d > 365 else ''}")
        bull_only = st.toggle("Nur Signale im Bull-Regime handeln (BTC über dem 200-Tage-Schnitt)", value=False,
                              key="btc_bull_only")
        st.caption("Standard (TP 2 % / SL 6 %): hohe Trefferquote, aber ein Verlust kostet drei Gewinne. "
                   "Alternative mit besserem Chance-Risiko-Verhältnis: TP 6 % / SL 4 %.")
    rule = Rule(drop_pct=drop, tp_pct=tp, sl_pct=sl, bull_only=bull_only)

    try:
        raw = load(days + rule.regime_days + 5)   # Vorlauf für den 200-Tage-Schnitt
    except Exception as exc:  # noqa: BLE001
        st.error(f"Binance-Daten konnten nicht geladen werden: {exc}")
        return
    if len(raw) < rule.window + 2:
        st.info("Noch zu wenige Kerzen für eine Auswertung.")
        return
    d = compute(raw, rule)
    d = d[d["ts"] >= d["ts"].iloc[-1] - pd.Timedelta(days=days)].reset_index(drop=True)
    s = status(d, rule)
    trades, open_trade = backtest(d, rule)
    st_ = stats(trades)

    # --- Live-Status ---
    t = s["ts"].tz_convert("Europe/Zurich") + pd.Timedelta(hours=1)
    c = st.columns(4)
    c[0].metric("BTC (Schluss letzte Stunde)", f"{s['close']:,.0f} $".replace(",", "'"),
                f"Stand {t:%d.%m. %H:%M}", delta_color="off")
    c[1].metric("Preisänderung 24h", f"{s['chg_pct']:+.2f} %",
                "Bedingung erfüllt" if s["price_ok"] else f"fehlt noch {s['missing_drop']:.2f} %",
                delta_color="normal" if s["price_ok"] else "off")
    c[2].metric("Spot-CVD 24h (Binance)", _mio(s["cvd"]),
                "Spot kauft netto" if s["cvd_ok"] else "Spot verkauft netto",
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
        st.info(f"Spot kauft bereits netto – das Signal kommt, falls BTC in 24h noch {s['missing_drop']:.2f} % "
                f"mehr fällt (unter ca. {s['close'] * (1 - s['missing_drop'] / 100):,.0f} $).".replace(",", "'"))
    else:
        st.caption("Kein Setup: Spot-Käufer fehlen. Ein Preisrückgang allein reicht nicht.")

    if open_trade:
        st.warning(f"Offener Signal-Trade seit {open_trade['Einstieg'].tz_convert('Europe/Zurich'):%d.%m. %H:%M}: "
                   f"Einstieg {open_trade['Einstiegspreis']:,.0f} $, aktuell {open_trade['Offen %']:+.2f} % · "
                   f"TP {open_trade['Take Profit']:,.0f} $ · SL {open_trade['Stop Loss']:,.0f} $".replace(",", "'"))

    # --- Backtest-Kennzahlen ---
    st.subheader("Backtest")
    c = st.columns(5)
    c[0].metric("Trades", st_["n"])
    c[1].metric("Trefferquote", f"{st_['winrate']:.0f} %" if st_["n"] else "–")
    c[2].metric("Ø pro Trade", f"{st_['avg']:+.2f} %" if st_["n"] else "–", "nach Gebühren", delta_color="off")
    c[3].metric("Summe aller Trades", f"{st_['total']:+.1f} %" if st_["n"] else "–")
    c[4].metric("Längste Verlustserie", st_["max_losses"])

    plot(signal_chart(d, trades, theme_mode()), "btc_sig_chart")

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

    with st.expander("Wie verlässlich ist das?"):
        st.markdown(
            "- Hyblock-Backtest (2 Jahre, TP 2 % / SL 6 %): 17 Trades, 82 % Treffer, nur leicht im Plus – "
            "ein Verlust (−6 %) frisst drei Gewinne (+2 %).\n"
            "- Mit TP 6 % / SL 4 %: 47 % Treffer, ebenfalls nur leicht positiv. Die Kante ist dünn.\n"
            "- Seit Mitte 2026 (Abwärtstrend) lief das Signal deutlich schlechter – darum die Regime-Färbung "
            "und der Schalter «Nur Signale im Bull-Regime».\n"
            "- Regime = Tagesschluss über/unter dem 200-Tage-Schnitt, gilt ab dem Folgetag.\n"
            "- Rund 8 Signale pro Jahr: zu wenige für eine verlässliche Statistik.\n"
            "- Die Werte hier können leicht von Hyblock abweichen (nur BTCUSDT auf Binance Spot, Delta aus "
            "Taker-Volumen).\n"
            "- Als Bestätigung oder Alarm nutzen, nicht als automatisches Handelssystem. Keine Anlageberatung.")


def signal_chart(d: pd.DataFrame, trades: pd.DataFrame, mode: str) -> go.Figure:
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
    fig.add_trace(go.Bar(x=x, y=cvd, name="Spot-CVD 24h (Mio. $)",
                         marker=dict(color=[STATUS["ok"] if v > 0 else STATUS["NA"] for v in cvd.fillna(0)],
                                     line=dict(width=0)),
                         hovertemplate="%{y:+,.1f} Mio. $<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="Mio. $", row=2, col=1)
    end = x.iloc[-1]
    fig.update_xaxes(range=[end - pd.Timedelta(days=90), end + pd.Timedelta(hours=6)],
                     rangeselector=dict(buttons=[dict(count=30, label="30T", step="day", stepmode="backward"),
                                                 dict(count=90, label="90T", step="day", stepmode="backward"),
                                                 dict(count=1, label="1J", step="year", stepmode="backward"),
                                                 dict(step="all", label="Alles")],
                                        bgcolor=p["surface"], font=dict(color=p["text2"])), row=1, col=1)
    fig = base_layout(fig, mode, "BTC mit Signalen – Hintergrund grün = Bull, rot = Bär (oben) · Spot-CVD 24h (unten)", height=560)
    fig.update_layout(hovermode="x unified", bargap=0)
    return fig
