"""BTC Spot-Absorption-Signal (aus der Hyblock-Analyse vom 23.11.2026).

Regel (Long): BTC fällt im Fenster (Standard 24 Stunden) um mehr als X % (Standard 3 %), aber das Binance-Spot-CVD
im selben Fenster ist positiv – echte Spot-Käufer nehmen den Abverkauf auf. Exit über festen Take Profit / Stop Loss.
Zeiteinheiten 15 Min, 1 Std und 4 Std; getestet (Hyblock, 2 Jahre) ist nur 1 Std mit 24h-Fenster.

Datenquelle: öffentliche Binance-Kerzen (BTCUSDT Spot) über data-api.binance.vision – kein API-Schlüssel
nötig, funktioniert auch auf Streamlit Cloud. Das Volumen-Delta wird aus dem Taker-Kaufvolumen berechnet
(Delta = Taker-Kauf − Taker-Verkauf, in USDT). Das entspricht Hyblocks "Volume Delta, Binance Spot, 0 – ∞".
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx
import numpy as np
import pandas as pd

KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
HOUR_MS = 3_600_000
TF_MIN = {"15m": 15, "1h": 60, "4h": 240}
# Startwerte je Zeiteinheit. Nur "1h" ist mit Hyblock gebacktestet; 15m/4h sind skaliert und im Backtest zu prüfen.
PRESETS = {
    "15m": {"label": "15 Min (kurzfristig)", "window_h": 4, "drop": 1.5, "tp": 1.0, "sl": 2.0,
            "days": 60, "days_opts": [30, 60, 90], "view_days": 7},
    "1h": {"label": "1 Std (Standard, getestet)", "window_h": 24, "drop": 3.0, "tp": 2.0, "sl": 6.0,
           "days": 730, "days_opts": [365, 730], "view_days": 90},
    "4h": {"label": "4 Std (Swing)", "window_h": 72, "drop": 6.0, "tp": 4.0, "sl": 8.0,
           "days": 1460, "days_opts": [730, 1460], "view_days": 365},
}
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"]


@dataclass(frozen=True)
class Rule:
    drop_pct: float = 3.0      # Preis muss im Fenster um mehr als so viel % fallen
    window_h: int = 24         # Fenster in Stunden für Preisänderung und CVD-Summe
    timeframe: str = "1h"      # Kerzengrösse: 15m, 1h oder 4h
    tp_pct: float = 2.0        # Take Profit in %
    sl_pct: float = 6.0        # Stop Loss in %
    fee_pct: float = 0.045     # Gebühr pro Seite in %
    regime_days: int = 200     # Regime: Tagesschluss über (Bull) / unter (Bär) dem gleitenden Schnitt
    bull_only: bool = False    # Signale nur im Bull-Regime handeln
    cvd_mode: str = "zero"     # "zero": CVD > 0 · "pct": CVD über seinem Perzentil der letzten `pct_days` Tage
    cvd_pct: float = 80.0      # Perzentil-Schwelle für cvd_mode="pct"
    pct_days: int = 90

    @property
    def bar_min(self) -> int:
        return TF_MIN[self.timeframe]

    @property
    def window(self) -> int:
        """Fenster in Kerzen."""
        return max(1, round(self.window_h * 60 / self.bar_min))


def fetch_klines(symbol: str = "BTCUSDT", days: int = 730, client: httpx.Client | None = None,
                 now_ms: int | None = None, interval: str = "1h") -> pd.DataFrame:
    """Lädt Kerzen (`interval`: 15m, 1h, 4h, 1d) der letzten `days` Tage (1000 Kerzen pro Anfrage)."""
    step = {"15m": 900_000, "1h": HOUR_MS, "4h": 4 * HOUR_MS, "1d": 24 * HOUR_MS}[interval]
    now_ms = now_ms or int(time.time() * 1000)
    cur = now_ms - days * 24 * HOUR_MS
    own = client is None
    client = client or httpx.Client(timeout=20)
    rows: list = []
    try:
        while cur < now_ms:
            r = client.get(KLINES_URL, params={"symbol": symbol, "interval": interval, "startTime": cur,
                                               "limit": 1000})
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            cur = int(batch[-1][0]) + step
            if len(batch) < 1000:
                break
    finally:
        if own:
            client.close()
    return to_frame(rows, now_ms)


def to_frame(rows: list, now_ms: int | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLS)
    if df.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "quote_volume", "delta", "closed"])
    for c in ["open", "high", "low", "close", "quote_volume", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["delta"] = 2 * df["taker_buy_quote"] - df["quote_volume"]
    now_ms = now_ms or int(time.time() * 1000)
    df["closed"] = df["close_time"].astype("int64") < now_ms
    return df[["ts", "open", "high", "low", "close", "quote_volume", "delta", "closed"]]


def compute(df: pd.DataFrame, rule: Rule = Rule(), daily: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fügt Preisänderung und Spot-CVD im Fenster, das Regime und das Signal hinzu (nur abgeschlossene Kerzen lösen
    aus). `daily`: Tageskerzen für das Regime; ohne sie wird aus den vorhandenen Kerzen auf Tage verdichtet."""
    out = df.copy()
    w = rule.window
    out["chg_pct"] = (out["close"] / out["close"].shift(w) - 1) * 100
    out["cvd"] = out["delta"].rolling(w).sum()
    out["cvd_rank"] = cvd_rank(out["cvd"], rule)
    out = add_regime(out, rule.regime_days, daily)
    out["cvd_ok"] = out["cvd"] > 0 if rule.cvd_mode == "zero" else out["cvd_rank"] > rule.cvd_pct
    out["signal"] = (out["chg_pct"] < -rule.drop_pct) & out["cvd_ok"] & out["closed"]
    if rule.bull_only:
        out["signal"] &= out["regime"] == "Bull"
    return out


def cvd_rank(cvd: pd.Series, rule: Rule) -> pd.Series:
    """Perzentil (0–100) des aktuellen CVD unter den Werten der letzten `pct_days` Tage."""
    n = max(10, round(rule.pct_days * 24 * 60 / rule.bar_min))
    return cvd.rolling(n, min_periods=max(10, n // 4)).rank(pct=True) * 100


WINDOWS_H = (4, 24, 72, 168)


def multi_window(df: pd.DataFrame, rule: Rule = Rule(), windows=WINDOWS_H) -> pd.DataFrame:
    """Preisänderung, Spot-CVD und CVD-Perzentil für mehrere Fenster gleichzeitig (letzte abgeschlossene Kerze)."""
    d = df[df["closed"]]
    rows = []
    for h in windows:
        r = Rule(timeframe=rule.timeframe, window_h=h, pct_days=rule.pct_days)
        w = r.window
        if len(d) <= w:
            continue
        cvd = d["delta"].rolling(w).sum()
        rank = cvd_rank(cvd, r)
        rows.append({"Fenster": f"{h // 24} T" if h >= 48 else f"{h} h",
                     "Preisänderung %": (d["close"].iloc[-1] / d["close"].iloc[-1 - w] - 1) * 100,
                     "Spot-CVD Mio. $": cvd.iloc[-1] / 1e6,
                     f"CVD-Perzentil ({rule.pct_days} T)": rank.iloc[-1]})
    return pd.DataFrame(rows)


def add_regime(d: pd.DataFrame, days: int = 200, daily_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Bull/Bär aus dem Tagesschluss gegenüber dem gleitenden Durchschnitt über `days` Tage.
    Nur abgeschlossene Tage zählen; der Wert gilt ab dem Folgetag (kein Blick in die Zukunft)."""
    out = d.copy()
    src = daily_df if daily_df is not None and len(daily_df) else out
    daily = src.set_index("ts")["close"].resample("1D").last().dropna()
    sma = daily.rolling(days).mean()
    reg = pd.Series(np.where(daily > sma, "Bull", "Bär"), index=daily.index).where(sma.notna())
    reg.index = reg.index + pd.Timedelta(days=1)          # gilt erst ab dem nächsten Tag
    day = out["ts"].dt.floor("D")
    out["regime"] = day.map(reg).astype("object")
    out["sma_regime"] = day.map(pd.Series(sma.to_numpy(), index=sma.index + pd.Timedelta(days=1)))
    return out


def regime_runs(d: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Zusammenhängende Abschnitte gleichen Regimes (für die Hintergrundfärbung im Chart)."""
    r = d[["ts", "regime"]].dropna()
    if r.empty:
        return []
    bar = r["ts"].diff().median() if len(r) > 1 else pd.Timedelta(hours=1)
    grp = (r["regime"] != r["regime"].shift()).cumsum()
    return [(g["ts"].iloc[0], g["ts"].iloc[-1] + bar, g["regime"].iloc[0])
            for _, g in r.groupby(grp)]


def backtest(d: pd.DataFrame, rule: Rule = Rule()) -> tuple[pd.DataFrame, dict | None]:
    """Eine Position gleichzeitig, Einstieg zum Schlusskurs der Signalkerze, danach TP/SL über Hoch/Tief.
    Berührt eine Kerze beide Marken, zählt konservativ der Stop. Rückgabe: (abgeschlossene Trades, offener Trade)."""
    hi, lo, cl = d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy()
    sig, ts = d["signal"].to_numpy(), d["ts"].to_numpy()
    reg = d["regime"].to_numpy() if "regime" in d else np.full(len(d), None)
    fees = 2 * rule.fee_pct
    trades, i, n = [], 0, len(d)
    open_trade = None
    while i < n:
        if not sig[i]:
            i += 1
            continue
        entry = cl[i]
        tp, sl = entry * (1 + rule.tp_pct / 100), entry * (1 - rule.sl_pct / 100)
        exit_j, reason = None, None
        for j in range(i + 1, n):
            if lo[j] <= sl:
                exit_j, reason = j, "Stop Loss"
                break
            if hi[j] >= tp:
                exit_j, reason = j, "Take Profit"
                break
        if exit_j is None:
            open_trade = {"Einstieg": pd.Timestamp(ts[i]), "Einstiegspreis": entry, "Take Profit": tp,
                          "Stop Loss": sl, "Aktuell": cl[-1], "Offen %": (cl[-1] / entry - 1) * 100}
            break
        gross = rule.tp_pct if reason == "Take Profit" else -rule.sl_pct
        trades.append({"Einstieg": pd.Timestamp(ts[i]), "Regime": reg[i] if isinstance(reg[i], str) else "–",
                       "Ausstieg": pd.Timestamp(ts[exit_j]),
                       "Einstiegspreis": entry, "Ausstiegspreis": tp if reason == "Take Profit" else sl,
                       "Grund": reason, "Dauer h": (exit_j - i) * rule.bar_min / 60,
                       "PnL %": gross - fees})
        i = exit_j + 1
    return pd.DataFrame(trades), open_trade


def stats_by_regime(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ["Bull", "Bär"]:
        t = trades[trades["Regime"] == name] if len(trades) else trades
        s = stats(t)
        rows.append({"Regime": name, "Trades": s["n"], "Trefferquote %": s["winrate"], "Ø pro Trade %": s["avg"],
                     "Summe %": s["total"]})
    return pd.DataFrame(rows)


def stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "winrate": np.nan, "avg": np.nan, "total": np.nan, "pf": np.nan, "max_losses": 0}
    p = trades["PnL %"]
    wins, losses = p[p > 0], p[p <= 0]
    streak = best = 0
    for v in p:
        streak = streak + 1 if v <= 0 else 0
        best = max(best, streak)
    return {"n": len(p), "winrate": len(wins) / len(p) * 100, "avg": p.mean(), "total": p.sum(),
            "pf": wins.sum() / -losses.sum() if len(losses) and losses.sum() < 0 else np.inf,
            "max_losses": best}


def status(d: pd.DataFrame, rule: Rule = Rule()) -> dict:
    """Lage der letzten abgeschlossenen Kerze: wie weit ist das Signal entfernt?"""
    last = d[d["closed"]].iloc[-1]
    price_ok = bool(last["chg_pct"] < -rule.drop_pct)
    cvd_ok = bool(last["cvd_ok"]) if "cvd_ok" in last else bool(last["cvd"] > 0)
    return {"ts": last["ts"], "close": float(last["close"]), "regime": last.get("regime"),
            "sma_regime": last.get("sma_regime"), "chg_pct": float(last["chg_pct"]),
            "cvd": float(last["cvd"]), "cvd_rank": last.get("cvd_rank"), "price_ok": price_ok, "cvd_ok": cvd_ok, "active": price_ok and cvd_ok,
            "missing_drop": max(0.0, float(last["chg_pct"]) + rule.drop_pct)}
