"""Technische Bestätigung der Warnung und einfache Ausstiegs-/Wiedereinstiegs-Simulation.

Idee: Die Warnung (Ampel) ist nur ein Filter. Gehandelt wird erst, wenn der Kurs technisch bestätigt
(Schlusskurs fällt unter den gleitenden Durchschnitt). Wiedereinstieg, sobald der Kurs wieder darüber schliesst.

Simulation: Handel zum Schlusskurs des Signaltags, ausserhalb des Marktes 0 % Zins, Kosten je Wechsel
einstellbar. Keine Steuern, kein Slippage über die Kosten hinaus. Keine Anlageberatung.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cross_below(close: pd.Series, ma: pd.Series) -> pd.Series:
    return (close < ma) & (close.shift(1) >= ma.shift(1))


def exit_signals(f: pd.DataFrame, ma_len: int = 50, ampel_min: int = 2, ampel_window: int = 20) -> dict[str, pd.Series]:
    c = f["close"]
    ma = c.rolling(ma_len, min_periods=ma_len).mean()
    ampel_recent = (f["ampel_score"] >= ampel_min).astype(int).rolling(ampel_window, min_periods=1).max().astype(bool)
    sma200 = c.rolling(200, min_periods=200).mean()
    return {
        f"Ampel + Schluss unter {ma_len}-Tage-Schnitt": ampel_recent & cross_below(c, ma),
        f"Nur Technik: Schluss unter {ma_len}-Tage-Schnitt": cross_below(c, ma),
        "Nur 200-Tage-Regel": cross_below(c, sma200),
    }


def debounce(sig: pd.Series, gap: int = 20) -> list[int]:
    out, last = [], -10_000
    for i in np.where(sig.fillna(False).values)[0]:
        if i - last >= gap:
            out.append(int(i))
            last = i
    return out


def signal_stats(f: pd.DataFrame, sig: pd.Series, episodes: pd.DataFrame, start: pd.Timestamp,
                 horizon: int = 63, threshold: float = -0.10) -> dict:
    c = f["close"].values
    idx = f.index
    pos = [i for i in debounce(sig) if idx[i] >= start]
    ep = episodes[pd.to_datetime(episodes["peak"]) >= start]
    caught, remaining = [], []
    for e in ep.itertuples():
        ip, it = idx.get_indexer([e.peak])[0], idx.get_indexer([e.trough])[0]
        inside = [i for i in pos if ip <= i <= it]
        if inside:
            caught.append(e.peak)
            remaining.append(c[it] / c[inside[0]] - 1)
    dd = [c[i + 1:i + 1 + horizon].min() / c[i] - 1 for i in pos if i + horizon < len(c)]
    r63 = [c[i + horizon] / c[i] - 1 for i in pos if i + horizon < len(c)]
    return {"signals": len(pos), "caught": len(caught), "episodes": len(ep),
            "remaining_drop": float(np.mean(remaining)) if remaining else np.nan,
            "hit_rate": float(np.mean([x <= threshold for x in dd])) if dd else np.nan,
            "median_ret_3m": float(np.median(r63)) if r63 else np.nan,
            "dates": [idx[i] for i in pos]}


def simulate(f: pd.DataFrame, exit_sig: pd.Series, reentry_ma: int = 50, cost: float = 0.001,
             start: pd.Timestamp | None = None) -> pd.DataFrame:
    """Investiert, bis ein Ausstiegssignal kommt; wieder investiert, wenn der Schluss über dem Durchschnitt liegt."""
    d = f.loc[f.index >= start] if start is not None else f
    c = d["close"].values
    ma = f["close"].rolling(reentry_ma, min_periods=reentry_ma).mean().reindex(d.index).values
    sig = exit_sig.reindex(d.index).fillna(False).values
    n = len(c)
    pos = np.ones(n)            # Position am ENDE von Tag i (gilt für die Rendite von Tag i+1)
    trades = np.zeros(n)
    inv = 1.0
    for i in range(n):
        if inv == 1.0 and sig[i]:
            inv, trades[i] = 0.0, 1
        elif inv == 0.0 and not np.isnan(ma[i]) and c[i] > ma[i]:
            inv, trades[i] = 1.0, 1
        pos[i] = inv
    ret = np.r_[0.0, c[1:] / c[:-1] - 1]
    strat = np.r_[0.0, pos[:-1]] * ret - trades * cost
    out = pd.DataFrame({"buy_hold": np.cumprod(1 + ret), "strategy": np.cumprod(1 + strat), "invested": pos,
                        "trade": trades}, index=d.index)
    return out


def perf_stats(eq: pd.Series, invested: pd.Series | None = None, trades: pd.Series | None = None) -> dict:
    eq = eq.dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    r = eq.pct_change().dropna()
    vol = r.std() * np.sqrt(252)
    out = {"Endwert (Start = 1)": eq.iloc[-1], "Rendite p. a.": cagr, "Max. Rückgang": dd, "Volatilität p. a.": vol,
           "Rendite/Risiko": cagr / vol if vol > 0 else np.nan}
    if invested is not None:
        out["Zeit investiert"] = invested.mean()
    if trades is not None:
        out["Wechsel"] = int(trades.sum())
    return out
