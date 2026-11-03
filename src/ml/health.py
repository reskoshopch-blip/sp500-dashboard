"""Markt-Gesundheitsindex: kurz-, mittel- und langfristig.

Jeder Indikator wird in einen Score 0–100 übersetzt: sein Perzentil im Vergleich zu den letzten 5 Jahren
(rollierendes Fenster, mindestens 1 Jahr Historie). Das Perzentil nutzt nur Werte bis zum jeweiligen Tag –
kein Blick in die Zukunft. Bei Indikatoren, bei denen hohe Werte schlecht sind (z. B. VIX), wird umgedreht.
Der Gesamtscore eines Horizonts ist der Durchschnitt seiner verfügbaren Indikatoren (mindestens 3).

  0–20 sehr schwach · 20–40 schwach · 40–60 neutral · 60–80 gesund · 80–100 sehr gesund

Der historische Test misst, wie sich der Index nach Tagen in jedem Score-Bereich entwickelte
(kurz: 10 Handelstage, mittel: 63, lang: 252).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOW, MIN_HIST = 1260, 252
HORIZONS = {"kurz": ("Kurzfristig", "Tage bis Wochen", 10), "mittel": ("Mittelfristig", "Wochen bis Monate", 63),
            "lang": ("Langfristig", "Monate bis Jahre", 252)}
LEVELS = [(0, 20, "sehr schwach"), (20, 40, "schwach"), (40, 60, "neutral"), (60, 80, "gesund"), (80, 101, "sehr gesund")]


@dataclass(frozen=True)
class Comp:
    key: str
    label: str
    horizon: str
    category: str
    col: str | None        # Spalte im Tagesdatensatz (oder None: wird in compute() berechnet)
    good_high: bool
    fmt: str               # pct | num | pp
    explain: str


COMPONENTS: tuple[Comp, ...] = (
    # ------------------------------------------------------------ kurzfristig
    Comp("k_trend20", "Index vs. 20-Tage-Schnitt", "kurz", "Trend", None, True, "pct",
         "Liegt der Index über seinem kurzen Durchschnitt?"),
    Comp("k_mom10", "Index-Rendite 10 Tage", "kurz", "Trend", None, True, "pct", "Kurzfristiges Momentum"),
    Comp("k_breadth20", "Aktien über 20-Tage-Schnitt", "kurz", "Marktbreite", "breadth_20", True, "pct",
         "Anteil der Aktien im kurzfristigen Aufwärtstrend"),
    Comp("k_mcclellan", "McClellan-Oszillator", "kurz", "Marktbreite", "mcclellan", True, "num",
         "Schwung der Netto-Gewinner (positiv = mehr Aktien steigen als fallen)"),
    Comp("k_div", "Tages-Divergenzen (20 Tage)", "kurz", "Marktbreite", "div_days_20", False, "num",
         "Tage, an denen der Index ≥ 0.5 % fiel, obwohl die Mehrheit der Aktien stieg"),
    Comp("k_vol", "Volatilität vs. 20-Tage-Schnitt", "kurz", "Volatilität", None, False, "pct",
         "Steigt die Nervosität gerade (VIX bzw. realisierte Volatilität)?"),
    Comp("k_vixterm", "VIX-Terminstruktur (VIX/VIX3M)", "kurz", "Volatilität", "vix_term", False, "num",
         "Über 1 = kurzfristige Angst grösser als langfristige (Stress)"),
    # ------------------------------------------------------------ mittelfristig
    Comp("m_trend50", "Index vs. 50-Tage-Schnitt", "mittel", "Trend", "spy_vs_sma50", True, "pct", "Mittelfristiger Trend"),
    Comp("m_mom63", "Index-Rendite 3 Monate", "mittel", "Trend", "spy_mom_63", True, "pct", "Momentum über 3 Monate"),
    Comp("m_breadth50", "Aktien über 50-Tage-Schnitt", "mittel", "Marktbreite", "breadth_50", True, "pct",
         "Anteil der Aktien im mittelfristigen Aufwärtstrend"),
    Comp("m_ad", "Advance-Decline-Linie 3 Monate", "mittel", "Marktbreite", "ad_63", True, "num",
         "Steigen über 3 Monate mehr Aktien als fallen? (je Aktie normiert)"),
    Comp("m_hilo", "Neue Hochs minus neue Tiefs", "mittel", "Marktbreite", "hi_lo", True, "pct",
         "Anteil Aktien auf 52-Wochen-Hoch minus Anteil auf Tief"),
    Comp("m_ew", "Gleichgewichtet vs. Index (RSP/SPY)", "mittel", "Konzentration", "ew_vs_cap_63", True, "pct",
         "Negativ = wenige Schwergewichte tragen den Index"),
    Comp("m_small", "Small Caps vs. Index (IWM/SPY)", "mittel", "Risikoappetit", "small_vs_large_63", True, "pct",
         "Kleine Firmen stärker = breiter Risikoappetit"),
    Comp("m_cyc", "Zyklisch vs. defensiv (XLY/XLP)", "mittel", "Risikoappetit", "cyc_vs_def_63", True, "pct",
         "Zyklischer Konsum stärker als Basiskonsum = Optimismus"),
    Comp("m_credit", "Kreditmarkt (HYG/IEF vs. 50T)", "mittel", "Risikoappetit", "credit_vs_sma50", True, "pct",
         "Hochzinsanleihen stärker als Staatsanleihen = Risikoappetit"),
    Comp("m_fx", "USD/JPY 1 Monat (Japan)", "mittel", "Währung", "fx_mom_20", True, "pct",
         "Schwacher Yen stützt japanische Exporteure; starker Yen belastet"),
    # ------------------------------------------------------------ langfristig
    Comp("l_trend200", "Index vs. 200-Tage-Schnitt", "lang", "Trend", "spy_vs_sma200", True, "pct", "Langfristiger Trend"),
    Comp("l_mom252", "Index-Rendite 12 Monate", "lang", "Trend", "spy_mom_252", True, "pct", "Momentum über 12 Monate"),
    Comp("l_breadth200", "Aktien über 200-Tage-Schnitt", "lang", "Marktbreite", "breadth_200", True, "pct",
         "Anteil der Aktien im langfristigen Aufwärtstrend"),
    Comp("l_below20", "Aktien ≥ 20 % unter 52W-Hoch", "lang", "Marktbreite", "pct_below20", False, "pct",
         "Anteil der Aktien im eigenen Bärenmarkt"),
    Comp("l_dist", "Abstand zum 52-Wochen-Hoch", "lang", "Trend", "spy_dist_high", True, "pct", "Wie weit vom Hoch?"),
    Comp("l_credit", "Kreditmarkt 12 Monate", "lang", "Risikoappetit", "credit_mom_252", True, "pct",
         "Langfristiger Trend Hochzins vs. Staatsanleihen"),
    Comp("l_curve", "Zinskurve (10 J. − 3 M.)", "lang", "Makro", "yield_curve", True, "pp",
         "Inverse Zinskurve (negativ) ging vielen Rezessionen voraus"),
    Comp("l_copper", "Kupfer/Gold 6 Monate", "lang", "Makro", "copper_gold_126", True, "pct",
         "Kupfer stärker als Gold = Wachstumserwartung"),
)
COMP_BY_KEY = {c.key: c for c in COMPONENTS}


def level(score: float) -> str:
    if score is None or pd.isna(score):
        return "keine Daten"
    for lo, hi, name in LEVELS:
        if lo <= score < hi:
            return name
    return "sehr gesund"


def raw_values(f: pd.DataFrame) -> pd.DataFrame:
    c = f["close"]
    out = pd.DataFrame(index=f.index)
    for comp in COMPONENTS:
        if comp.col is not None:
            out[comp.key] = f[comp.col] if comp.col in f else np.nan
    out["k_trend20"] = c / c.rolling(20, min_periods=20).mean() - 1
    out["k_mom10"] = c / c.shift(10) - 1
    v = f["vix"] if f["vix"].notna().any() else f["spy_vol_21"]
    out["k_vol"] = v / v.rolling(20, min_periods=15).mean() - 1
    return out


def percentile_scores(raw: pd.DataFrame) -> pd.DataFrame:
    """Rollierendes Perzentil (nur Vergangenheit) -> 0..100, 'gut' = hoch."""
    sc = pd.DataFrame(index=raw.index)
    for comp in COMPONENTS:
        x = raw[comp.key]
        if x.notna().sum() < MIN_HIST:
            continue
        r = x.rolling(WINDOW, min_periods=MIN_HIST).rank(pct=True)
        sc[comp.key] = (r if comp.good_high else 1 - r) * 100
    return sc


def composites(scores: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=scores.index)
    for h in HORIZONS:
        cols = [c.key for c in COMPONENTS if c.horizon == h and c.key in scores]
        if not cols:
            out[f"H_{h}"] = np.nan
            continue
        sub = scores[cols]
        out[f"H_{h}"] = sub.mean(axis=1).where(sub.notna().sum(axis=1) >= min(3, len(cols)))
        out[f"n_{h}"] = sub.notna().sum(axis=1)
    out["H_gesamt"] = out[[f"H_{h}" for h in HORIZONS]].mean(axis=1)
    return out


def backtest(close: pd.Series, comp: pd.DataFrame) -> pd.DataFrame:
    """Folgerendite und Rückgangsrisiko je Score-Bereich und Horizont (nur abgeschlossene Zeiträume)."""
    c = close.values
    rows = []
    for h, (name, _, days) in HORIZONS.items():
        sc = comp[f"H_{h}"].values
        n = len(c)
        fwd = np.full(n, np.nan)
        mdd = np.full(n, np.nan)
        for i in range(n - days):
            fwd[i] = c[i + days] / c[i] - 1
            mdd[i] = c[i + 1:i + 1 + days].min() / c[i] - 1
        df = pd.DataFrame({"score": sc, "fwd": fwd, "mdd": mdd}).dropna()
        if df.empty:
            continue
        base = {"fwd": df["fwd"].mean(), "pos": (df["fwd"] > 0).mean(), "dd10": (df["mdd"] <= -0.10).mean()}
        for lo, hi, lab in LEVELS:
            g = df[(df["score"] >= lo) & (df["score"] < hi)]
            rows.append({"horizon": h, "Bereich": f"{lo}–{min(hi, 100)} {lab}", "lo": lo, "Tage": len(g),
                         "Ø Rendite": g["fwd"].mean() if len(g) else np.nan,
                         "Median Rendite": g["fwd"].median() if len(g) else np.nan,
                         "Anteil positiv": (g["fwd"] > 0).mean() if len(g) else np.nan,
                         "Rückgang ≥ 10 % zwischendurch": (g["mdd"] <= -0.10).mean() if len(g) else np.nan,
                         "Basis Ø Rendite": base["fwd"], "Basis positiv": base["pos"], "Basis ≥10%": base["dd10"],
                         "Rangkorrelation": float(df["score"].corr(df["fwd"], method="spearman"))})
    return pd.DataFrame(rows)


def compute_health(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw = raw_values(f)
    scores = percentile_scores(raw)
    comp = composites(scores)
    daily = pd.concat([f[["close"]], raw.add_prefix("raw_"), scores.add_prefix("s_"), comp], axis=1)
    bt = backtest(f["close"], comp)
    return daily, bt, list(scores.columns)


# ---------------------------------------------------------------------------------------------- Regel-Backtest
ENTRY_MODES = {
    "dip": "Dip: Kurzfrist-Score fällt unter die Schwelle",
    "dreh": "Drehung: Kurzfrist-Score steigt wieder über die Schwelle (bestätigt)",
}


def rule_trades(d: pd.DataFrame, entry: str = "dip", kurz_th: float = 20, lang_min: float = 40,
                hold: int = 21, cost: float = 0.0) -> pd.DataFrame:
    """Long-Trades ohne Überschneidung: Einstieg zum Schlusskurs des Signaltags, Ausstieg nach `hold` Handelstagen.

    entry="dip":  H_kurz < kurz_th und H_lang >= lang_min (erster Tag, an dem die Bedingung neu erfüllt ist).
    entry="dreh": H_kurz kreuzt von unten über kurz_th (Vortag < Schwelle) und H_lang >= lang_min.
    Nur Werte bis zum Signaltag werden genutzt; nicht abgeschlossene Trades werden als offen markiert."""
    d = d[d["close"].notna()]
    k, l, c = d["H_kurz"].values, d["H_lang"].values, d["close"].values
    dates = d.index
    ok_l = np.nan_to_num(l, nan=-1) >= lang_min
    below = np.nan_to_num(k, nan=999) < kurz_th
    prev_below = np.r_[False, below[:-1]]
    if entry == "dip":
        sig = below & ok_l & ~(prev_below & np.r_[False, ok_l[:-1]])
    else:
        sig = ~below & ~np.isnan(k) & prev_below & ok_l
    rows, i, n = [], 0, len(c)
    while i < n:
        if sig[i]:
            j = min(i + hold, n - 1)
            path = c[i + 1:j + 1]
            closed = i + hold <= n - 1
            ret = c[j] / c[i] - 1 - (2 * cost if closed else 0)
            rows.append({"Einstieg": dates[i], "Ausstieg": dates[j] if closed else pd.NaT,
                         "Kurs Einstieg": c[i], "Kurs Ausstieg": c[j] if closed else np.nan,
                         "Kurz": k[i], "Lang": l[i], "Rendite": ret,
                         "Max. Rückgang": (path.min() / c[i] - 1) if len(path) else np.nan,
                         "Status": "abgeschlossen" if closed else "offen"})
            i = j + 1 if closed else n
        else:
            i += 1
    return pd.DataFrame(rows)


def trade_stats(t: pd.DataFrame) -> dict:
    t = t[t["Status"] == "abgeschlossen"] if not t.empty else t
    if t.empty:
        return {"n": 0}
    r = t["Rendite"]
    wins, losses = r[r > 0], r[r <= 0]
    return {"n": len(r), "winrate": float((r > 0).mean()), "avg": float(r.mean()), "median": float(r.median()),
            "avg_win": float(wins.mean()) if len(wins) else np.nan,
            "avg_loss": float(losses.mean()) if len(losses) else np.nan,
            "pf": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else np.inf,
            "worst": float(r.min()), "best": float(r.max()), "worst_dd": float(t["Max. Rückgang"].min())}


def base_stats(d: pd.DataFrame, hold: int, start=None) -> dict:
    """Vergleich: alle Tage (gleiche Haltedauer) – zeigt, ob die Regel besser als 'irgendwann kaufen' ist."""
    c = d["close"].dropna()
    fwd = (c.shift(-hold) / c - 1).dropna()
    if start is not None:
        fwd = fwd[fwd.index >= start]
    return {"winrate": float((fwd > 0).mean()), "avg": float(fwd.mean())} if len(fwd) else {}
