"""COT-Modell: Gold, Öl, Kupfer, S&P 500, Nasdaq 100 → COT-Index 0–100 → Long / Neutral / Short.

Daten: CFTC Commitments of Traders (Legacy, nur Futures) – dieselben Zahlen, die Wirtschaftskalender wie Tradays als
„CFTC … speculative net positions“ zeigen (= Non-Commercials netto). Zusätzlich Commercials (Absicherer) und
Non-Reportables (kleine Spekulanten) sowie das Open Interest.

Zeitpunkte: Stichtag Dienstag, Veröffentlichung Freitag 15:30 New York. Das Modell handelt erst zum Schlusskurs
des folgenden Montags (Stichtag + 6 Tage) – kein Blick in die Zukunft.

Ablauf je Markt
1. Merkmale je Woche: Netto-Positionen in % des Open Interest für die drei Gruppen, COT-Indizes (Lage innerhalb der
   letzten 26/52/156 Wochen, 0–100), 4-Wochen-Veränderungen, Open-Interest-Veränderung.
2. Jedes Merkmal → Perzentil 0–100 (nur Vergangenheit).
3. Lernen (jährlich am 1. Januar, nur abgeschlossene Zeiträume): Zusammenhang jedes Merkmals mit der Rendite der
   folgenden 13 Wochen → Richtung und Gewicht (t-Wert, gedeckelt). Grundrichtung ohne Signal: Commercials
   hoch = bullisch, Spekulanten-Extreme konträr.
4. COT-Score = gewichteter Durchschnitt → Long ab Schwelle, Short unter Schwelle, sonst Neutral.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKETS: dict[str, dict] = {
    "GOLD": {"label": "Gold (COMEX)", "codes": ["088691"], "price": "GC=F", "unit": "USD/oz"},
    "OIL": {"label": "Rohöl WTI (NYMEX)", "codes": ["067651"], "price": "CL=F", "unit": "USD/Barrel"},
    "COPPER": {"label": "Kupfer (COMEX)", "codes": ["085692"], "price": "HG=F", "unit": "USD/lb"},
    "SPX": {"label": "S&P 500 (E-mini, CME)", "codes": ["13874A"], "price": "^GSPC", "unit": "Punkte"},
    "NDX": {"label": "Nasdaq 100 (E-mini, CME)", "codes": ["209742"], "price": "^NDX", "unit": "Punkte"},
}
ALL_CODES = sorted({c for m in MARKETS.values() for c in m["codes"]})
TRADE_LAG_DAYS = 6          # Dienstag-Stichtag -> Montag-Schlusskurs
H_LEARN = 13                # Wochen
WIN_PCT, MIN_PCT = 156, 104  # Perzentil-Fenster (Wochen)

FEATURES = {
    # key: (Bezeichnung, Grundrichtung +1 hoch = bullisch / -1 / 0)
    "nc_net": ("Spekulanten netto (% OI) – „speculative net positions“", -1),
    "c_net": ("Commercials netto (% OI)", 1),
    "nr_net": ("Kleine Spekulanten netto (% OI)", -1),
    "nc_idx26": ("COT-Index Spekulanten 26 W.", -1),
    "nc_idx52": ("COT-Index Spekulanten 52 W.", -1),
    "nc_idx156": ("COT-Index Spekulanten 3 J.", -1),
    "c_idx26": ("COT-Index Commercials 26 W.", 1),
    "c_idx52": ("COT-Index Commercials 52 W.", 1),
    "c_idx156": ("COT-Index Commercials 3 J.", 1),
    "nc_chg4": ("Spekulanten, Veränderung 4 W.", 0),
    "c_chg4": ("Commercials, Veränderung 4 W.", 0),
    "oi_chg13": ("Open Interest, Veränderung 13 W.", 0),
}


def cot_index(x: pd.Series, n: int) -> pd.Series:
    lo, hi = x.rolling(n, min_periods=max(13, n // 2)).min(), x.rolling(n, min_periods=max(13, n // 2)).max()
    return (x - lo) / (hi - lo).replace(0, np.nan) * 100


def weekly_frame(cot: pd.DataFrame, codes: list[str], price: pd.Series | None) -> pd.DataFrame:
    """Wöchentliche Rohmerkmale + Kurs zum Handelszeitpunkt (Montag nach Veröffentlichung)."""
    c = cot[cot["code"].astype(str).isin(codes)]
    if c.empty:
        return pd.DataFrame()
    g = c.groupby("date")[["oi", "nc_long", "nc_short", "c_long", "c_short", "nr_long", "nr_short"]].sum().sort_index()
    g = g[g["oi"] > 0]
    f = pd.DataFrame(index=g.index)
    f["oi"] = g["oi"]
    f["nc_net"] = (g["nc_long"] - g["nc_short"]) / g["oi"]
    f["c_net"] = (g["c_long"] - g["c_short"]) / g["oi"]
    f["nr_net"] = (g["nr_long"] - g["nr_short"]) / g["oi"]
    f["nc_contracts"] = g["nc_long"] - g["nc_short"]
    for grp in ("nc", "c"):
        for n in (26, 52, 156):
            f[f"{grp}_idx{n}"] = cot_index(f[f"{grp}_net"], n)
        f[f"{grp}_chg4"] = f[f"{grp}_net"].diff(4) * 100
    f["oi_chg13"] = g["oi"].pct_change(13)
    f["trade_date"] = f.index + pd.Timedelta(days=TRADE_LAG_DAYS)
    if price is not None and len(price):
        px = price.sort_index()
        pos = px.index.searchsorted(f["trade_date"], side="right") - 1
        ok = pos >= 0
        vals = np.full(len(f), np.nan)
        vals[ok] = px.values[pos[ok]]
        # Kurs nur verwenden, wenn er nicht älter als 7 Tage ist
        age = np.full(len(f), 999)
        age[ok] = (f["trade_date"].values[ok] - px.index.values[pos[ok]]).astype("timedelta64[D]").astype(int)
        f["price"] = np.where(age <= 7, vals, np.nan)
    else:
        f["price"] = np.nan
    return f


def percentiles(f: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for k in FEATURES:
        if k in f and f[k].notna().sum() >= MIN_PCT:
            out[k] = f[k].rolling(WIN_PCT, min_periods=MIN_PCT).rank(pct=True) * 100
    return pd.DataFrame(out, index=f.index)


def fwd_returns(price: pd.Series, h: int) -> pd.Series:
    return price.shift(-h) / price - 1


def learn(sc: pd.DataFrame, fwd: pd.Series, years: list[int], h: int = H_LEARN) -> pd.DataFrame:
    rows = []
    dates = sc.index
    for y in years:
        cut = pd.Timestamp(f"{y}-01-01")
        # nur Wochen, deren 13-Wochen-Ergebnis vor dem Lerntag feststand
        usable = dates[(dates + pd.Timedelta(days=TRADE_LAG_DAYS + 7 * h)) < cut]
        for k in sc.columns:
            prior = FEATURES[k][1]
            s, r = sc.loc[usable, k], fwd.loc[usable]
            ok = s.notna() & r.notna()
            n = int(ok.sum())
            ic, t = np.nan, 0.0
            if n >= 104:
                ic = float(s[ok].corr(r[ok], method="spearman"))
                t = ic * np.sqrt(n / h) if pd.notna(ic) else 0.0
            if abs(t) >= 1:
                sign, w, src = int(np.sign(t)), float(min(abs(t), 3.0)), "gelernt"
            elif prior != 0:
                sign, w, src = prior, 0.5, "Grundrichtung"
            else:
                sign, w, src = 1, 0.0, "kein Signal"
            rows.append({"year": y, "key": k, "n": n, "ic": ic, "t": t, "sign": sign, "weight": w, "source": src})
    return pd.DataFrame(rows)


def score(sc: pd.DataFrame, learned: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=sc.index)
    years = sorted(learned["year"].unique())
    for i, y in enumerate(years):
        lo = pd.Timestamp(f"{y}-01-01")
        hi = pd.Timestamp(f"{years[i + 1]}-01-01") if i + 1 < len(years) else sc.index.max() + pd.Timedelta(days=1)
        m = (sc.index >= lo) & (sc.index < hi)
        if not m.any():
            continue
        ly = learned[learned["year"] == y].set_index("key")
        num = pd.Series(0.0, index=sc.index[m])
        den = pd.Series(0.0, index=sc.index[m])
        for k in sc.columns:
            if k not in ly.index or ly.loc[k, "weight"] <= 0:
                continue
            v = sc.loc[m, k]
            o = v if ly.loc[k, "sign"] > 0 else 100 - v
            w = ly.loc[k, "weight"]
            num += o.fillna(0) * w * v.notna()
            den += w * v.notna()
        out[m] = (num / den).where(den > 0)
    return out


def signal(s: pd.Series, long_th: float = 60, short_th: float = 40) -> pd.Series:
    sig = pd.Series(np.where(s >= long_th, 1, np.where(s <= short_th, -1, 0)), index=s.index, dtype=float)
    return sig.where(s.notna())


SIG_NAME = {1: "Long", 0: "Neutral", -1: "Short"}


def backtest(f: pd.DataFrame, sig: pd.Series, start=None, allow_short: bool = True, cost: float = 0.0005) -> dict:
    """Position der Woche t (Signal zum Montagsschluss) × Rendite bis zum nächsten Montag. Kosten je Wechsel."""
    d = pd.DataFrame({"p": f["price"], "s": sig}).dropna(subset=["p"])
    if start is not None:
        d = d[d.index >= start]
    d = d.dropna(subset=["s"])
    if len(d) < 30:
        return {}
    r = d["p"].shift(-1) / d["p"] - 1
    pos = d["s"].clip(lower=0 if not allow_short else -1)
    turn = pos.diff().abs().fillna(abs(pos.iloc[0]))
    strat = (pos * r - turn * cost).fillna(0)
    bh = r.fillna(0)
    eq, eq_bh = (1 + strat).cumprod(), (1 + bh).cumprod()
    years = (d.index[-1] - d.index[0]).days / 365.25

    def cagr(e):
        return float(e.iloc[-1] ** (1 / years) - 1) if years > 0 and e.iloc[-1] > 0 else np.nan

    def mdd(e):
        return float((e / e.cummax() - 1).min())
    # Trades = zusammenhängende Phasen mit gleicher Position ≠ 0
    trades = []
    cur, start_i = None, None
    vals = pos.values
    for i, v in enumerate(vals):
        if cur is None or v != cur:
            if cur not in (None, 0):
                trades.append((start_i, i, cur))
            cur, start_i = v, i
    if cur not in (None, 0):
        trades.append((start_i, len(vals) - 1, cur))
    tr = []
    p = d["p"].values
    for a, b, side in trades:
        ret = (p[b] / p[a] - 1) * side - 2 * cost
        tr.append({"Einstieg": d.index[a] + pd.Timedelta(days=TRADE_LAG_DAYS), "Ausstieg":
                   d.index[b] + pd.Timedelta(days=TRADE_LAG_DAYS), "Richtung": SIG_NAME[int(side)],
                   "Wochen": b - a, "Rendite": ret, "offen": b == len(vals) - 1})
    tr = pd.DataFrame(tr)
    closed = tr[~tr["offen"]] if len(tr) else tr
    wins, losses = (closed["Rendite"][closed["Rendite"] > 0], closed["Rendite"][closed["Rendite"] <= 0]) \
        if len(closed) else (pd.Series(dtype=float), pd.Series(dtype=float))
    return {
        "equity": pd.DataFrame({"Modell": eq, "Kaufen & Halten": eq_bh}),
        "cagr": cagr(eq), "cagr_bh": cagr(eq_bh), "mdd": mdd(eq), "mdd_bh": mdd(eq_bh),
        "trades": tr, "n_trades": int(len(closed)),
        "winrate": float((closed["Rendite"] > 0).mean()) if len(closed) else np.nan,
        "pf": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else np.inf,
        "avg_trade": float(closed["Rendite"].mean()) if len(closed) else np.nan,
        "time_long": float((pos > 0).mean()), "time_short": float((pos < 0).mean()),
        "hit_weeks": float(((pos * r) > 0).sum() / max(((pos != 0) & r.notna()).sum(), 1)),
        "start": str(d.index[0].date()),
    }


def by_signal(f: pd.DataFrame, sig: pd.Series, start=None) -> pd.DataFrame:
    """Was nach jedem Signal passierte: Ø Rendite und Anteil positiv nach 4 und 13 Wochen."""
    rows = []
    d = f.copy()
    d["s"] = sig
    if start is not None:
        d = d[d.index >= start]
    for h in (4, 13, 26):
        d[f"r{h}"] = fwd_returns(f["price"], h).reindex(d.index)
    base = {h: d[f"r{h}"].dropna() for h in (4, 13, 26)}
    for sv in (1, 0, -1):
        g = d[d["s"] == sv]
        row = {"Signal": SIG_NAME[sv], "Wochen": len(g)}
        for h in (4, 13, 26):
            x = g[f"r{h}"].dropna()
            row[f"Ø {h} W."] = x.mean() if len(x) else np.nan
            row[f"positiv {h} W."] = (x > 0).mean() if len(x) else np.nan
        rows.append(row)
    rows.append({"Signal": "Alle Wochen", "Wochen": int(d["s"].notna().sum()),
                 **{f"Ø {h} W.": base[h].mean() for h in (4, 13, 26)},
                 **{f"positiv {h} W.": (base[h] > 0).mean() for h in (4, 13, 26)}})
    return pd.DataFrame(rows)


def run_market(cot: pd.DataFrame, codes: list[str], price: pd.Series | None) -> dict | None:
    f = weekly_frame(cot, codes, price)
    if f.empty or len(f) < 200:
        return None
    sc = percentiles(f)
    fwd = fwd_returns(f["price"], H_LEARN)
    first = f.index.min().year + 4
    years = list(range(first, f.index.max().year + 1))
    learned = learn(sc, fwd, years)
    f["score"] = score(sc, learned)
    f["classic"] = f["c_idx156"]       # klassischer Commercials-COT-Index (Larry Williams)
    for k in sc.columns:
        f[f"s_{k}"] = sc[k]
    # Testbeginn: erstes Jahr mit gelernten Gewichten und Kursen
    priced = f[f["price"].notna() & f["score"].notna()]
    start = None
    if len(priced):
        y0 = max(priced.index.min().year + 1, first + 1)
        start = pd.Timestamp(f"{y0}-01-01")
    return {"weekly": f, "learned": learned, "start": start}
