"""Champagner-Index: python -m src.geo.champagne

Idee: Wird mehr Champagner gekauft, sind Stimmung und Luxuskonsum hoch. Direkte Absatzzahlen gibt es nur jährlich
(Comité Champagne), darum ein täglicher Ersatz über die Börse: ein gleichgewichteter Korb der börsenkotierten
Champagnerhäuser (Laurent-Perrier, Vranken-Pommery, Lanson-BCC, LVMH mit Moët & Chandon) im Vergleich zum
europäischen Gesamtmarkt (STOXX Europe 600).

* Rohsignal: relative Stärke des Korbs zum Markt über 3 Monate.
* Score 0–100: Perzentil dieses Werts in den letzten 5 Jahren (nur damals bekannte Daten).
* Gelernt: Ob ein hoher Score für den S&P 500 bullisch oder bärisch (Euphorie) war, wird jedes Jahr nur mit den
  Jahren davor bestimmt (Spearman-Korrelation zur 3-Monats-Rendite); danach zeigt der Score immer „hoch = bullisch“.
Ergebnis: data/geo/champagne_daily.parquet und data/geo/champagne.json.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import get_settings
from src.ml.prices import fetch_prices, load_prices
from src.utils import atomic_write_text

HOUSES = {"LPE.PA": "Laurent-Perrier", "VRAP.PA": "Vranken-Pommery", "ALLAN.PA": "Lanson-BCC",
          "MC.PA": "LVMH (Moët & Chandon)"}
MARKET = ["^STOXX", "^STOXX50E"]          # erster verfügbarer
SPY = "SPY"
SUBDIR = "geo_prices"
START = "2000-01-01"
LOOK = 63                                  # 3 Monate relative Stärke
WINDOW = 1260                              # 5 Jahre für das Perzentil
H = 63                                     # Lernziel: S&P 500 in 3 Monaten


def geo_dir():
    d = get_settings().data_dir / "geo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def basket(prices: dict[str, pd.Series]) -> pd.Series:
    """Gleichgewichteter Korb aus Tagesrenditen (Häuser kommen dazu, sobald sie Kurse haben)."""
    rets = pd.DataFrame({t: prices[t].pct_change() for t in HOUSES if t in prices})
    rets = rets.clip(-0.3, 0.3)                              # Datenfehler/Splits abfangen
    r = rets.mean(axis=1, skipna=True).fillna(0)
    return (1 + r).cumprod()


def rolling_pct(s: pd.Series, window: int = WINDOW, min_periods: int = 252) -> pd.Series:
    return s.rolling(window, min_periods=min_periods).apply(lambda a: (a[:-1] < a[-1]).mean() * 100 if len(a) > 1
                                                            else np.nan, raw=True)


def learn_sign(df: pd.DataFrame) -> pd.Series:
    """Je Kalenderjahr: Vorzeichen der Korrelation Score ↔ S&P-Rendite in 3 Monaten, nur mit früheren Daten."""
    sign = pd.Series(np.nan, index=df.index)
    fwd = df["spy"].shift(-H) / df["spy"] - 1
    for y in sorted(df.index.year.unique()):
        start = pd.Timestamp(f"{y}-01-01")
        cutoff = start - pd.Timedelta(days=int(H * 1.5))          # Ziel muss vor Jahresbeginn bekannt sein
        past = pd.DataFrame({"s": df["score"], "f": fwd})[df.index < cutoff].dropna()
        if len(past) < 500:
            continue
        ic = past["s"].corr(past["f"], method="spearman")
        sign[df.index.year == y] = 1.0 if ic >= 0 else -1.0
    return sign


def bucket_stats(df: pd.DataFrame, col: str = "score_learned") -> pd.DataFrame:
    rows = []
    for h, lab in ((21, "1 Monat"), (63, "3 Monate")):
        fwd = df["spy"].shift(-h) / df["spy"] - 1
        x = pd.DataFrame({"s": df[col], "f": fwd}).dropna()
        for name, m in (("Score ≥ 70 (bullisch)", x["s"] >= 70), ("30–70 (neutral)", (x["s"] > 30) & (x["s"] < 70)),
                        ("Score ≤ 30 (bärisch)", x["s"] <= 30), ("Alle Tage (Basis)", x["s"].notna())):
            g = x.loc[m, "f"]
            rows.append({"Horizont": lab, "Zustand": name, "Tage": int(len(g)),
                         "Trefferquote": float((g > 0).mean()) if len(g) else None,
                         "Ø Rendite": float(g.mean()) if len(g) else None})
    return pd.DataFrame(rows)


def build(prices: dict[str, pd.Series]) -> pd.DataFrame:
    mkt_t = next((t for t in MARKET if t in prices), None)
    if mkt_t is None or SPY not in prices or not any(t in prices for t in HOUSES):
        raise RuntimeError("Kurse fehlen (Champagnerhäuser, STOXX oder SPY).")
    b = basket(prices)
    idx = b.index.union(prices[SPY].index)
    df = pd.DataFrame({"basket": b, "market": prices[mkt_t], "spy": prices[SPY]}).reindex(idx).ffill()
    df = df.dropna()
    df["rel"] = df["basket"] / df["market"]
    df["rel_mom"] = df["rel"].pct_change(LOOK)
    df["score"] = rolling_pct(df["rel_mom"])
    df["sign"] = learn_sign(df)
    df["score_learned"] = np.where(df["sign"] < 0, 100 - df["score"], df["score"])
    df.loc[df["sign"].isna(), "score_learned"] = np.nan
    return df


def run(force: bool = False, downloader=None) -> dict:
    s = get_settings()
    tickers = [*HOUSES, *MARKET, SPY]
    info = fetch_prices(s, tickers, force=force, start=START, subdir=SUBDIR, require_benchmark=False,
                        downloader=downloader)
    prices = load_prices(s, tickers, subdir=SUBDIR, include_benchmark=False)
    df = build(prices)
    out = df.reset_index().rename(columns={"index": "date"})
    out.to_parquet(geo_dir() / "champagne_daily.parquet", index=False)
    last = df.dropna(subset=["score"]).iloc[-1]
    fwd = df["spy"].shift(-H) / df["spy"] - 1
    valid = pd.DataFrame({"s": df["score_learned"], "f": fwd}).dropna()
    summary = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "date": f"{df.index[-1]:%Y-%m-%d}", "score": float(last["score"]),
        "score_learned": None if pd.isna(last["score_learned"]) else float(last["score_learned"]),
        "direction": "Euphorie = bärisch" if last["sign"] < 0 else "Stärke = bullisch" if last["sign"] > 0 else None,
        "rel_mom": float(last["rel_mom"]),
        "ic": float(valid["s"].corr(valid["f"], method="spearman")) if len(valid) > 200 else None,
        "houses": {t: HOUSES[t] for t in HOUSES if t in prices},
        "missing": [t for t in tickers if t not in prices], "failed": info.get("failed", []),
        "stats": bucket_stats(df).to_dict("records")}
    atomic_write_text(geo_dir() / "champagne.json", json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> int:
    try:
        r = run()
    except Exception as exc:  # noqa: BLE001
        print(f"Champagner-Index nicht berechenbar: {exc}", file=sys.stderr)
        return 1
    print(f"Champagner-Index {r['date']}: Score {r['score']:.0f}, gelernt {r['score_learned']}, {r['direction']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
