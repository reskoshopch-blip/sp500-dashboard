"""Synthetische ML-Datensätze – NUR für Tests. Keine realen Kurse oder Bilanzen."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.features import BS_FEATURES, MARKET_FEATURES, PRICE_FEATURES

SECTORS = ["Energy", "Financials", "Health Care", "Industrials", "Information Technology", "Utilities"]


def spy_series(seed: int = 0, end: str = "2026-10-20") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-06-02", end)
    r = rng.normal(0.0004, 0.011, len(idx))
    r[(idx > "2008-09-01") & (idx < "2009-03-10")] -= 0.004   # Bärenmarkt-Phase
    r[(idx > "2020-02-20") & (idx < "2020-03-25")] -= 0.02
    r[(idx > "2022-01-05") & (idx < "2022-10-10")] -= 0.0015
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


def synthetic_dataset(n_comp: int = 220, seed: int = 1, signal: bool = True,
                      last_price: str = "2026-10-20") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    spy = spy_series()
    sma = spy.rolling(200).mean()
    rows = []
    for c in range(n_comp):
        sector = SECTORS[c % len(SECTORS)]
        for fy in range(2010, 2026):
            t0 = pd.Timestamp(fy + 1, 2, 1) + pd.Timedelta(days=int(rng.integers(0, 60)))
            t0 = spy.index[spy.index.searchsorted(t0)]
            r = {f: rng.normal() for f in list(BS_FEATURES) + list(PRICE_FEATURES)}
            r["equity_negative"] = float(rng.random() < 0.05)
            if rng.random() < 0.1:
                r["current_ratio"] = np.nan          # wie Banken: nicht gemeldet
            vs = float(spy.loc[t0] / sma.loc[t0] - 1)
            r.update(mkt_vs_sma200=vs, mkt_bull=float(vs > 0), mkt_mom_12m=rng.normal(0.08, 0.1),
                     mkt_vol_3m=abs(rng.normal(0.15, 0.05)))
            bull = vs > 0
            latent = (0.9 * r["g_equity"] - 0.9 * r["debt_to_assets"] * (1.6 if not bull else 0.6)) if signal else 0
            excess = 0.08 * latent + rng.normal(0, 0.18)
            max_dd = -abs(rng.normal(0.12, 0.08)) - (0.12 * max(r["debt_to_assets"], 0) if signal else 0)
            label_end = t0 + pd.Timedelta(days=365)
            done = label_end <= pd.Timestamp(last_price)
            rows.append({"cik": 10000 + c, "fiscal_year": fy, "primary_ticker": f"S{c:03d}", "name": f"Synth {c}",
                         "sector": sector, "t0": t0, "t0_year": t0.year, "label_end": label_end,
                         "filed": t0 - pd.Timedelta(days=1), "period_end": pd.Timestamp(fy, 12, 31),
                         "regime": "Bull" if bull else "Bear", **r,
                         "ret": excess + 0.1 if done else np.nan, "bench_ret": 0.1 if done else np.nan,
                         "excess_ret": excess if done else np.nan, "max_dd": max_dd if done else np.nan,
                         "y_outperf": float(excess > 0) if done else np.nan,
                         "y_dd30": float(max_dd < -0.3) if done else np.nan})
    return pd.DataFrame(rows)
