"""Synthetische Makro- und COT-Reihen im Format der echten Quellen (nur für Tests – keine echten Daten)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.fundamental.catalog import FRED

MONTHLY = {"PCOPPUSDM", "M2SL", "UNRATE", "PAYEMS", "INDPRO", "CPILFESL", "UMCSENT", "PERMIT", "IRLTLT01JPM156N",
           "IRSTCI01JPM156N", "JPNASSETS", "JPNPROINDMISMEI", "XTEXVA01JPM664S", "LRUN64TTJPM156S",
           "JPNCPIALLMINMEI", "JPNLOLITOAASTSAM", "USALOLITOAASTSAM"}
WEEKLY = {"WALCL", "WTREGEN", "NFCI", "ICSA"}
LEVEL = {"DGS10": 4, "DGS2": 3, "DGS3MO": 2, "DFF": 2, "DFII10": 1, "T10YIE": 2, "BAA10Y": 2, "BAMLH0A0HYM2": 4,
         "NFCI": 0, "UNRATE": 5, "UMCSENT": 85, "IRLTLT01JPM156N": 1, "IRSTCI01JPM156N": 0.1,
         "JPNLOLITOAASTSAM": 100, "USALOLITOAASTSAM": 100, "DEXJPUS": 110, "LRUN64TTJPM156S": 3}


def fred_series(sid: str, start="1985-01-01", end="2026-10-23", seed=0) -> pd.Series:
    rng = np.random.default_rng(abs(hash(sid)) % 2**32 + seed)
    freq = "MS" if sid in MONTHLY else ("W-WED" if sid in WEEKLY else "B")
    idx = pd.date_range(start, end, freq=freq)
    if sid in LEVEL:
        step = 0.05 if freq == "B" else 0.2
        v = LEVEL[sid] + np.cumsum(rng.normal(0, step, len(idx))) * (0.2 if freq == "B" else 1)
    else:
        v = 100 * np.exp(np.cumsum(rng.normal(0.0004 if freq == "B" else 0.003, 0.01, len(idx))))
    return pd.Series(v, index=idx, name=sid)


def fred_downloader(sid: str) -> pd.Series:
    if sid not in FRED:
        raise RuntimeError("unbekannt")
    return fred_series(sid)


def cot_frame(codes=("13874A", "240741"), start="1990-01-02", end="2026-10-20") -> pd.DataFrame:
    rows = []
    for k, code in enumerate(codes):
        rng = np.random.default_rng(k)
        dates = pd.date_range(start, end, freq="W-TUE")
        oi = 100000 + np.cumsum(rng.normal(0, 1000, len(dates))).clip(-50000)
        net = np.cumsum(rng.normal(0, 800, len(dates)))
        for d, o, n in zip(dates, oi, net):
            rows.append({"date": d, "code": code, "name": f"TEST {code}", "oi": o, "nc_long": 30000 + max(n, 0),
                         "nc_short": 30000 + max(-n, 0), "c_long": 40000, "c_short": 40000 + n,
                         "nr_long": 10000, "nr_short": 10000})
    return pd.DataFrame(rows)


def cot_downloader(codes):
    return cot_frame(tuple(codes))
