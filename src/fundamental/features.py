"""Tägliche Fundamental-Indikatoren je Markt (point-in-time).

Jede Transformation (z. B. Wachstum über 12 Monate) wird in der ursprünglichen Frequenz der Reihe berechnet,
danach um den Veröffentlichungsverzug verschoben und auf die Handelstage des Marktes übertragen. Ein Wert gilt
nur so lange, wie es der Katalog erlaubt (z. B. 75 Tage für Monatsdaten) – danach ist er fehlend, nicht alt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.fundamental.catalog import COT_CODES, COT_LAG, FRED, for_market


def known_on(s: pd.Series | None, idx: pd.DatetimeIndex, lag: int, max_age: int) -> pd.Series:
    """Wert, der an jedem Tag von idx bekannt war (Beobachtungsdatum + lag ≤ Tag, Alter ≤ max_age)."""
    if s is None or len(s.dropna()) == 0:
        return pd.Series(np.nan, index=idx)
    s = s.dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    avail = s.index + pd.Timedelta(days=lag)
    df = pd.DataFrame({"v": s.values, "obs": s.index}, index=avail)
    df = df[~df.index.duplicated(keep="last")]
    r = df.reindex(df.index.union(idx)).ffill().reindex(idx)
    age = (pd.Series(idx, index=idx) - pd.to_datetime(r["obs"])).dt.days
    return r["v"].where(age <= max_age + lag).astype(float)


def _f(fred: dict, sid: str) -> pd.Series | None:
    s = fred.get(sid)
    return None if s is None or s.empty else s


def _k(fred: dict, sid: str, transformed: pd.Series | None, idx) -> pd.Series:
    _, lag, max_age = FRED[sid]
    return known_on(transformed, idx, lag, max_age)


def _daily_diff(s, n):
    return None if s is None else s - s.shift(n)


def _pct(s, n):
    return None if s is None else s / s.shift(n) - 1


def _ann(s, n):
    return None if s is None else (s / s.shift(n)) ** (12 / n) - 1


def cot_features(cot: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    if cot is None or cot.empty:
        return pd.DataFrame()
    c = cot[cot["code"].isin(codes)].copy()
    if c.empty:
        return pd.DataFrame()
    g = c.groupby("date")[["oi", "nc_long", "nc_short", "c_long", "c_short"]].sum().sort_index()
    g = g[g["oi"] > 0]
    nc = (g["nc_long"] - g["nc_short"]) / g["oi"]
    cm = (g["c_long"] - g["c_short"]) / g["oi"]

    def idx3y(x):
        lo, hi = x.rolling(156, min_periods=52).min(), x.rolling(156, min_periods=52).max()
        return ((x - lo) / (hi - lo).replace(0, np.nan)) * 100

    return pd.DataFrame({"cot_nc": nc, "cot_index": idx3y(nc), "cot_comm_index": idx3y(cm),
                         "cot_chg": nc.diff(8) * 100})


def build(market: str, idx: pd.DatetimeIndex, fred: dict[str, pd.Series], cot: pd.DataFrame | None = None,
          earn: pd.DataFrame | None = None, pe_daily: pd.Series | None = None,
          commodities: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Rohwerte aller Indikatoren des Marktes auf den Handelstagen idx."""
    f = fred
    out: dict[str, pd.Series] = {}
    dgs10, dgs2, dff = _f(f, "DGS10"), _f(f, "DGS2"), _f(f, "DFF")
    out["us10y_chg"] = _k(f, "DGS10", _daily_diff(dgs10, 126), idx)
    out["us2y_chg"] = _k(f, "DGS2", _daily_diff(dgs2, 126), idx)
    if dgs2 is not None and dff is not None:
        out["policy_exp"] = _k(f, "DGS2", dgs2 - dff.reindex(dgs2.index, method="ffill"), idx)
    if dgs10 is not None and dgs2 is not None:
        out["curve_10y2y"] = _k(f, "DGS10", dgs10 - dgs2.reindex(dgs10.index), idx)
    if dgs10 is not None and _f(f, "DGS3MO") is not None:
        out["curve_10y3m"] = _k(f, "DGS10", dgs10 - f["DGS3MO"].reindex(dgs10.index), idx)
    real = _f(f, "DFII10")
    out["real10"] = _k(f, "DFII10", real, idx)
    out["real10_chg"] = _k(f, "DFII10", _daily_diff(real, 126), idx)
    out["breakeven"] = _k(f, "T10YIE", _f(f, "T10YIE"), idx)
    jp10 = _f(f, "IRLTLT01JPM156N")
    out["jp10y_chg"] = _k(f, "IRLTLT01JPM156N", _daily_diff(jp10, 6), idx)
    if jp10 is not None and dgs10 is not None:
        us_m = dgs10.resample("MS").mean()
        spread = (us_m - jp10).dropna()
        out["us_jp_spread"] = _k(f, "IRLTLT01JPM156N", spread, idx)
        out["us_jp_spread_chg"] = _k(f, "IRLTLT01JPM156N", spread.diff(6), idx)
    out["jp_short_chg"] = _k(f, "IRSTCI01JPM156N", _daily_diff(_f(f, "IRSTCI01JPM156N"), 12), idx)
    # Liquidität
    walcl, tga, rrp = _f(f, "WALCL"), _f(f, "WTREGEN"), _f(f, "RRPONTSYD")
    if walcl is not None and tga is not None:
        r = rrp.reindex(walcl.index, method="ffill").fillna(0) * 1000 if rrp is not None else 0
        net = walcl - tga.reindex(walcl.index, method="ffill") - r
        out["net_liq_chg"] = _k(f, "WALCL", _pct(net.dropna(), 13), idx)
    out["nfci"] = _k(f, "NFCI", _f(f, "NFCI"), idx)
    baa = _f(f, "BAA10Y")
    out["credit_baa"] = _k(f, "BAA10Y", baa, idx)
    out["credit_baa_chg"] = _k(f, "BAA10Y", _daily_diff(baa, 63), idx)
    out["hy_spread"] = _k(f, "BAMLH0A0HYM2", _f(f, "BAMLH0A0HYM2"), idx)
    out["m2_yoy"] = _k(f, "M2SL", _pct(_f(f, "M2SL"), 12), idx)
    out["boj_assets_yoy"] = _k(f, "JPNASSETS", _pct(_f(f, "JPNASSETS"), 12), idx)
    # Konjunktur
    icsa = _f(f, "ICSA")
    out["claims_chg"] = _k(f, "ICSA", _pct(icsa.rolling(4).mean(), 26) if icsa is not None else None, idx)
    un = _f(f, "UNRATE")
    if un is not None:
        u3 = un.rolling(3).mean()
        out["sahm"] = _k(f, "UNRATE", u3 - u3.shift(1).rolling(12).min(), idx)
    out["payrolls_3m"] = _k(f, "PAYEMS", _ann(_f(f, "PAYEMS"), 3), idx)
    out["indpro_yoy"] = _k(f, "INDPRO", _pct(_f(f, "INDPRO"), 12), idx)
    out["core_cpi_3m"] = _k(f, "CPILFESL", _ann(_f(f, "CPILFESL"), 3), idx)
    out["sentiment"] = _k(f, "UMCSENT", _f(f, "UMCSENT"), idx)
    out["permits_yoy"] = _k(f, "PERMIT", _pct(_f(f, "PERMIT"), 12), idx)
    # Rohstoffe: Yahoo täglich, Lücken (v. a. vor 2000) aus FRED
    com = commodities or {}

    def daily_pct(s, n):
        return None if s is None or s.empty else _pct(s[s > 0], n)

    oil = _f(f, "DCOILWTICO")
    oil_y = com.get("oil")
    for key, n in (("oil_yoy", 252), ("oil_3m", 63)):
        a = known_on(daily_pct(oil_y, n), idx, 1, 10)
        b = _k(f, "DCOILWTICO", daily_pct(oil, n), idx)
        out[key] = a.combine_first(b)
    cu, au = com.get("copper"), com.get("gold")
    cu_m = _f(f, "PCOPPUSDM")
    for key, n, nm in (("copper_yoy", 252, 12), ("copper_3m", 63, 3)):
        a = known_on(daily_pct(cu, n), idx, 1, 10)
        b = _k(f, "PCOPPUSDM", _pct(cu_m, nm), idx)
        out[key] = a.combine_first(b)
    if cu is not None and au is not None and not cu.empty and not au.empty:
        ratio = (cu / au.reindex(cu.index, method="ffill")).dropna()
        out["copper_gold"] = known_on(_pct(ratio[ratio > 0], 126), idx, 1, 10)
    out["gold_yoy"] = known_on(daily_pct(au, 252), idx, 1, 10)
    out["us_cli"] = _k(f, "USALOLITOAASTSAM", _daily_diff(_f(f, "USALOLITOAASTSAM"), 6), idx)
    out["jp_indpro_yoy"] = _k(f, "JPNPROINDMISMEI", _pct(_f(f, "JPNPROINDMISMEI"), 12), idx)
    out["jp_exports_yoy"] = _k(f, "XTEXVA01JPM664S", _pct(_f(f, "XTEXVA01JPM664S"), 12), idx)
    out["jp_cli"] = _k(f, "JPNLOLITOAASTSAM", _daily_diff(_f(f, "JPNLOLITOAASTSAM"), 6), idx)
    out["jp_cpi_yoy"] = _k(f, "JPNCPIALLMINMEI", _pct(_f(f, "JPNCPIALLMINMEI"), 12), idx)
    # Gewinne (bereits point-in-time zum Monatsende berechnet)
    if earn is not None and not earn.empty:
        for col in ("ni_growth", "ni_growth_chg", "rev_growth", "margin_chg"):
            if col in earn:
                out[col] = known_on(earn[col], idx, 0, 45)
    if pe_daily is not None and len(pe_daily.dropna()):
        pe = pe_daily.reindex(idx)
        out["pe"] = pe
        if real is not None:
            out["erp"] = 100 / pe - out["real10"]
    # Währung
    fx = _f(f, "DEXJPUS")
    out["usdjpy_chg"] = _k(f, "DEXJPUS", _pct(fx, 63), idx)
    out["usdjpy_vs_1y"] = _k(f, "DEXJPUS", fx / fx.rolling(252, min_periods=200).mean() - 1 if fx is not None else None, idx)
    out["dollar_chg"] = _k(f, "DTWEXBGS", _pct(_f(f, "DTWEXBGS"), 63), idx)
    # COT
    cf = cot_features(cot, COT_CODES.get(market, [])) if cot is not None else pd.DataFrame()
    for col in cf.columns:
        out[col] = known_on(cf[col], idx, COT_LAG, 21)
    keys = [i.key for i in for_market(market)]
    df = pd.DataFrame({k: out[k] for k in keys if k in out}, index=idx)
    return df.astype(float)
