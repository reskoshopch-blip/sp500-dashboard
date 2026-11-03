"""Point-in-time-Datensatz: eine Zeile pro Unternehmen und Geschäftsjahr.

Verhinderung von Look-ahead-Bias:
  * Bilanzwerte 'wie erstmals gemeldet' (original_value), nicht spätere Korrekturen (Restatements).
  * Stichtag der Prognose t0 = erster Handelstag NACH dem Einreichungsdatum des 10-K (Filing-Datum),
    nicht das Geschäftsjahresende – erst dann war die Bilanz öffentlich.
  * Kursmerkmale nutzen nur Kurse bis t0, Labels nur Kurse nach t0.

Labels (Horizont H, Standard 365 Kalendertage ab t0):
  excess_ret = Rendite Aktie − Rendite SPY (beide inkl. Dividenden)
  y_outperf  = 1 wenn excess_ret > 0
  max_dd     = tiefster Schlusskurs in (t0, t0+H] relativ zum Kurs bei t0 − 1
  y_dd30     = 1 wenn max_dd < −30 %
Labels sind NA, solange t0+H noch nicht erreicht ist – diese Zeilen sind die 'Live'-Prognosen.

Marktregime bei t0 (aus SPY): Bull, wenn SPY über seinem 200-Tage-Durchschnitt, sonst Bear.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.metrics import total_debt
from src.normalization.derive import add_derived_items

# ------------------------------------------------------------------ Merkmalsdefinitionen
BS_FEATURES: dict[str, str] = {
    "log_assets": "Grösse (log Bilanzsumme)",
    "equity_ratio": "Eigenkapitalquote",
    "liab_to_assets": "Verbindlichkeiten / Bilanzsumme",
    "debt_to_assets": "Finanzschulden / Bilanzsumme",
    "debt_to_equity": "Debt-to-Equity",
    "cash_to_assets": "Cash / Bilanzsumme",
    "current_ratio": "Current Ratio",
    "quick_ratio": "Quick Ratio",
    "wc_to_assets": "Working Capital / Bilanzsumme",
    "goodwill_to_assets": "Goodwill / Bilanzsumme",
    "intang_to_assets": "Immaterielle / Bilanzsumme",
    "ppe_to_assets": "Sachanlagen / Bilanzsumme",
    "receiv_to_assets": "Forderungen / Bilanzsumme",
    "inventory_to_assets": "Vorräte / Bilanzsumme",
    "retained_to_assets": "Gewinnrücklagen / Bilanzsumme",
    "treasury_to_assets": "Eigene Aktien / Bilanzsumme",
    "lease_to_assets": "Leasingverbindl. / Bilanzsumme",
    "equity_negative": "Negatives Eigenkapital (0/1)",
    "g_assets": "Wachstum Bilanzsumme",
    "g_liab": "Wachstum Verbindlichkeiten",
    "g_equity": "Wachstum Eigenkapital",
    "g_debt": "Wachstum Finanzschulden",
    "g_cash": "Wachstum Cash",
    "g_goodwill": "Wachstum Goodwill",
    "g_receiv": "Wachstum Forderungen",
    "g_inventory": "Wachstum Vorräte",
    "g_retained": "Veränderung Gewinnrücklagen / Bilanzsumme",
    "d_equity_ratio": "Δ Eigenkapitalquote",
    "d_debt_to_assets": "Δ Finanzschulden / Bilanzsumme",
    "d_current_ratio": "Δ Current Ratio",
    "d_cash_to_assets": "Δ Cash / Bilanzsumme",
    "d_treasury_to_assets": "Δ Eigene Aktien / Bilanzsumme (Rückkäufe)",
}
MARKET_FEATURES: dict[str, str] = {
    "mkt_bull": "Marktregime Bull (SPY > SMA200)",
    "mkt_vs_sma200": "SPY Abstand zu SMA200",
    "mkt_mom_12m": "SPY Rendite 12 Monate",
    "mkt_vol_3m": "SPY Volatilität 3 Monate",
}
PRICE_FEATURES: dict[str, str] = {
    "mom_12m": "Aktie Rendite 12 Monate vor t0",
    "mom_1m": "Aktie Rendite 1 Monat vor t0",
    "vol_12m": "Aktie Volatilität 12 Monate",
    "dist_52w_high": "Abstand zum 52-Wochen-Hoch",
}
CATEGORICAL = {"sector": "Branche"}
FEATURE_LABELS = {**BS_FEATURES, **MARKET_FEATURES, **PRICE_FEATURES, **CATEGORICAL}


# ------------------------------------------------------------------ Bilanz wie erstmals gemeldet
def pit_wide(items: pd.DataFrame) -> pd.DataFrame:
    """Wide-Tabelle (cik, fiscal_year) mit Werten wie erstmals gemeldet; Ableitungen neu berechnet."""
    base = items[~items["is_derived"].astype(bool)].copy()
    base["value"] = pd.to_numeric(base["original_value"], errors="coerce").fillna(
        pd.to_numeric(base["value"], errors="coerce"))
    keep = [c for c in ["cik", "fiscal_year", "period_end", "item_key", "value", "unit", "tag", "accn", "form",
                        "filed", "restated"] if c in base.columns]
    base = add_derived_items(base[keep].reset_index(drop=True), ["cik", "fiscal_year", "period_end"])
    return base.pivot_table(index=["cik", "fiscal_year"], columns="item_key", values="value",
                            aggfunc="first").reset_index()


def availability_dates(periods: pd.DataFrame) -> pd.DataFrame:
    """Frühestes Einreichungsdatum eines Jahresberichts mit dem Stichtag des Geschäftsjahres."""
    p = periods.copy()
    p["period_end"] = pd.to_datetime(p["period_end"])
    p["filed"] = pd.to_datetime(p["filed"])
    fy_end = p.groupby(["cik", "fiscal_year"])["period_end"].transform("max")
    p = p[p["period_end"] == fy_end]
    return p.groupby(["cik", "fiscal_year"], as_index=False).agg(period_end=("period_end", "max"),
                                                                 filed=("filed", "min"))


# ------------------------------------------------------------------ Merkmale aus der Bilanz
def _ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b).where(b > 0)


def _growth(cur: pd.Series, prev: pd.Series) -> pd.Series:
    return (cur / prev - 1).where(prev > 0)


def balance_features(wide: pd.DataFrame) -> pd.DataFrame:
    w = wide.copy()
    for col in ["total_assets", "total_liabilities", "equity_total", "cash", "current_assets",
                "current_liabilities", "goodwill", "intangibles", "ppe", "receivables", "inventory",
                "retained_earnings", "treasury_stock", "op_lease_liab", "st_investments", "ltd_total",
                "ltd_noncurrent", "st_debt", "ltd_current", "debt_current_total"]:
        if col not in w:
            w[col] = np.nan
    td = [total_debt(r) for r in w.to_dict("records")]
    w["total_debt"] = [v if status != "NA" else np.nan for v, status, _ in td]
    ta = w["total_assets"]
    f = pd.DataFrame({"cik": w["cik"].astype(int), "fiscal_year": w["fiscal_year"].astype(int)})
    f["log_assets"] = np.log10(ta.where(ta > 0))
    f["equity_ratio"] = _ratio(w["equity_total"], ta)
    f["liab_to_assets"] = _ratio(w["total_liabilities"], ta)
    f["debt_to_assets"] = _ratio(w["total_debt"], ta)
    f["debt_to_equity"] = _ratio(w["total_debt"], w["equity_total"]).clip(upper=20)
    f["cash_to_assets"] = _ratio(w["cash"], ta)
    f["current_ratio"] = _ratio(w["current_assets"], w["current_liabilities"]).clip(upper=20)
    quick_num = w["cash"] + w["st_investments"].fillna(0) + w["receivables"].fillna(0)
    f["quick_ratio"] = _ratio(quick_num, w["current_liabilities"]).clip(upper=20)
    f["wc_to_assets"] = _ratio(w["current_assets"] - w["current_liabilities"], ta)
    f["goodwill_to_assets"] = _ratio(w["goodwill"], ta)
    f["intang_to_assets"] = _ratio(w["intangibles"], ta)
    f["ppe_to_assets"] = _ratio(w["ppe"], ta)
    f["receiv_to_assets"] = _ratio(w["receivables"], ta)
    f["inventory_to_assets"] = _ratio(w["inventory"], ta)
    f["retained_to_assets"] = _ratio(w["retained_earnings"], ta)
    f["treasury_to_assets"] = _ratio(w["treasury_stock"], ta)
    f["lease_to_assets"] = _ratio(w["op_lease_liab"], ta)
    f["equity_negative"] = (w["equity_total"] < 0).astype(float).where(w["equity_total"].notna())

    prev = w.assign(fiscal_year=w["fiscal_year"] + 1).set_index(["cik", "fiscal_year"])
    cur = w.set_index(["cik", "fiscal_year"])
    pv = prev.reindex(cur.index)
    fprev = f.assign(fiscal_year=f["fiscal_year"] + 1).set_index(["cik", "fiscal_year"]).reindex(cur.index)
    fi = f.set_index(["cik", "fiscal_year"])
    for name, col in [("g_assets", "total_assets"), ("g_liab", "total_liabilities"), ("g_equity", "equity_total"),
                      ("g_debt", "total_debt"), ("g_cash", "cash"), ("g_goodwill", "goodwill"),
                      ("g_receiv", "receivables"), ("g_inventory", "inventory")]:
        fi[name] = _growth(cur[col], pv[col]).clip(-1, 10).values
    fi["g_retained"] = ((cur["retained_earnings"] - pv["retained_earnings"]) / pv["total_assets"]) \
        .where(pv["total_assets"] > 0).values
    for name, col in [("d_equity_ratio", "equity_ratio"), ("d_debt_to_assets", "debt_to_assets"),
                      ("d_current_ratio", "current_ratio"), ("d_cash_to_assets", "cash_to_assets"),
                      ("d_treasury_to_assets", "treasury_to_assets")]:
        fi[name] = (fi[col] - fprev[col]).values
    return fi.reset_index()


# ------------------------------------------------------------------ Kurse, Labels, Marktregime
def market_frame(spy: pd.Series) -> pd.DataFrame:
    s = spy.sort_index()
    sma = s.rolling(200, min_periods=200).mean()
    r = np.log(s).diff()
    return pd.DataFrame({
        "close": s, "sma200": sma,
        "vs_sma200": s / sma - 1,
        "vol_3m": r.rolling(63, min_periods=50).std() * np.sqrt(252),
    })


def _idx_after(index: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    return int(index.searchsorted(ts, side="right"))


def _idx_on_or_before(index: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    return int(index.searchsorted(ts, side="right")) - 1


def price_block(s: pd.Series, filed: pd.Timestamp, horizon: int) -> dict:
    """Kursmerkmale bis t0 und Labels nach t0 für eine Aktie."""
    out = {"t0": pd.NaT, "p0": np.nan, "ret": np.nan, "max_dd": np.nan, "mom_12m": np.nan, "mom_1m": np.nan,
           "vol_12m": np.nan, "dist_52w_high": np.nan, "label_end": pd.NaT}
    idx = s.index
    i0 = _idx_after(idx, filed)
    if i0 >= len(idx) or (idx[i0] - filed).days > 10:
        return out
    t0, p0 = idx[i0], float(s.iloc[i0])
    out.update(t0=t0, p0=p0, label_end=t0 + pd.Timedelta(days=horizon))
    ib = _idx_on_or_before(idx, t0 - pd.Timedelta(days=365))
    if ib >= 0 and (t0 - pd.Timedelta(days=365) - idx[ib]).days <= 10:
        past = s.iloc[ib:i0 + 1]
        out["mom_12m"] = p0 / float(s.iloc[ib]) - 1
        out["vol_12m"] = float(np.log(past).diff().std() * np.sqrt(252))
        out["dist_52w_high"] = p0 / float(past.max()) - 1
    im = _idx_on_or_before(idx, t0 - pd.Timedelta(days=30))
    if im >= 0 and (t0 - pd.Timedelta(days=30) - idx[im]).days <= 10:
        out["mom_1m"] = p0 / float(s.iloc[im]) - 1
    end = out["label_end"]
    if idx[-1] >= end:
        i1 = _idx_on_or_before(idx, end)
        if (end - idx[i1]).days <= 10 and i1 > i0:
            out["ret"] = float(s.iloc[i1]) / p0 - 1
            out["max_dd"] = float(s.iloc[i0 + 1:i1 + 1].min()) / p0 - 1
    return out


def bench_return(spy: pd.Series, t0: pd.Timestamp, end: pd.Timestamp) -> float:
    idx = spy.index
    i0 = _idx_on_or_before(idx, t0)
    if i0 < 0 or idx[-1] < end:
        return np.nan
    i1 = _idx_on_or_before(idx, end)
    return float(spy.iloc[i1]) / float(spy.iloc[i0]) - 1


def build_dataset(items: pd.DataFrame, periods: pd.DataFrame, companies: pd.DataFrame,
                  prices: dict[str, pd.Series], spy: pd.Series, years: tuple[int, ...],
                  horizon: int = 365, dd_threshold: float = -0.30) -> pd.DataFrame:
    wide = pit_wide(items)
    feats = balance_features(wide)
    feats = feats[feats["fiscal_year"].isin(years)]
    avail = availability_dates(periods)
    df = feats.merge(avail, on=["cik", "fiscal_year"], how="inner")
    comp = companies[["cik", "primary_ticker", "name", "sector"]].copy()
    comp["cik"] = comp["cik"].astype(int)
    df = df.merge(comp, on="cik", how="inner")
    mkt = market_frame(spy)

    rows = []
    for r in df.itertuples(index=False):
        s = prices.get(r.primary_ticker)
        if s is None:
            continue
        pb = price_block(s, pd.Timestamp(r.filed), horizon)
        if pd.isna(pb["t0"]):
            continue
        im = _idx_on_or_before(mkt.index, pb["t0"])
        m = mkt.iloc[im] if im >= 0 else None
        br = bench_return(spy, pb["t0"], pb["label_end"]) if pd.notna(pb["ret"]) else np.nan
        rows.append({"cik": r.cik, "fiscal_year": r.fiscal_year, **pb,
                     "mkt_vs_sma200": m["vs_sma200"] if m is not None else np.nan,
                     "mkt_vol_3m": m["vol_3m"] if m is not None else np.nan,
                     "mkt_mom_12m": _spy_mom(spy, pb["t0"]),
                     "bench_ret": br})
    px = pd.DataFrame(rows)
    if px.empty:
        return pd.DataFrame()
    out = df.merge(px, on=["cik", "fiscal_year"], how="inner")
    out["mkt_bull"] = (out["mkt_vs_sma200"] > 0).astype(float).where(out["mkt_vs_sma200"].notna())
    out["regime"] = np.where(out["mkt_vs_sma200"].isna(), "unbekannt",
                             np.where(out["mkt_vs_sma200"] > 0, "Bull", "Bear"))
    out["excess_ret"] = out["ret"] - out["bench_ret"]
    out["y_outperf"] = (out["excess_ret"] > 0).astype(float).where(out["excess_ret"].notna())
    out["y_dd30"] = (out["max_dd"] < dd_threshold).astype(float).where(out["max_dd"].notna())
    out["sector"] = out["sector"].fillna("Unbekannt")
    out["t0_year"] = pd.to_datetime(out["t0"]).dt.year
    return out.sort_values(["t0", "cik"]).reset_index(drop=True)


def _spy_mom(spy: pd.Series, t0: pd.Timestamp) -> float:
    i0 = _idx_on_or_before(spy.index, t0)
    ib = _idx_on_or_before(spy.index, t0 - pd.Timedelta(days=365))
    if i0 < 0 or ib < 0:
        return np.nan
    return float(spy.iloc[i0]) / float(spy.iloc[ib]) - 1
