"""Gewinne und Umsätze des S&P 500 aus den SEC-XBRL-Daten (point-in-time).

Quelle: die bereits geladenen Company-Facts-Dateien (data/raw/companyfacts, Duration-Facts aus 10-K/10-Q).

Regeln
* Je Periode (start, end) zählt der **zuerst gemeldete** Wert; bekannt ab dem Einreichungsdatum (filed).
  Spätere Korrekturen werden nicht rückwirkend eingesetzt (kein Blick in die Zukunft).
* Quartale: Perioden von 80–100 Tagen. Das 4. Quartal meldet kaum eine Firma separat; es wird als
  Geschäftsjahr minus 9-Monats-Wert (gleicher Beginn) berechnet und gilt ab Einreichung des 10-K.
* TTM (letzte 12 Monate) = Summe von 4 aufeinanderfolgenden Quartalen (Abstand erstes–letztes Quartalsende
  250–290 Tage), bekannt ab dem spätesten der vier Einreichungsdaten.
* Aggregate sind **like-for-like**: Wachstum nur über Firmen, die in beiden Zeitpunkten Werte haben.
  So verzerren Indexwechsel und fehlende Firmen das Wachstum nicht.
* Fehlende Werte bleiben fehlend (keine Null).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONCEPTS = {
    "ni": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "rev": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenuesNetOfInterestExpense"],
}
FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT"}


def _duration_facts(cf: dict, tags: list[str]) -> pd.DataFrame:
    """Alle Duration-Facts der ersten passenden Tags (mehrere Tags werden zusammengeführt, frühere Tags haben
    Vorrang je Periode – z. B. wechseln viele Firmen 2018 von SalesRevenueNet auf RevenueFromContract...)."""
    gaap = cf.get("facts", {}).get("us-gaap", {})
    frames = []
    for prio, tag in enumerate(tags):
        units = gaap.get(tag, {}).get("units", {})
        rows = units.get("USD")
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if "start" not in df or df.empty:
            continue
        df = df[df["form"].isin(FORMS)]
        df["prio"] = prio
        frames.append(df[["start", "end", "val", "filed", "form", "prio"]])
    if not frames:
        return pd.DataFrame(columns=["start", "end", "val", "filed", "days"])
    df = pd.concat(frames, ignore_index=True)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.dropna(subset=["start", "end", "filed", "val"])
    df["days"] = (df["end"] - df["start"]).dt.days
    # zuerst gemeldeter Wert je Periode; bei gleichem Datum der bevorzugte Tag
    df = df.sort_values(["start", "end", "filed", "prio"]).drop_duplicates(["start", "end"], keep="first")
    return df


def quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """Quartalswerte (end, val, known) inkl. abgeleitetem 4. Quartal."""
    if df.empty:
        return pd.DataFrame(columns=["end", "val", "known"])
    q = df[(df["days"] >= 80) & (df["days"] <= 100)][["start", "end", "val", "filed"]].rename(columns={"filed": "known"})
    fy = df[(df["days"] >= 350) & (df["days"] <= 380)]
    m9 = df[(df["days"] >= 260) & (df["days"] <= 285)]
    derived = []
    if len(fy) and len(m9):
        m9i = m9.set_index("start")
        for r in fy.itertuples():
            if r.start not in m9i.index:
                continue
            m = m9i.loc[[r.start]].iloc[0]
            if (r.end - m["end"]).days < 80 or (r.end - m["end"]).days > 100:
                continue
            derived.append({"start": m["end"] + pd.Timedelta(days=1), "end": r.end, "val": r.val - m["val"],
                            "known": max(r.filed, m["filed"])})
    if derived:
        d = pd.DataFrame(derived)
        q = pd.concat([q, d[~d["end"].isin(q["end"])]], ignore_index=True)
    # je Quartalsende nur ein Wert (frühest bekannt)
    q = q.sort_values(["end", "known"]).drop_duplicates("end", keep="first")
    return q[["end", "val", "known"]].reset_index(drop=True)


def ttm(q: pd.DataFrame) -> pd.DataFrame:
    """TTM-Reihe: (end, ttm, known)."""
    if len(q) < 4:
        return pd.DataFrame(columns=["end", "ttm", "known"])
    q = q.sort_values("end").reset_index(drop=True)
    rows = []
    for i in range(3, len(q)):
        w = q.iloc[i - 3:i + 1]
        span = (w["end"].iloc[-1] - w["end"].iloc[0]).days
        if 250 <= span <= 290:
            rows.append({"end": w["end"].iloc[-1], "ttm": float(w["val"].sum()), "known": w["known"].max()})
    return pd.DataFrame(rows)


def company_ttm(cf: dict) -> dict[str, pd.DataFrame]:
    return {k: ttm(quarterly(_duration_facts(cf, tags))) for k, tags in CONCEPTS.items()}


def shares_outstanding(cf: dict) -> tuple[float | None, int]:
    """Aktuellste Aktienanzahl (Deckblatt, dei). Rückgabe (Anzahl, Anzahl Aktienklassen am Stichtag)."""
    rows = cf.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares")
    if not rows:
        return None, 0
    df = pd.DataFrame(rows)
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    last = df[df["end"] == df["end"].max()]
    if last.empty or (pd.Timestamp.today() - last["end"].iloc[0]).days > 400:   # veraltet (z. B. Delisting)
        return None, 0
    return float(last["val"].sum()), int(len(last))


def load_companyfacts(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def collect(raw_dir: Path, ciks: list[int], progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Liest alle Company-Facts-Dateien → TTM-Tabelle (cik, kind, end, ttm, known) und Aktienzahlen."""
    rows, shares = [], []
    for n, cik in enumerate(ciks):
        p = raw_dir / "companyfacts" / f"CIK{int(cik):010d}.json.gz"
        if not p.exists():
            continue
        try:
            cf = load_companyfacts(p)
        except Exception:  # noqa: BLE001 - defekte Cache-Datei überspringen
            continue
        for kind, t in company_ttm(cf).items():
            if len(t):
                rows.append(t.assign(cik=int(cik), kind=kind))
        s, ncls = shares_outstanding(cf)
        shares.append({"cik": int(cik), "shares": s, "classes": ncls})
        if progress and n % 50 == 0:
            progress(n)
    t = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["end", "ttm", "known", "cik", "kind"])
    return t, pd.DataFrame(shares)


def _asof(t: pd.DataFrame, date: pd.Timestamp, max_age_days: int = 200) -> pd.Series:
    """Je Firma der neueste TTM-Wert, der am Datum bekannt war und nicht zu alt ist (Index cik)."""
    k = t[(t["known"] <= date) & (t["end"] >= date - pd.Timedelta(days=max_age_days))]
    if k.empty:
        return pd.Series(dtype=float)
    k = k.sort_values(["cik", "end", "known"]).groupby("cik").tail(1)
    return k.set_index("cik")[["end", "ttm"]]


def _value_at(t: pd.DataFrame, date: pd.Timestamp, target_end: pd.Series) -> pd.Series:
    """Je Firma der TTM-Wert für ein Periodenende ~ target_end (± 45 Tage), bekannt am Datum."""
    k = t[t["known"] <= date][["cik", "end", "ttm"]]
    if k.empty or target_end.empty:
        return pd.Series(dtype=float)
    tg = pd.DataFrame({"cik": target_end.index.astype(int), "target": target_end.values})
    m = k.merge(tg, on="cik")
    m["diff"] = (m["end"] - m["target"]).abs().dt.days
    m = m[m["diff"] <= 45]
    if m.empty:
        return pd.Series(dtype=float)
    m = m.sort_values(["cik", "diff"]).drop_duplicates("cik")
    return pd.Series(m["ttm"].values, index=m["cik"].values, dtype=float)


def aggregate(t: pd.DataFrame, dates: pd.DatetimeIndex, min_firms: int = 20) -> pd.DataFrame:
    """Monatliche Aggregate (point-in-time) über alle Firmen der Liste:
    ni_growth / rev_growth: like-for-like TTM-Wachstum gegenüber Vorjahr
    margin: Summe Gewinn / Summe Umsatz (Firmen mit beidem)
    margin_chg: Veränderung der Marge gegenüber Vorjahr (Prozentpunkte, like-for-like)
    e_chain: verketteter Gewinnindex (Monat für Monat like-for-like) – Basis für das KGV
    n_firms: Anzahl Firmen mit aktuellem Gewinnwert."""
    ni, rev = t[t["kind"] == "ni"], t[t["kind"] == "rev"]
    rows, prev_now = [], None
    chain = 1.0
    for d in dates:
        now_ni = _asof(ni, d)
        now_rev = _asof(rev, d)
        r = {"date": d, "n_firms": len(now_ni)}
        if len(now_ni) >= min_firms:
            ya = _value_at(ni, d, now_ni["end"] - pd.DateOffset(years=1))
            common = ya.index.intersection(now_ni.index)
            base = ya[common].sum()
            if len(common) >= min_firms and base > 0:
                r["ni_growth"] = now_ni.loc[common, "ttm"].sum() / base - 1
            # Monatskette
            if prev_now is not None:
                c2 = now_ni.index.intersection(prev_now.index)
                b2 = prev_now.loc[c2, "ttm"].sum()
                if len(c2) >= min_firms and b2 > 0:
                    chain *= now_ni.loc[c2, "ttm"].sum() / b2
            r["e_chain"] = chain
            prev_now = now_ni
        if len(now_rev) >= min_firms:
            ya = _value_at(rev, d, now_rev["end"] - pd.DateOffset(years=1))
            common = ya.index.intersection(now_rev.index)
            base = ya[common].sum()
            if len(common) >= min_firms and base > 0:
                r["rev_growth"] = now_rev.loc[common, "ttm"].sum() / base - 1
            both = now_rev.index.intersection(now_ni.index)
            if len(both) >= min_firms:
                r["margin"] = now_ni.loc[both, "ttm"].sum() / now_rev.loc[both, "ttm"].sum()
                ya_ni = _value_at(ni, d, now_ni.loc[both, "end"] - pd.DateOffset(years=1))
                ya_rev = _value_at(rev, d, now_rev.loc[both, "end"] - pd.DateOffset(years=1))
                cc = ya_ni.index.intersection(ya_rev.index)
                if len(cc) >= min_firms and ya_rev[cc].sum() > 0:
                    r["margin_chg"] = (now_ni.loc[cc, "ttm"].sum() / now_rev.loc[cc, "ttm"].sum()
                                       - ya_ni[cc].sum() / ya_rev[cc].sum())
        rows.append(r)
    out = pd.DataFrame(rows).set_index("date")
    # Wachstum beschleunigt/verlangsamt sich (Veränderung gegenüber vor 3 Monaten)
    if "ni_growth" in out:
        out["ni_growth_chg"] = out["ni_growth"] - out["ni_growth"].shift(3)
    return out


def current_pe(t: pd.DataFrame, shares: pd.DataFrame, prices: dict[str, pd.Series], cik_ticker: dict[int, str],
               date: pd.Timestamp) -> dict:
    """Aktuelles Gesamt-KGV = Summe Börsenwerte / Summe TTM-Gewinne (nur Firmen mit einer Aktienklasse,
    Aktienzahl und Kurs). Mehrklassige Firmen (z. B. Alphabet, Berkshire) werden ausgelassen."""
    now = _asof(t[t["kind"] == "ni"], date)
    caps, earn = 0.0, 0.0
    n = 0
    for r in shares.itertuples():
        if r.classes != 1 or not r.shares or r.cik not in now.index:
            continue
        px = prices.get(cik_ticker.get(r.cik, ""))
        if px is None or px.empty:
            continue
        px = px[px.index <= date]
        if px.empty or (date - px.index[-1]).days > 10:
            continue
        caps += float(px.iloc[-1]) * r.shares
        earn += float(now.loc[r.cik, "ttm"])
        n += 1
    if n < 50 or earn <= 0:
        return {"n": n}
    return {"n": n, "pe": caps / earn, "earnings_yield": earn / caps}


def pe_series(agg: pd.DataFrame, index_px: pd.Series, anchor_pe: float | None) -> pd.Series:
    """KGV-Näherung über die Zeit: Indexstand / verketteter Gewinnindex, auf das aktuelle KGV skaliert.
    Ohne Anker bleibt die Reihe relativ (nur Perzentile sinnvoll)."""
    if "e_chain" not in agg or index_px is None or index_px.empty:
        return pd.Series(dtype=float)
    e = agg["e_chain"].dropna()
    px = index_px.reindex(e.index, method="ffill")
    raw = px / e
    if anchor_pe and len(raw.dropna()):
        raw = raw / raw.dropna().iloc[-1] * anchor_pe
    return raw
