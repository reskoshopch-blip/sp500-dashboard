"""Normalisierung der SEC-Company-Facts-JSON zu Bilanz-Zeitreihen.

Kernregeln (siehe auch docs/METHODIK.md):
  1. Bilanzwerte sind Instant-Facts (nur 'end', kein 'start'). Duration-Facts (GuV, Cashflow)
     werden verworfen – damit können auch keine Quartalswerte aufsummiert werden.
  2. Das Geschäftsjahresende (Stichtag) eines Jahresberichts ist das 'end'-Datum seiner
     Assets-Facts (Fallback: LiabilitiesAndStockholdersEquity, StockholdersEquity).
  3. Das Geschäftsjahr-Label stammt aus dem Feld 'fy' des Jahresberichts mit diesem Stichtag
     (DocumentFiscalYearFocus des Unternehmens). Ist es unplausibel, gilt: Stichtag in Jan/Feb ->
     Vorjahr, sonst Kalenderjahr des Stichtags.
  4. Ein Stichtag erscheint in mehreren Berichten (aktuelles Jahr + Vorjahresvergleich, 10-K/A).
     Verwendet wird der zuletzt eingereichte Wert ("latest as reported", berücksichtigt Restatements).
     Der ursprünglich gemeldete Wert, die Anzahl Meldungen und ein Restatement-Flag werden gespeichert.
  5. Nur Währungseinheiten (ISO-Code). Werte werden nicht umgerechnet; Aggregationen nutzen nur USD.
  6. Fehlende Werte bleiben fehlend (NaN/NA), 0 ist ein echter gemeldeter Nullwert.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.normalization.taxonomy import ITEMS

ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A", "20-F", "20-F/A", "40-F", "40-F/A"}
QUARTERLY_FORMS = {"10-Q", "10-Q/A"}
TAXONOMIES = ("us-gaap", "ifrs-full", "srt")
PERIOD_ANCHOR_TAGS = ("Assets", "LiabilitiesAndStockholdersEquity", "StockholdersEquity")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
RESTATEMENT_TOLERANCE = 0.5  # USD – Rundungsdifferenzen gelten nicht als Restatement


@dataclass
class CompanyResult:
    cik: int
    entity_name: str
    periods: pd.DataFrame            # Jahresberichte: accn, form, filed, period_end, fy_reported, fiscal_year, fy_source
    annual_all: pd.DataFrame         # alle Instant-Facts (alle Tags) pro Geschäftsjahr, dedupliziert
    annual_items: pd.DataFrame       # kanonische Positionen pro Geschäftsjahr
    latest_items: pd.DataFrame       # kanonische Positionen zum neuesten Bilanzstichtag (10-Q oder 10-K)
    issues: list[str] = field(default_factory=list)


def flatten_instant_facts(companyfacts: dict) -> pd.DataFrame:
    """Alle Instant-Facts in Währungseinheiten als Long-Format."""
    rows = []
    facts = companyfacts.get("facts", {}) or {}
    for taxonomy in TAXONOMIES:
        for tag, payload in (facts.get(taxonomy) or {}).items():
            label = payload.get("label") or tag
            for unit, entries in (payload.get("units") or {}).items():
                if not CURRENCY_RE.match(unit):
                    continue
                for e in entries:
                    if "start" in e:          # Duration-Fact -> nicht Bilanz
                        continue
                    if e.get("val") is None or e.get("end") is None:
                        continue
                    rows.append((taxonomy, tag, label, unit, e["end"], e["val"], e.get("accn"), e.get("fy"),
                                 e.get("fp"), e.get("form"), e.get("filed"), e.get("frame")))
    df = pd.DataFrame(rows, columns=["taxonomy", "tag", "label", "unit", "end", "val", "accn", "fy", "fp",
                                     "form", "filed", "frame"])
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
    return df.dropna(subset=["end", "val"])


def fallback_fiscal_year(end: pd.Timestamp) -> int:
    return end.year - 1 if end.month <= 2 else end.year


def detect_periods(df: pd.DataFrame, forms: set[str]) -> pd.DataFrame:
    """Stichtag jedes Berichts = 'end' seiner Assets-Facts (bzw. Fallback-Tag)."""
    sub = df[df["form"].isin(forms) & (df["taxonomy"] != "dei")]
    out = []
    for accn, g in sub.groupby("accn"):
        period_end = None
        for anchor in PERIOD_ANCHOR_TAGS:
            a = g[g["tag"] == anchor]
            if not a.empty:
                period_end = a["end"].max()
                break
        if period_end is None:
            continue
        first = g.sort_values("filed").iloc[0]
        out.append({"accn": accn, "form": first["form"], "filed": g["filed"].min(), "period_end": period_end,
                    "fy_reported": first["fy"], "fp": first["fp"]})
    return pd.DataFrame(out, columns=["accn", "form", "filed", "period_end", "fy_reported", "fp"])


def assign_fiscal_years(periods: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    issues: list[str] = []
    if periods.empty:
        return periods.assign(fiscal_year=pd.Series(dtype="Int64"), fy_source=pd.Series(dtype=str)), issues
    p = periods.sort_values("filed").copy()
    fys, srcs = [], []
    for _, r in p.iterrows():
        end = r["period_end"]
        fy = r["fy_reported"]
        if pd.notna(fy) and int(fy) in (end.year, end.year - 1) and not (int(fy) == end.year - 1 and end.month > 3):
            fys.append(int(fy))
            srcs.append("DEI fy")
        else:
            fb = fallback_fiscal_year(end)
            if pd.notna(fy):
                issues.append(f"Unplausibles fy={int(fy)} für Stichtag {end.date()} ({r['accn']}) – Regel ergibt {fb}")
            fys.append(fb)
            srcs.append("Regel (Stichtag)")
    p["fiscal_year"] = pd.array(fys, dtype="Int64")
    p["fy_source"] = srcs
    # Ein Stichtag -> ein Geschäftsjahr (frühester Bericht entscheidet)
    first_per_end = p.groupby("period_end")["fiscal_year"].first()
    p["fiscal_year"] = p["period_end"].map(first_per_end).astype("Int64")
    # Geschäftsjahresende geändert? Gleiches Label mit mehreren Stichtagen
    dup = p.groupby("fiscal_year")["period_end"].nunique()
    for fy in dup[dup > 1].index:
        ends = sorted(p.loc[p["fiscal_year"] == fy, "period_end"].unique())
        issues.append(f"Geschäftsjahr {fy} hat mehrere Stichtage {[str(pd.Timestamp(e).date()) for e in ends]} "
                      "(Umstellung des Geschäftsjahres) – der späteste wird verwendet")
    return p, issues


def add_comparative_periods(df: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    """Vorjahresstichtage, für die kein eigener Jahresbericht vorliegt (z. B. Jahr vor dem Börsengang),
    werden aus der Vergleichsspalte des Folgeberichts übernommen: Geschäftsjahr = fy des Berichts − 1."""
    if periods.empty:
        return periods
    known_ends = set(periods["period_end"])
    known_fys = set(periods["fiscal_year"].dropna().astype(int))
    extra = []
    anchors = df[df["tag"].isin(PERIOD_ANCHOR_TAGS)]
    by_accn = {k: g for k, g in anchors.groupby("accn")}
    for _, p in periods.sort_values("filed", ascending=False).iterrows():
        anchor = by_accn.get(p["accn"])
        if anchor is None:
            continue
        for end in sorted(set(anchor["end"]) - {p["period_end"]}):
            gap = (p["period_end"] - end).days
            fy = int(p["fiscal_year"]) - 1
            if 330 <= gap <= 400 and end not in known_ends and fy not in known_fys:
                extra.append({"accn": p["accn"], "form": p["form"], "filed": p["filed"], "period_end": end,
                              "fy_reported": p["fy_reported"], "fp": p["fp"], "fiscal_year": fy,
                              "fy_source": "Vergleichsspalte Folgebericht"})
                known_ends.add(end)
                known_fys.add(fy)
    if not extra:
        return periods
    out = pd.concat([periods, pd.DataFrame(extra)], ignore_index=True)
    out["fiscal_year"] = out["fiscal_year"].astype("Int64")
    return out


def _dedupe_versions(facts: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Pro Schlüssel den zuletzt eingereichten Wert wählen; Originalwert und Restatement-Flag behalten."""
    if facts.empty:
        return facts.assign(original_value=[], original_accn=[], original_filed=[], n_versions=[],
                            restated=[], conflict=[])
    f = facts.sort_values(keys + ["filed", "accn"])
    g = f.groupby(keys, sort=False)
    latest = g.tail(1).set_index(keys)
    first = g.head(1).set_index(keys)
    stats = g["val"].agg(n_versions="count", vmin="min", vmax="max")
    # Konflikt: innerhalb desselben (letzten) Berichts mehrere unterschiedliche Werte
    last_accn = latest["accn"]
    same = f.merge(last_accn.rename("last_accn").reset_index(), on=keys)
    same = same[same["accn"] == same["last_accn"]]
    conflict = same.groupby(keys)["val"].nunique() > 1
    out = latest.copy()
    out["original_value"] = first["val"]
    out["original_accn"] = first["accn"]
    out["original_filed"] = first["filed"]
    out["n_versions"] = stats["n_versions"]
    out["restated"] = (stats["vmax"] - stats["vmin"]).abs() > RESTATEMENT_TOLERANCE
    out["conflict"] = conflict.reindex(out.index).fillna(False).astype(bool)
    return out.reset_index()


def build_annual_all(df: pd.DataFrame, periods: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    if df.empty or periods.empty:
        return pd.DataFrame()
    # Pro Geschäftsjahr der späteste Stichtag (falls Geschäftsjahr umgestellt wurde)
    fy_end = periods.groupby("fiscal_year")["period_end"].max()
    fy_end = fy_end[fy_end.index.isin(years)]
    end_to_fy = {v: k for k, v in fy_end.items()}
    ann = df[df["form"].isin(ANNUAL_FORMS) & df["end"].isin(list(end_to_fy))].copy()
    if ann.empty:
        return pd.DataFrame()
    ann["fiscal_year"] = ann["end"].map(end_to_fy).astype("Int64")
    out = _dedupe_versions(ann, ["taxonomy", "tag", "unit", "fiscal_year"])
    out = out.rename(columns={"val": "value", "end": "period_end"})
    return out[["fiscal_year", "period_end", "taxonomy", "tag", "label", "unit", "value", "accn", "form", "filed",
                "original_value", "original_accn", "original_filed", "n_versions", "restated", "conflict"]]


def select_canonical(all_facts: pd.DataFrame, period_cols: list[str]) -> pd.DataFrame:
    """Wählt pro Position den ersten verfügbaren Kandidaten-Tag (nur us-gaap, nur USD)."""
    cols = period_cols + ["item_key", "value", "unit", "tag", "accn", "form", "filed", "original_value", "original_filed",
                          "n_versions", "restated", "conflict", "is_derived", "derivation", "tag_rank"]
    if all_facts.empty:
        return pd.DataFrame(columns=cols)
    usd = all_facts[(all_facts["taxonomy"] == "us-gaap") & (all_facts["unit"] == "USD")]
    parts = []
    for item in ITEMS:
        rank = {t: i for i, t in enumerate(item.tags)}
        cand = usd[usd["tag"].isin(rank)].copy()
        if cand.empty:
            continue
        cand["tag_rank"] = cand["tag"].map(rank)
        best = cand.sort_values(period_cols + ["tag_rank"]).groupby(period_cols, as_index=False).head(1)
        best = best.assign(item_key=item.key, is_derived=False, derivation="")
        parts.append(best)
    if not parts:
        return pd.DataFrame(columns=cols)
    res = pd.concat(parts, ignore_index=True)
    return res[[c for c in cols if c in res.columns]]


def build_latest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp | None, str | None]:
    """Neuester verfügbarer Bilanzstichtag aus 10-Q oder 10-K (optionaler Quartalsmodus)."""
    periods = detect_periods(df, ANNUAL_FORMS | QUARTERLY_FORMS)
    if periods.empty:
        return pd.DataFrame(), None, None
    latest_end = periods["period_end"].max()
    accns = periods.loc[periods["period_end"] == latest_end].sort_values("filed")
    form = accns.iloc[-1]["form"]
    sub = df[(df["end"] == latest_end) & df["accn"].isin(accns["accn"])].copy()
    sub["period_end_key"] = latest_end
    dedup = _dedupe_versions(sub, ["taxonomy", "tag", "unit", "period_end_key"]).rename(columns={"val": "value"})
    return dedup, latest_end, form


def normalize_company(companyfacts: dict, years: tuple[int, ...]) -> CompanyResult:
    from src.normalization.derive import add_derived_items  # zirkuläre Imports vermeiden
    from src.normalization.schemas import validate_companyfacts

    validate_companyfacts(companyfacts)  # wirft ValidationError bei unerwartetem Antwortformat
    cik = int(companyfacts.get("cik", 0))
    name = companyfacts.get("entityName", "")
    df = flatten_instant_facts(companyfacts)
    issues: list[str] = []
    if df.empty:
        issues.append("Keine Instant-Facts in Währungseinheiten vorhanden")
        empty = pd.DataFrame()
        return CompanyResult(cik, name, empty, empty, empty, empty, issues)
    if not (df["taxonomy"] == "us-gaap").any():
        issues.append("Keine us-gaap-Facts (evtl. IFRS-Filer) – Standardpositionen nicht verfügbar")

    periods = detect_periods(df, ANNUAL_FORMS)
    periods, p_issues = assign_fiscal_years(periods)
    issues += p_issues
    periods = add_comparative_periods(df, periods)

    annual_all = build_annual_all(df, periods, years)
    annual_items = select_canonical(annual_all, ["fiscal_year", "period_end"]) if not annual_all.empty else \
        select_canonical(pd.DataFrame(), ["fiscal_year", "period_end"])
    annual_items = add_derived_items(annual_items, ["fiscal_year", "period_end"])

    latest_raw, latest_end, latest_form = build_latest(df)
    if not latest_raw.empty:
        latest_raw = latest_raw.rename(columns={"period_end_key": "period_end"})
        latest_items = select_canonical(latest_raw, ["period_end"])
        latest_items = add_derived_items(latest_items, ["period_end"])
        latest_items["source_form"] = latest_form
    else:
        latest_items = pd.DataFrame()

    nonusd = annual_all[annual_all["unit"] != "USD"]["unit"].unique().tolist() if not annual_all.empty else []
    if nonusd:
        issues.append(f"Werte in Fremdwährung gemeldet ({', '.join(nonusd)}) – nicht umgerechnet, nicht aggregiert")
    for d in (annual_all, annual_items, latest_items, periods):
        if not d.empty:
            d.insert(0, "cik", cik)
    return CompanyResult(cik, name, periods, annual_all, annual_items, latest_items, issues)


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
