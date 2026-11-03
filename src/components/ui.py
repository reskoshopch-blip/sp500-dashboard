"""Gemeinsame UI-Bausteine: Theme, Filterkontext, Kennzahlen-Kacheln, Exporte."""
from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src.components.data import db_version, load_all
from src.config import get_settings

CSS = """
<style>
  .block-container {padding-top: 2.2rem; max-width: 1400px;}
  div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); border-radius: 10px; padding: 12px 14px;}
  div[data-testid="stMetricLabel"] p {font-size: .82rem; opacity: .85;}
  .note {font-size: .82rem; opacity: .75; margin-top: -.4rem;}
  .pill {display:inline-block; padding: 1px 8px; border-radius: 999px; font-size: .75rem;
         border: 1px solid rgba(128,128,128,.35); margin-right: 4px;}
  .pill.ok {border-color:#0ca30c;} .pill.warn {border-color:#fab219;} .pill.na {border-color:#d03b3b;}
</style>
"""


def theme_mode() -> str:
    try:
        t = st.context.theme.type
        return "light" if t == "light" else "dark"
    except Exception:  # noqa: BLE001
        return "dark"


@dataclass
class Ctx:
    data: dict
    years: list[int]
    sectors: list[str]
    membership_only: bool
    mode: str
    universe: set[int]           # CIKs im gewählten Universum (Sektorfilter)
    items: pd.DataFrame          # gefiltert auf Jahre, Universum, ggf. Mitgliedschaft
    metrics: pd.DataFrame
    companies: pd.DataFrame


def sidebar_context() -> Ctx:
    data = load_all(db_version())
    settings = get_settings()
    comps = data["companies"]
    years_all = sorted(set(settings.fiscal_years))
    st.sidebar.markdown("### Filter")
    years = st.sidebar.multiselect("Geschäftsjahre", years_all, default=years_all, key="f_years")
    years = sorted(years) or years_all
    sectors_all = sorted(comps["sector"].unique())
    sectors = st.sidebar.multiselect("Branchen (GICS-Sektor)", sectors_all, default=[], key="f_sectors",
                                     placeholder="Alle Branchen")
    has_hist = data["membership"]["was_member"].notna().any() if len(data["membership"]) else False
    membership_only = st.sidebar.toggle(
        "Nur Jahre mit Indexmitgliedschaft", value=False, disabled=not has_hist, key="f_member",
        help="Nutzt die aus der Änderungshistorie rekonstruierte Zusammensetzung zum Geschäftsjahresende. "
             "Aus: aktuelle Zusammensetzung wird rückwirkend verwendet.")
    if not has_hist:
        st.sidebar.caption("Keine Änderungshistorie verfügbar – aktuelle Zusammensetzung wird rückwirkend verwendet.")
    universe_df = comps if not sectors else comps[comps["sector"].isin(sectors)]
    universe = set(universe_df["cik"])
    items = data["items"]
    items = items[items["fiscal_year"].isin(years) & items["cik"].isin(universe)]
    metrics = data["metrics"]
    metrics = metrics[metrics["fiscal_year"].isin(years) & metrics["cik"].isin(universe)]
    if membership_only:
        mem = data["membership"]
        keep = mem[mem["was_member"] == True][["cik", "fiscal_year"]]  # noqa: E712
        items = items.merge(keep, on=["cik", "fiscal_year"])
        metrics = metrics.merge(keep, on=["cik", "fiscal_year"])
    meta = data["meta"]
    st.sidebar.divider()
    st.sidebar.caption(f"Datenbank erstellt: {meta.get('built_at', '–')} UTC  \n"
                       f"Mitgliederliste: {meta.get('constituents_source', '–')}  \n"
                       f"Hell/Dunkel: Menü ⋮ → Settings → Theme")
    return Ctx(data, years, sectors, membership_only, theme_mode(), universe, items, metrics,
               universe_df)


def company_picker(ctx: Ctx, label: str, key: str, multi: bool = True, default: list[str] | None = None):
    comps = ctx.companies.sort_values("primary_ticker")
    loaded = comps[comps["fetch_status"] == "ok"]
    opts = loaded["label"].tolist()
    if multi:
        tick = set(default or [])
        dflt = loaded[loaded["ticker"].map(lambda t: bool(set(t.split("/")) & tick))]["label"].tolist()
        dflt = dflt or opts[: min(5, len(opts))]
        sel = st.multiselect(label, opts, default=dflt, key=key)
        return loaded[loaded["label"].isin(sel)]
    idx = 0
    if default:
        hits = loaded.reset_index(drop=True)
        m = hits[hits["ticker"].map(lambda t: bool(set(t.split("/")) & set(default)))]
        idx = int(m.index[0]) if len(m) else 0
    sel = st.selectbox(label, opts, index=idx if opts else None, key=key)
    return loaded[loaded["label"] == sel]


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            out = df.copy()
            for c in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[c]):
                    out[c] = out[c].dt.tz_localize(None) if getattr(out[c].dt, "tz", None) else out[c]
            out.to_excel(xw, sheet_name=name[:31], index=False)
    return buf.getvalue()


def export_buttons(df: pd.DataFrame, basename: str, key: str) -> None:
    c1, c2, _ = st.columns([1, 1, 4])
    c1.download_button("CSV exportieren", df.to_csv(index=False).encode("utf-8-sig"), f"{basename}.csv",
                       "text/csv", key=f"{key}_csv", width="stretch")
    c2.download_button("Excel exportieren", to_excel_bytes({basename: df}), f"{basename}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{key}_xlsx",
                       width="stretch")


def plot(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}})


def na(v, fmt=None) -> str:
    if v is None or pd.isna(v):
        return "Nicht verfügbar"
    return fmt(v) if fmt else str(v)
