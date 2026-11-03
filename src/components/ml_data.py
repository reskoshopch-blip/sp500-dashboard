"""Zugriff auf die ML-Ergebnisse (data/ml/*.parquet) für die Oberfläche."""
from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from src.config import get_settings
from src.ml.model import FEATURE_SETS, TARGETS

DISCLAIMER = ("Statistische Auswertung historischer Daten – keine Anlageberatung. Vergangene Zusammenhänge "
              "müssen sich nicht wiederholen. Ich bin kein Finanzberater.")
TABLES = ("oos", "folds", "regimes", "calibration", "importance", "pdp", "live", "drivers", "buckets", "regime",
          "dataset")


def ml_version() -> float:
    p = get_settings().ml_dir / "meta.json"
    return p.stat().st_mtime if p.exists() else 0.0


def ml_ready() -> bool:
    return ml_version() > 0


@st.cache_data(show_spinner=False)
def load_ml(version: float) -> dict:
    d = get_settings().ml_dir
    out = {"meta": json.loads((d / "meta.json").read_text(encoding="utf-8"))}
    for t in TABLES:
        p = d / f"{t}.parquet"
        out[t] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    return out


@st.cache_resource(show_spinner=False)
def load_model(version: float, target: str, fset: str):
    p = get_settings().ml_dir / "models" / f"{target}_{fset}.joblib"
    return joblib.load(p) if p.exists() else None


def ml_status() -> dict | None:
    p = get_settings().ml_dir / "status.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def flabel(meta: dict, f: str) -> str:
    return meta.get("feature_labels", {}).get(f, f)


def pick(df: pd.DataFrame, target: str, fset: str) -> pd.DataFrame:
    if df.empty or "target" not in df:
        return df
    return df[(df["target"] == target) & (df["fset"] == fset)]


def selectors(key: str, with_fset: bool = True) -> tuple[str, str]:
    c1, c2 = st.columns(2)
    target = c1.selectbox("Was soll vorhergesagt werden?", list(TARGETS), key=f"{key}_t",
                          format_func=lambda k: TARGETS[k]["label"])
    fset = "bilanz"
    if with_fset:
        fset = c2.selectbox("Merkmale", list(FEATURE_SETS), key=f"{key}_f",
                            format_func=lambda k: FEATURE_SETS[k]["label"],
                            help="'Bilanz + Marktregime' zeigt, was die Bilanz allein erklärt. Mit Kursverlauf kommen "
                                 "Momentum und Volatilität der Aktie bis zum Filing-Datum dazu.")
    return target, fset


def require_ml() -> dict | None:
    if not ml_ready():
        st.info("Noch keine ML-Ergebnisse. Auf der Seite **Daten laden** unten „Kurse laden + Modelle trainieren“ "
                "starten (oder ml.bat ausführen).")
        return None
    return load_ml(ml_version())


def top_version(key: str = "SPY") -> float:
    p = get_settings().ml_dir / f"top_{key}_meta.json"
    return p.stat().st_mtime if p.exists() else 0.0


def top_markets() -> list[str]:
    from src.ml.markets import MARKETS
    return [k for k in MARKETS if top_version(k) > 0]


@st.cache_data(show_spinner=False)
def load_top(version: float, key: str = "SPY") -> dict | None:
    d = get_settings().ml_dir
    if not (d / f"top_{key}_meta.json").exists():
        return None
    out = {"meta": json.loads((d / f"top_{key}_meta.json").read_text(encoding="utf-8"))}
    for t in ("daily", "episodes", "coef", "red"):
        p = d / f"top_{key}_{t}.parquet"
        out[t] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    if len(out["daily"]):
        out["daily"]["date"] = pd.to_datetime(out["daily"]["date"])
    return out


@st.cache_data(show_spinner=False)
def load_health(version: float, key: str) -> dict | None:
    d = get_settings().ml_dir
    p = d / f"health_{key}_daily.parquet"
    if not p.exists():
        return None
    daily = pd.read_parquet(p)
    daily["date"] = pd.to_datetime(daily["date"])
    bt = pd.read_parquet(d / f"health_{key}_bt.parquet") if (d / f"health_{key}_bt.parquet").exists() else pd.DataFrame()
    return {"daily": daily, "bt": bt}
