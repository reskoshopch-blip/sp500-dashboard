"""Datenabruf aus der Oberfläche starten und den Fortschritt verfolgen."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import streamlit as st

from src.components.data import db_exists, pipeline_status
from src.components.fund_data import fund_status
from src.components.ml_data import ml_status
from src.config import PROJECT_ROOT, ConfigError, get_settings


def start_module(module: str, args: list[str], logname: str) -> None:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    log = open(settings.data_dir / logname, "a", encoding="utf-8")  # noqa: SIM115
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    subprocess.Popen([sys.executable, "-m", module, *args], cwd=str(PROJECT_ROOT), stdout=log,
                     stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"}, **kw)


def start_pipeline(args: list[str]) -> None:
    start_module("src.pipeline", args, "pipeline.log")


@st.fragment(run_every=3)
def ml_panel() -> None:
    s = ml_status()
    if not s:
        st.caption("Noch kein ML-Lauf gestartet.")
        return
    st.write(f"**Status:** {s.get('state')} · Schritt: {s.get('step')} · aktualisiert {s.get('updated_at', '')} UTC")
    total, done = s.get("total") or 0, s.get("done") or 0
    if total and s.get("state") == "läuft":
        st.progress(min(done / total, 1.0), text=f"{done} / {total}")
    if s.get("state") == "fertig":
        st.success(f"Fertig – {s.get('n_samples')} Firmenjahre im Datensatz"
                   + (f", für {s.get('prices_failed')} Ticker keine Kurse" if s.get("prices_failed") else "")
                   + ". Seite neu laden (F5), dann die ML-Seiten links öffnen.")
    elif s.get("state") == "Fehler":
        st.error(s.get("step"))
    log = get_settings().data_dir / "ml.log"
    if log.exists():
        with st.expander("ML-Protokoll (letzte Zeilen)"):
            st.code("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]))


@st.fragment(run_every=3)
def progress_panel() -> None:
    s = pipeline_status()
    if not s:
        st.caption("Noch kein Datenabruf gestartet.")
        return
    state = s.get("state")
    total, done = s.get("total") or 0, s.get("done") or 0
    st.write(f"**Status:** {state} · Schritt: {s.get('step')} · aktualisiert {s.get('updated_at', '')} UTC")
    if total:
        st.progress(min(done / total, 1.0), text=f"{done} / {total} Unternehmen")
    if s.get("counts"):
        st.caption(" · ".join(f"{k}: {v}" for k, v in s["counts"].items()))
    if state == "fertig":
        st.success(f"Fertig – {s.get('loaded')} von {s.get('universe')} Unternehmen mit Daten. "
                   "Seite neu laden (F5), um die neuen Daten zu sehen.")
    elif state == "Fehler":
        st.error(s.get("step"))
    log = get_settings().data_dir / "pipeline.log"
    if log.exists():
        with st.expander("Protokoll (letzte Zeilen)"):
            st.code("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]))


@st.fragment(run_every=3)
def fund_panel() -> None:
    s = fund_status()
    if not s:
        st.caption("Noch keine Fundamentalanalyse gestartet.")
        return
    st.write(f"**Status:** {s.get('state')} · Schritt: {s.get('step')} · aktualisiert {s.get('updated_at', '')} UTC")
    if s.get("state") == "fertig":
        st.success(f"Fertig – {s.get('fred_ok', 0)} Makroreihen, {s.get('cot_rows', 0)} COT-Wochen"
                   + (f", {s['fred_failed']} Reihen nicht erreichbar" if s.get("fred_failed") else "")
                   + ". Seite neu laden (F5), dann links „Fundamental-Terminal“ öffnen.")
    elif s.get("state") == "Fehler":
        st.error(s.get("step"))
    log = get_settings().data_dir / "fundamental.log"
    if log.exists():
        with st.expander("Protokoll (letzte Zeilen)"):
            st.code("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]))


def cot_status() -> dict | None:
    p = get_settings().data_dir / "cot" / "status.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None


@st.fragment(run_every=3)
def cot_panel() -> None:
    s = cot_status()
    if not s:
        st.caption("Noch kein COT-Lauf gestartet.")
        return
    st.write(f"**Status:** {s.get('state')} · Schritt: {s.get('step')} · aktualisiert {s.get('updated_at', '')} UTC")
    if s.get("state") == "fertig":
        st.success("Fertig. Seite neu laden (F5), dann links „COT-Signale“ öffnen.")
    elif s.get("state") == "Fehler":
        st.error(s.get("step"))
    log = get_settings().data_dir / "cot.log"
    if log.exists():
        with st.expander("Protokoll (letzte Zeilen)"):
            st.code("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]))


def page() -> None:
    st.title("Daten laden und aktualisieren")
    from src.cloud.pull import is_cloud
    if is_cloud():
        man = get_settings().data_dir / ".cloud_manifest.json"
        when = json.loads(man.read_text(encoding="utf-8")).get("synced_at", "–") if man.exists() else "–"
        st.info("☁️ **Cloud-Ansicht.** Daten werden auf deinem PC berechnet und von dort hochgeladen "
                f"(letzter Upload: {when[:16].replace('T', ' ')} UTC). Abrufe und Training startest du auf dem PC "
                "(start.bat); danach erscheinen die neuen Ergebnisse hier innerhalb von ca. 10 Minuten.")
        return
    try:
        settings = get_settings()
        ua = settings.require_user_agent()
        st.caption(f"SEC User-Agent: `{ua}` · Rate Limit: {settings.requests_per_second}/s · "
                   f"Cache-Gültigkeit: {settings.max_cache_age_days} Tage · Datenordner: `{settings.data_dir}`")
    except ConfigError as exc:
        st.error(str(exc))
        st.stop()
    if not db_exists():
        st.info("Es ist noch keine Datenbank vorhanden. Starte zuerst einen Testlauf oder den vollständigen Abruf. "
                "Der vollständige Abruf lädt ca. 500 Company-Facts-Dateien von der SEC (je nach Verbindung "
                "15–40 Minuten) und kann jederzeit unterbrochen und fortgesetzt werden.")
    s = pipeline_status()
    running = False
    if s and s.get("state") == "läuft":
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(s["updated_at"])).total_seconds()
        except (KeyError, ValueError):
            age = 0
        running = age < 600  # kein Lebenszeichen seit 10 Min. -> Lauf gilt als abgebrochen
        if not running:
            st.warning("Der letzte Abruf wurde offenbar unterbrochen. Ein neuer Lauf setzt dort fort, wo er aufgehört hat.")
    c1, c2, c3 = st.columns(3)
    if c1.button("Testlauf (20 Unternehmen)", disabled=running, width="stretch"):
        start_pipeline(["--limit", "20"])
        st.rerun()
    if c2.button("Alle Unternehmen (inkrementell)", disabled=running, type="primary", width="stretch"):
        start_pipeline([])
        st.rerun()
    if c3.button("Alles neu laden (Cache ignorieren)", disabled=running, width="stretch"):
        start_pipeline(["--force"])
        st.rerun()
    with st.form("tickers"):
        t = st.text_input("Nur bestimmte Ticker laden (kommagetrennt)", placeholder="AAPL, MSFT, JPM")
        if st.form_submit_button("Laden", disabled=running) and t.strip():
            start_pipeline(["--tickers", t.replace(" ", "")])
            st.rerun()
    if running:
        st.caption("Ein Abruf läuft. Die Datenbank wird erst am Ende ersetzt – das Dashboard bleibt nutzbar.")
    progress_panel()

    st.divider()
    st.subheader("Machine Learning: Kurse laden und Modelle trainieren")
    st.caption("Lädt tägliche Kurse (Yahoo Finance, bereinigt um Splits/Dividenden) für alle Unternehmen und SPY, baut "
               "den Point-in-time-Datensatz 2010–2025, testet die Modelle im Walk-forward-Backtest und erstellt die "
               "aktuellen Prognosen. Dauer: Kurse ca. 2–5 Min., Training ca. 3–10 Min. Voraussetzung: Bilanzdaten "
               "sind geladen.")
    ms = ml_status()
    ml_running = False
    if ms and ms.get("state") == "läuft":
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(ms["updated_at"])).total_seconds()
        except (KeyError, ValueError):
            age = 0
        ml_running = age < 900
    c1, c2, c3 = st.columns(3)
    if c3.button("Markt-Analysen berechnen (Terminal + Top-Warnung)", disabled=ml_running or running or not db_exists(),
                 width="stretch", help="Lädt Indizes mit langer Historie, VIX, Kreditmarkt, USD/JPY, Zinsen, Rohstoffe, "
                                       "Sektor-ETFs und die 225 Nikkei-Aktien und berechnet Markt-Terminal und Top-Warnung. Beim ersten Mal 5–10 Min., "
                                       "danach 1–3 Min."):
        start_module("src.ml.run", ["--top-only"], "ml.log")
        st.rerun()
    if c1.button("Kurse laden + Modelle trainieren", type="primary", disabled=ml_running or running or not db_exists(),
                 width="stretch"):
        start_module("src.ml.run", [], "ml.log")
        st.rerun()
    if c2.button("Nur neu trainieren (ohne Kursabruf)", disabled=ml_running or running or not db_exists(),
                 width="stretch"):
        start_module("src.ml.run", ["--skip-prices"], "ml.log")
        st.rerun()
    ml_panel()

    st.divider()
    st.subheader("Fundamentalanalyse (Makro, Notenbanken, Gewinne, COT)")
    st.caption("Lädt rund 30 Makroreihen von FRED (USA und Japan), die COT-Positionierung der CFTC, den S&P-500-Kursindex "
               "und wertet die Gewinne aller S&P-500-Firmen aus den SEC-Daten aus. Danach lernt das Terminal Richtung und "
               "Gewicht jedes Indikators und testet die Prognosen. Erster Lauf ca. 3–8 Minuten, danach 2–4 Minuten.")
    fs = fund_status()
    f_running = False
    if fs and fs.get("state") == "läuft":
        try:
            f_running = (datetime.now(timezone.utc) - datetime.fromisoformat(fs["updated_at"])).total_seconds() < 900
        except (KeyError, ValueError):
            f_running = False
    if st.button("Fundamentalanalyse berechnen", type="primary", disabled=f_running or running, width="stretch"):
        start_module("src.fundamental.run", [], "fundamental.log")
        st.rerun()
    fund_panel()

    st.divider()
    st.subheader("COT-Modell (Gold, Öl, Kupfer, S&P 500, Nasdaq 100)")
    st.caption("Lädt die Commitments-of-Traders-Daten der CFTC und die Kurse und lernt je Markt den COT-Score. "
               "Ca. 1–3 Minuten. Neue COT-Daten erscheinen jeden Freitag.")
    cs = cot_status()
    c_running = False
    if cs and cs.get("state") == "läuft":
        try:
            c_running = (datetime.now(timezone.utc) - datetime.fromisoformat(cs["updated_at"])).total_seconds() < 600
        except (KeyError, ValueError):
            c_running = False
    if st.button("COT-Modell berechnen", disabled=c_running, width="stretch"):
        start_module("src.cot.run", [], "cot.log")
        st.rerun()
    cot_panel()
