"""Einstiegspunkt: streamlit run app.py"""
from __future__ import annotations

import hmac
import os
from pathlib import Path

import streamlit as st

# Streamlit-Cloud: Secrets als Umgebungsvariablen bereitstellen (lokal gibt es keine Secrets-Datei)
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)) and _k not in os.environ:
            os.environ[_k] = str(_v)
except Exception:  # noqa: BLE001
    pass

from src.cloud.pull import default_data_dir, is_cloud, pull  # noqa: E402

if is_cloud():
    os.environ.setdefault("DATA_DIR", str(default_data_dir()))

from src.components.data import db_exists
from src.components.ui import CSS
from src.config import ConfigError, get_settings
from src.pages import (compare, consensus, cot, data_admin, fundamental, gex, method, ml_backtest, ml_effects, ml_forecast, ml_regime, ml_terminal, ml_top,
                       overview, pizza,
                       profile, quality, sectors, table)

st.set_page_config(page_title="S&P 500 Balance Sheet Intelligence", page_icon="📊", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


def password_gate() -> None:
    """Optionaler Passwortschutz (DASHBOARD_PASSWORD in den Secrets bzw. der Umgebung)."""
    pw = os.getenv("DASHBOARD_PASSWORD", "")
    if not pw or st.session_state.get("auth_ok"):
        return
    st.title("Anmelden")
    with st.form("login"):
        given = st.text_input("Passwort", type="password")
        if st.form_submit_button("Öffnen", type="primary"):
            if hmac.compare_digest(given.encode(), pw.encode()):
                st.session_state["auth_ok"] = True
                st.rerun()
            st.error("Falsches Passwort.")
    st.stop()


@st.cache_resource(ttl=600, show_spinner="Lade aktuelle Daten aus der Cloud …")
def cloud_pull(data_dir: str) -> dict:
    return pull(Path(data_dir))


password_gate()
if is_cloud():
    try:
        info = cloud_pull(os.environ["DATA_DIR"])
        st.sidebar.caption(f"☁️ Cloud-Ansicht · Daten vom PC, Stand {info.get('synced_at', '–')[:16].replace('T', ' ')} UTC")
    except Exception as exc:  # noqa: BLE001 - mit vorhandenen Daten weiter
        st.sidebar.error(f"Cloud-Daten konnten nicht geladen werden: {exc}")

try:
    get_settings()
except ConfigError as exc:
    st.error(f"Konfigurationsfehler:\n\n{exc}")
    st.stop()

admin = st.Page(data_admin.page, title="Daten laden", icon=":material/cloud_download:", url_path="daten")
meth = st.Page(method.page, title="Methodik & Quellen", icon=":material/menu_book:", url_path="methodik")

if db_exists():
    pages = {
        "Analyse": [
            st.Page(overview.page, title="Übersicht", icon=":material/dashboard:", default=True),
            st.Page(compare.page, title="Unternehmensvergleich", icon=":material/compare_arrows:", url_path="vergleich"),
            st.Page(sectors.page, title="Branchen", icon=":material/factory:", url_path="branchen"),
            st.Page(profile.page, title="Unternehmensprofil", icon=":material/business:", url_path="profil"),
            st.Page(table.page, title="Datentabelle", icon=":material/table:", url_path="tabelle"),
        ],
        "Markt & Machine Learning": [
            st.Page(ml_terminal.page, title="Markt-Terminal", icon=":material/monitor_heart:", url_path="terminal"),
            st.Page(fundamental.page, title="Fundamental-Terminal", icon=":material/account_balance:",
                    url_path="fundamental"),
            st.Page(cot.page, title="COT-Signale", icon=":material/groups:", url_path="cot"),
            st.Page(gex.page, title="Gamma Exposure (GEX)", icon=":material/bolt:", url_path="gex"),
            st.Page(pizza.page, title="Pizza-Index (Konflikt)", icon=":material/local_pizza:", url_path="pizza"),
            st.Page(consensus.page, title="Modell-Konsens S&P 500", icon=":material/how_to_vote:", url_path="konsens"),
            st.Page(ml_forecast.page, title="ML-Prognosen", icon=":material/insights:", url_path="ml-prognosen"),
            st.Page(ml_backtest.page, title="ML-Backtest", icon=":material/history:", url_path="ml-backtest"),
            st.Page(ml_effects.page, title="Wirkung von Bilanzentwicklungen", icon=":material/science:",
                    url_path="ml-wirkung"),
            st.Page(ml_regime.page, title="Marktregime", icon=":material/show_chart:", url_path="marktregime"),
            st.Page(ml_top.page, title="Top-Warnung", icon=":material/warning:", url_path="top-warnung"),
        ],
        "Daten": [st.Page(quality.page, title="Datenqualität", icon=":material/fact_check:", url_path="qualitaet"),
                  admin, meth],
    }
else:
    admin = st.Page(data_admin.page, title="Daten laden", icon=":material/cloud_download:", url_path="daten",
                    default=True)
    pages = {"Daten": [admin, meth]}

st.navigation(pages).run()
