"""AnalyticsFX-API (https://www.analytics-fx.com/docs/api): Gamma Exposure (GEX) und Zeta-Regime.

Der API-Schlüssel kommt ausschliesslich aus der Umgebung bzw. der Datei .env (ANALYTICSFX_API_KEY) und wird nie
gespeichert, protokolliert oder angezeigt.

Kosten laut Doku: GEX 2 Credits, Zeta 1 Credit. Fehler 400 (keine Optionskette) und 502 (Dienst down) werden laut
Anbieter nicht berechnet.
"""
from __future__ import annotations

import os
import time

import httpx

import src.config  # noqa: F401  (lädt .env)
from src.config import PROJECT_ROOT

try:  # zusätzlich gex.env (eigene Datei nur für den API-Schlüssel); .env hat Vorrang
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / "gex.env", override=False)
except ImportError:  # pragma: no cover
    pass

BASE = "https://www.analytics-fx.com/api/v1"


class AFXError(RuntimeError):
    pass


def api_key() -> str | None:
    k = (os.getenv("ANALYTICSFX_API_KEY") or "").strip().strip('"').strip("'")
    # Tippfehler abfangen, z. B. "afx_live afx_live_…" oder Leerzeichen: das letzte afx_-Wort zählt
    parts = [p for p in k.split() if p.startswith("afx_") and len(p) > 12]
    return parts[-1] if parts else (k or None)


def symbols() -> list[str]:
    raw = os.getenv("GEX_SYMBOLS", "SPY,QQQ,SPX")
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


ZETA_ASSETS = {"SPY", "QQQ", "DAX"}


class AFXClient:
    def __init__(self, key: str | None = None, transport: httpx.BaseTransport | None = None, timeout: float = 60):
        key = key or api_key()
        if not key:
            raise AFXError("Kein API-Schlüssel: In der Datei .env die Zeile ANALYTICSFX_API_KEY=afx_live_... eintragen.")
        self.client = httpx.Client(base_url=BASE, timeout=timeout, transport=transport,
                                   headers={"Authorization": f"Bearer {key}", "Accept": "application/json",
                                            "User-Agent": "sp500-dashboard-gex/1.0"})

    def _get(self, path: str, params: dict | None = None) -> dict:
        last = ""
        for attempt in range(3):
            try:
                r = self.client.get(path, params=params)
            except httpx.HTTPError as exc:
                last = str(exc)
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (401, 403):
                raise AFXError("API-Schlüssel ungültig oder ohne Berechtigung (HTTP %d)." % r.status_code)
            if r.status_code == 402:
                raise AFXError("Keine Credits mehr übrig (HTTP 402).")
            if r.status_code == 400:
                try:
                    msg = r.json().get("error", {})
                except ValueError:
                    msg = r.text[:200]
                raise AFXError(f"Anfrage abgelehnt: {msg}")
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                last = "HTTP 429 (zu viele Anfragen)"
                continue
            last = f"HTTP {r.status_code}"
            time.sleep(2 * (attempt + 1))
        raise AFXError(f"{path}: {last}")

    def gex(self, symbol: str, expiration: str | None = None) -> dict:
        return self._get(f"/gex/{symbol}", {"expiration": expiration} if expiration else None)

    def zeta(self, asset: str) -> dict:
        return self._get(f"/zeta/{asset}")

    def close(self) -> None:
        self.client.close()
