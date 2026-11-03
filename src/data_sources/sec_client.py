"""Rate-limitierter, cachender Client für die öffentlichen SEC-EDGAR-APIs.

Endpunkte (kein API-Key nötig):
  * https://www.sec.gov/files/company_tickers.json            Ticker -> CIK
  * https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json   alle XBRL-Facts eines Unternehmens
  * https://data.sec.gov/api/xbrl/companyconcept/CIK##########/us-gaap/<Tag>.json
  * https://data.sec.gov/submissions/CIK##########.json      Stammdaten, SIC-Code, Filings

SEC Fair Access: max. 10 Anfragen/Sekunde, aussagekräftiger User-Agent mit Kontakt.
"""
from __future__ import annotations

import gzip
import json
import logging
import threading
import time
from email.utils import formatdate
from pathlib import Path
from typing import Any

import httpx

from src.config import Settings

log = logging.getLogger(__name__)

SEC_DATA = "https://data.sec.gov"
SEC_WWW = "https://www.sec.gov"


class SecError(RuntimeError):
    pass


class SecNotFound(SecError):
    pass


def cik10(cik: int | str) -> str:
    return str(int(str(cik).strip())).zfill(10)


def filing_index_url(cik: int | str, accn: str) -> str:
    """Link zur Filing-Indexseite im SEC-Archiv (Originalbericht inkl. 10-K-Dokument)."""
    return f"{SEC_WWW}/Archives/edgar/data/{int(str(cik))}/{accn.replace('-', '')}/{accn}-index.htm"


class RateLimiter:
    def __init__(self, per_second: float):
        self.min_interval = 1.0 / max(per_second, 0.1)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self.min_interval


class SecClient:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.limiter = RateLimiter(settings.requests_per_second)
        self.client = httpx.Client(
            headers={
                "User-Agent": settings.require_user_agent(),
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=settings.http_timeout,
            follow_redirects=True,
            transport=transport,
        )
        self.cache_dir = settings.raw_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- HTTP
    def _get(self, url: str, if_modified_since: float | None = None) -> httpx.Response | None:
        """GET mit Rate Limit, Retries und Backoff. Gibt None bei 304 zurück."""
        headers = {}
        if if_modified_since:
            headers["If-Modified-Since"] = formatdate(if_modified_since, usegmt=True)
        last_exc: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            self.limiter.wait()
            try:
                resp = self.client.get(url, headers=headers)
            except httpx.HTTPError as exc:
                last_exc = exc
                delay = min(2 ** attempt, 60)
                log.warning("Netzwerkfehler bei %s (%s) – neuer Versuch in %ss", url, exc, delay)
                time.sleep(delay)
                continue
            if resp.status_code == 304:
                return None
            if resp.status_code == 404:
                raise SecNotFound(f"Nicht gefunden (404): {url}")
            if resp.status_code == 403:
                raise SecError(
                    f"SEC verweigert den Zugriff (403) auf {url}. Häufige Ursache: fehlender oder "
                    "ungültiger SEC_USER_AGENT, oder das Netzwerk blockiert sec.gov."
                )
            if resp.status_code in (429, 500, 502, 503, 504):
                delay = min(2 ** (attempt + 1), 120)
                log.warning("HTTP %s bei %s – Backoff %ss", resp.status_code, url, delay)
                time.sleep(delay)
                last_exc = SecError(f"HTTP {resp.status_code}")
                continue
            resp.raise_for_status()
            return resp
        raise SecError(f"Abruf fehlgeschlagen nach {self.settings.max_retries + 1} Versuchen: {url} ({last_exc})")

    # --------------------------------------------------------------- Cache
    def _cache_path(self, kind: str, key: str) -> Path:
        d = self.cache_dir / kind
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.json.gz"

    def _cached_json(self, kind: str, key: str, url: str, max_age_days: float, force: bool = False) -> dict[str, Any]:
        path = self._cache_path(kind, key)
        age_ok = path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400
        if path.exists() and age_ok and not force:
            return _read_gz_json(path)
        resp = self._get(url, if_modified_since=path.stat().st_mtime if path.exists() and not force else None)
        if resp is None:  # 304 Not Modified -> Cache bleibt gültig
            path.touch()
            return _read_gz_json(path)
        data = resp.json()
        if self.settings.keep_raw_cache or kind != "companyfacts":
            _write_gz_json(path, data)
        return data

    # ---------------------------------------------------------- Endpunkte
    def company_tickers(self, force: bool = False) -> dict[str, Any]:
        return self._cached_json("meta", "company_tickers", f"{SEC_WWW}/files/company_tickers.json", 1, force)

    def company_facts(self, cik: int | str, force: bool = False) -> dict[str, Any]:
        c = cik10(cik)
        return self._cached_json(
            "companyfacts", f"CIK{c}", f"{SEC_DATA}/api/xbrl/companyfacts/CIK{c}.json",
            self.settings.max_cache_age_days, force,
        )

    def company_concept(self, cik: int | str, tag: str, taxonomy: str = "us-gaap") -> dict[str, Any]:
        """Unabhängiger Endpunkt – wird für die stichprobenartige Verifikation genutzt (nicht gecacht)."""
        c = cik10(cik)
        resp = self._get(f"{SEC_DATA}/api/xbrl/companyconcept/CIK{c}/{taxonomy}/{tag}.json")
        assert resp is not None
        return resp.json()

    def submissions(self, cik: int | str, force: bool = False) -> dict[str, Any]:
        c = cik10(cik)
        return self._cached_json(
            "submissions", f"CIK{c}", f"{SEC_DATA}/submissions/CIK{c}.json",
            self.settings.max_cache_age_days, force,
        )

    def close(self) -> None:
        self.client.close()


def _read_gz_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _write_gz_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(data, fh)
    tmp.replace(path)  # atomar: abgebrochene Downloads hinterlassen keine kaputten Cache-Dateien
