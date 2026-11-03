"""Nikkei-225-Mitglieder (für die Marktbreite des JP225).

Quellen, in dieser Reihenfolge:
  1. Offizielle Komponentenliste von Nikkei Inc. (indexes.nikkei.co.jp)
  2. Wikipedia-Artikel "Nikkei 225" (Codes in der Form "TYO: 7203")
  3. Lokale Datei data/nikkei225_constituents.csv (Spalte 'code', optional 'name') – auch als Snapshot
Yahoo-Finance-Ticker = vierstelliger Code + ".T" (z. B. 7203.T für Toyota).
"""
from __future__ import annotations

import io
import logging
import re
from datetime import datetime, timezone

import httpx
import pandas as pd

from src.config import Settings

log = logging.getLogger(__name__)
SOURCES = [
    ("Nikkei Inc. Komponentenliste", "https://indexes.nikkei.co.jp/en/nkave/index/component"),
    ("Wikipedia: Nikkei 225", "https://en.wikipedia.org/wiki/Nikkei_225"),
]
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml", "Accept-Language": "en-US,en;q=0.9",
}
# TSE-Codes: 4 Zeichen, seit 2024 auch mit Buchstabe am Ende (z. B. 285A = Kioxia)
CODE = r"(\d{3}[0-9A-Z])"
CODE_PATTERNS = [
    re.compile(r"(?:TYO|TSE)\s*[:：]?\s*(?:<[^>]+>\s*)*" + CODE + r"\b"),
    re.compile(r"<td[^>]*>\s*(?:<[^>]+>\s*)*" + CODE + r"\s*(?:</[^>]+>\s*)*</td>"),
    re.compile(r">\s*" + CODE + r"\s*<"),
]


def _plausible(codes: list[str]) -> list[str]:
    codes = [c for c in dict.fromkeys(codes) if re.fullmatch(r"\d{3}[0-9A-Z]", c) and int(c[:3]) >= 130]
    return codes if 200 <= len(codes) <= 260 else []


def codes_from_tables(html: str) -> list[str]:
    """Offizielle Seite: je Branche eine Tabelle mit Spalten 'Code' und 'Company Name'."""
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return []
    codes = []
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "code" in cols:
            col = t.columns[cols.index("code")]
            codes += t[col].astype(str).str.strip().str.upper().tolist()
    return _plausible(codes)


def extract_codes(html: str) -> list[str]:
    """Findet TSE-Codes; akzeptiert nur Ergebnisse in plausibler Grösse (200–260)."""
    codes = codes_from_tables(html)
    if codes:
        return codes
    for pat in CODE_PATTERNS:
        codes = _plausible(pat.findall(html))
        if codes:
            return codes
    return []


def fetch_nikkei_constituents(settings: Settings) -> tuple[list[str], str]:
    csv = settings.data_dir / "nikkei225_constituents.csv"
    errors = []
    for name, url in SOURCES:
        try:
            r = httpx.get(url, headers=BROWSER_HEADERS, timeout=settings.http_timeout, follow_redirects=True)
            r.raise_for_status()
            codes = extract_codes(r.text)
            if codes:
                settings.data_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"code": codes}).to_csv(csv, index=False)
                stamp = datetime.now(timezone.utc).date()
                return [f"{c}.T" for c in codes], f"{name} (abgerufen {stamp}, {len(codes)} Werte)"
            errors.append(f"{name}: keine plausible Liste gefunden")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
    if csv.exists():
        codes = pd.read_csv(csv, dtype=str)["code"].str.extract(r"(\d{3}[0-9A-Z])")[0].dropna().tolist()
        if codes:
            return [f"{c}.T" for c in codes], f"Lokale Liste {csv.name} ({len(codes)} Werte)"
    log.warning("Nikkei-225-Liste nicht verfügbar: %s", "; ".join(errors))
    return [], "nicht verfügbar (" + "; ".join(errors)[:300] + ")"
