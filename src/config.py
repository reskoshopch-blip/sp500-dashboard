"""Zentrale Konfiguration. Werte kommen aus Umgebungsvariablen bzw. der Datei .env.

Es werden keine Zugangsdaten im Quellcode gespeichert. Die SEC verlangt lediglich einen
aussagekräftigen User-Agent mit Kontaktadresse (SEC Fair Access Policy).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


class ConfigError(RuntimeError):
    """Verständliche Konfigurationsfehler für Endbenutzer."""


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Die Einstellung {name}={raw!r} ist keine ganze Zahl.") from exc


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Die Einstellung {name}={raw!r} ist keine Zahl.") from exc


@dataclass(frozen=True)
class Settings:
    sec_user_agent: str
    data_dir: Path
    requests_per_second: float
    max_cache_age_days: int
    fiscal_years: tuple[int, ...]
    keep_raw_cache: bool
    http_timeout: float
    max_retries: int
    constituents_url: str
    local_constituents_csv: Path
    ml_years: tuple[int, ...] = ()
    ml_horizon_days: int = 365
    extra_years_for_growth: int = 1  # Vorjahr laden, damit das Wachstum im ersten Jahr berechenbar ist
    paths: dict = field(default_factory=dict)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "sp500_balance_sheets.duckdb"

    @property
    def status_path(self) -> Path:
        return self.data_dir / "pipeline_status.json"

    @property
    def ml_dir(self) -> Path:
        return self.data_dir / "ml"

    @property
    def load_years(self) -> tuple[int, ...]:
        """Dashboard-Jahre plus ML-Trainingsjahre (jeweils mit Vorjahr für Wachstumsraten)."""
        all_years = set(self.fiscal_years) | set(self.ml_years)
        first = min(all_years) - self.extra_years_for_growth
        return tuple(range(first, max(all_years) + 1))

    def require_user_agent(self) -> str:
        ua = self.sec_user_agent.strip()
        if not ua or any(p in ua.lower() for p in ("example.com", "beispiel.ch", "vorname nachname")) or not re.search(r"[^@\s]+@[^@\s]+\.[^@\s]+", ua):
            raise ConfigError(
                "SEC_USER_AGENT ist nicht gesetzt oder enthält keine echte E-Mail-Adresse.\n"
                "Die SEC verlangt einen User-Agent der Form 'Name E-Mail', z. B.\n"
                "  SEC_USER_AGENT=Max Muster max.muster@beispiel.ch\n"
                "Trage ihn in die Datei .env im Projektordner ein (Vorlage: .env.example)."
            )
        return ua


def _years(name: str, default: str) -> tuple[int, ...]:
    raw = os.getenv(name, default).strip()
    m = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", raw)
    if m:
        return tuple(range(int(m.group(1)), int(m.group(2)) + 1))
    try:
        years = tuple(sorted(int(y) for y in raw.split(",") if y.strip()))
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} ist ungültig. Beispiel: 2021-2025") from exc
    if not years:
        raise ConfigError(f"{name} ist leer. Beispiel: 2021-2025")
    return years


def get_settings() -> Settings:
    years = _years("FISCAL_YEARS", "2021-2025")
    ml_years = _years("ML_YEARS", "2010-2025")
    rps = _float("SEC_REQUESTS_PER_SECOND", 5.0)
    if rps > 10:
        raise ConfigError("SEC_REQUESTS_PER_SECOND darf laut SEC Fair Access Policy höchstens 10 sein.")
    data_dir = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return Settings(
        sec_user_agent=os.getenv("SEC_USER_AGENT", ""),
        data_dir=data_dir,
        requests_per_second=rps,
        max_cache_age_days=_int("MAX_CACHE_AGE_DAYS", 7),
        fiscal_years=years,
        keep_raw_cache=os.getenv("KEEP_RAW_CACHE", "true").strip().lower() in {"1", "true", "yes", "ja"},
        http_timeout=_float("HTTP_TIMEOUT_SECONDS", 60.0),
        max_retries=_int("HTTP_MAX_RETRIES", 5),
        constituents_url=os.getenv(
            "SP500_CONSTITUENTS_URL", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        ),
        local_constituents_csv=data_dir / "sp500_constituents.csv",
        ml_years=ml_years,
        ml_horizon_days=_int("ML_HORIZON_DAYS", 365),
    )
