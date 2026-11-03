"""Pentagon-Pizza-Index (Konflikt-Frühwarnung): python -m src.geo.pizza

Idee: Vor grösseren Militäreinsätzen wird im Pentagon nachts durchgearbeitet – die Pizzerien in der Nähe sind dann
ungewöhnlich voll. PizzINT (pizzint.watch) misst das über Google-Maps-Auslastung und fasst es als DOUGHCON-Stufe
zusammen (5 = ruhig … 1 = höchste Alarmstufe).

Der PC ruft die Daten ab (Cloud-Rechner dürfen die Seite nicht direkt abfragen), hängt jede Messung an
data/geo/pizza_history.parquet an und speichert die letzte Messung in data/geo/pizza_latest.json. Der Cloud-Sync lädt
beides hoch. Es gibt keine offizielle Schnittstelle: Die Felder werden tolerant gelesen, und wenn die JSON-Daten nicht
erreichbar sind, wird die DOUGHCON-Stufe aus der Startseite gelesen. Unterhaltungs-/OSINT-Projekt – kein validiertes
Signal; der eigene Verlauf entsteht erst mit dem Sammeln.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone

import httpx
import pandas as pd

from src.config import get_settings
from src.utils import atomic_write_text

BASE = "https://www.pizzint.watch"
SPIKE_PCT = 50          # Auslastung ≥ 50 % über normal = Ausschlag
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/130.0 Safari/537.36", "Accept": "application/json,text/html;q=0.9,*/*;q=0.8"}
LEVELS = {5: "ruhig", 4: "erhöht", 3: "aufmerksam", 2: "hoch", 1: "maximal"}


def geo_dir():
    d = get_settings().data_dir / "geo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find(obj, keys):
    """Ersten Wert zu einem der Schlüssel suchen (rekursiv, Gross/Klein egal)."""
    keys = {k.lower() for k in keys}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in keys and v is not None:
                return v
        for v in obj.values():
            r = _find(v, keys)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find(v, keys)
            if r is not None:
                return r
    return None


def _num(v):
    try:
        return float(str(v).replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def parse_json(js) -> dict:
    level = _num(_find(js, ["defcon_level", "doughcon_level", "doughcon", "defcon", "level"]))
    places = _find(js, ["pizzerias", "places", "locations", "restaurants"])
    rows = []
    for p in places if isinstance(places, list) else []:
        if not isinstance(p, dict):
            continue
        cur = _num(_find(p, ["current_busyness", "current_popularity", "busyness", "current"]))
        norm = _num(_find(p, ["normal_busyness", "usual_popularity", "normal", "usual"]))
        pct = _num(_find(p, ["percentage_above_normal", "pct_above_normal", "above_normal"]))
        if pct is None and cur is not None and norm:
            pct = (cur / norm - 1) * 100
        is_open = _find(p, ["is_open", "open"])
        rows.append({"name": str(_find(p, ["name", "title"]) or "?"), "distance": _find(p, ["distance"]),
                     "current": cur, "normal": norm, "pct_above": pct,
                     "is_open": bool(is_open) if is_open is not None else None})
    return {"level": int(level) if level and 1 <= level <= 5 else None, "places": rows,
            "source_updated": _find(js, ["last_updated", "updated_at", "timestamp"]),
            "alert": _find(js, ["alert_message", "alert"]), "source": "json"}


def parse_html(html: str) -> dict:
    m = re.search(r"DOUGHCON\s*(?:<[^>]*>\s*)*([1-5])", html, re.I)
    return {"level": int(m.group(1)) if m else None, "places": [], "source_updated": None, "alert": None,
            "source": "html"}


def fetch(client: httpx.Client | None = None) -> dict:
    own = client is None
    client = client or httpx.Client(timeout=30, headers=HEADERS, follow_redirects=True)
    try:
        try:
            r = client.get(f"{BASE}/api/dashboard-data", params={"_t": int(time.time() * 1000)})
            if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
                out = parse_json(r.json())
                if out["level"] is not None or out["places"]:
                    return out
        except (httpx.HTTPError, ValueError):
            pass
        r = client.get(BASE + "/")
        r.raise_for_status()
        return parse_html(r.text)
    finally:
        if own:
            client.close()


def summarize(d: dict, now: datetime) -> dict:
    pl = pd.DataFrame(d["places"])
    pct = pl["pct_above"].dropna() if "pct_above" in pl else pd.Series(dtype=float)
    return {"taken_at": now, "level": d["level"], "n_places": len(pl),
            "n_open": int(pl["is_open"].fillna(False).sum()) if "is_open" in pl else None,
            "avg_pct_above": float(pct.mean()) if len(pct) else None,
            "max_pct_above": float(pct.max()) if len(pct) else None,
            "n_spikes": int((pct >= SPIKE_PCT).sum()) if len(pct) else 0, "source": d["source"]}


def load_history() -> pd.DataFrame:
    p = geo_dir() / "pizza_history.parquet"
    if not p.exists():
        return pd.DataFrame()
    h = pd.read_parquet(p)
    h["taken_at"] = pd.to_datetime(h["taken_at"], utc=True)
    return h


def collect(client: httpx.Client | None = None, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    d = fetch(client)
    row = summarize(d, now)
    h = pd.concat([load_history(), pd.DataFrame([row])], ignore_index=True)
    h["taken_at"] = pd.to_datetime(h["taken_at"], utc=True)
    h.to_parquet(geo_dir() / "pizza_history.parquet", index=False)
    latest = {**row, "taken_at": now.isoformat(timespec="seconds"), "places": d["places"], "alert": d["alert"],
              "source_updated": d["source_updated"]}
    atomic_write_text(geo_dir() / "pizza_latest.json", json.dumps(latest, ensure_ascii=False, default=str))
    return latest


def main() -> int:
    try:
        r = collect()
    except Exception as exc:  # noqa: BLE001
        print(f"Pizza-Index nicht abrufbar: {exc}", file=sys.stderr)
        return 1
    print(f"Pizza-Index: DOUGHCON {r['level']} ({LEVELS.get(r['level'], '?')}), {r['n_places']} Pizzerien, "
          f"{r['n_spikes']} Ausschläge, Quelle {r['source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
