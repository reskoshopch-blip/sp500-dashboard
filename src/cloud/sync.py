"""Ergebnisse vom PC in den privaten Supabase-Bucket hochladen: python -m src.cloud.sync [--force]

Hochgeladen werden nur die fertigen Ergebnisse, die das Dashboard anzeigt (Datenbank komprimiert, ML-, Fundamental-,
COT- und GEX-Ergebnisse) – keine Rohdaten, keine Kurse, keine Schlüssel. Nur geänderte Dateien werden übertragen
(Vergleich per SHA-256). Zum Schluss wird manifest.json geschrieben; die Cloud-App lädt danach genau diese Dateien.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.cloud.storage import CloudError, Storage
from src.config import get_settings
from src.utils import atomic_write_text

TOP_FILES = ["sp500_balance_sheets.duckdb", "pipeline_status.json", "constituents_snapshot.csv",
             "constituents_changes_snapshot.csv", "nikkei225_constituents.csv"]
FOLDERS = {"ml": ("*.parquet", "*.json"), "ml/models": ("*.joblib",), "fundamental": ("*.parquet", "*.json"),
           "cot": ("*.parquet", "*.json"), "gex": ("*.parquet", "*.json"), "geo": ("*.parquet", "*.json")}
GZIP = {"sp500_balance_sheets.duckdb"}
SKIP = {"fundamental/cot.parquet", "fundamental/cot_markets.parquet"}   # Rohdaten, von der App nicht benötigt


def select_files(data_dir: Path) -> list[Path]:
    files = [data_dir / f for f in TOP_FILES if (data_dir / f).exists()]
    for folder, patterns in FOLDERS.items():
        d = data_dir / folder
        if d.is_dir():
            for pat in patterns:
                files += [p for p in d.glob(pat) if p.is_file()]
    raw = data_dir / "gex" / "raw"                       # je Symbol nur der neueste GEX-Schnappschuss (Profil)
    if raw.is_dir():
        for sym in raw.iterdir():
            snaps = sorted(f for f in sym.glob("*.json") if not f.stem.endswith("_zeta"))
            if snaps:
                files.append(snaps[-1])
    return sorted({f for f in files if f.relative_to(data_dir).as_posix() not in SKIP})


def _refresh_gex_model(data_dir: Path) -> None:
    """GEX-Auswertung (model.json) nachrechnen, falls sie fehlt oder älter als die Schnappschüsse ist."""
    snaps, model = data_dir / "gex" / "snapshots.parquet", data_dir / "gex" / "model.json"
    if snaps.exists() and (not model.exists() or model.stat().st_mtime < snaps.stat().st_mtime):
        try:
            from src.gex.learn import run as gex_learn
            gex_learn()
        except Exception as exc:  # noqa: BLE001 - Upload trotzdem fortsetzen
            print(f"  Hinweis: GEX-Auswertung nicht möglich ({exc})")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sync(force: bool = False, storage: Storage | None = None, progress=print, only: str | None = None) -> dict:
    s = get_settings()
    data_dir = s.data_dir
    local_manifest = data_dir / ".cloud_manifest.json"
    old = {}
    if local_manifest.exists() and not force:
        try:
            old = {f["path"]: f for f in json.loads(local_manifest.read_text(encoding="utf-8"))["files"]}
        except (ValueError, KeyError):
            old = {}
    if only and not old:          # ohne bisheriges Manifest immer alles hochladen
        only = None
    if not only:
        _refresh_gex_model(data_dir)
    own = storage is None
    storage = storage or Storage.from_env()
    entries, uploaded, total = [], [], 0
    try:
        files = select_files(data_dir)
        if only:
            files = [p for p in files if p.relative_to(data_dir).as_posix().startswith(only.rstrip("/") + "/")]
            mine = {p.relative_to(data_dir).as_posix() for p in files}
            entries = [e for r, e in old.items() if r not in mine and (data_dir / r).exists()]
        for p in files:
            rel = p.relative_to(data_dir).as_posix()
            digest = sha256(p)
            gz = rel in GZIP
            remote = rel + (".gz" if gz else "")
            entry = {"path": rel, "remote": remote, "sha256": digest, "gzip": gz, "size": p.stat().st_size,
                     "mtime": p.stat().st_mtime}
            if old.get(rel, {}).get("sha256") != digest:
                data = p.read_bytes()
                if gz:
                    data = gzip.compress(data, 6)
                storage.upload(remote, data)
                uploaded.append(rel)
                total += len(data)
                progress(f"  hochgeladen: {rel} ({len(data) / 1e6:.1f} MB)")
            entries.append(entry)
        manifest = {"synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": entries}
        storage.upload("manifest.json", json.dumps(manifest).encode("utf-8"), "application/json")
        atomic_write_text(local_manifest, json.dumps(manifest))
    finally:
        if own:
            storage.close()
    return {"uploaded": uploaded, "n_files": len(entries), "mb": total / 1e6, "synced_at": manifest["synced_at"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Dashboard-Ergebnisse nach Supabase hochladen")
    ap.add_argument("--force", action="store_true", help="alle Dateien neu hochladen")
    ap.add_argument("--only", help="nur diesen Ordner hochladen (z.B. geo), Rest des Manifests bleibt")
    a = ap.parse_args(argv)
    try:
        r = sync(a.force, only=a.only)
    except CloudError as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 2
    print(f"Cloud-Sync fertig: {len(r['uploaded'])} von {r['n_files']} Dateien hochgeladen ({r['mb']:.1f} MB).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
