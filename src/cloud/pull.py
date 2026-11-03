"""Cloud-Ansicht: lädt die vom PC synchronisierten Ergebnisse aus Supabase in einen lokalen Ordner.

Aktiv, wenn DASHBOARD_CLOUD=1 gesetzt ist (Streamlit-Cloud-Secrets). DATA_DIR zeigt dann auf einen beschreibbaren
temporären Ordner; nur geänderte Dateien werden geladen.
"""
from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path

from src.cloud.storage import Storage


def is_cloud() -> bool:
    return (os.getenv("DASHBOARD_CLOUD") or "").strip().lower() in {"1", "true", "ja", "yes"}


def default_data_dir() -> Path:
    return Path(tempfile.gettempdir()) / "sp500_dashboard_data"


def pull(data_dir: Path, storage: Storage | None = None) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    own = storage is None
    storage = storage or Storage.from_env()
    try:
        manifest = json.loads(storage.download("manifest.json"))
        local = data_dir / ".cloud_manifest.json"
        have = {}
        if local.exists():
            try:
                have = {f["path"]: f["sha256"] for f in json.loads(local.read_text(encoding="utf-8"))["files"]}
            except (ValueError, KeyError):
                have = {}
        changed = []
        todo = [f for f in manifest["files"]
                if not (have.get(f["path"]) == f["sha256"] and (data_dir / f["path"]).exists())]
        blobs = storage.download_many([f["remote"] for f in todo]) if todo and hasattr(storage, "download_many") \
            else {}
        for f in todo:
            target = data_dir / f["path"]
            data = blobs[f["remote"]] if f["remote"] in blobs else storage.download(f["remote"])
            if f.get("gzip"):
                data = gzip.decompress(data)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, target)
            changed.append(f["path"])
        local.write_text(json.dumps(manifest), encoding="utf-8")
    finally:
        if own:
            storage.close()
    return {"changed": changed, "synced_at": manifest.get("synced_at"), "n_files": len(manifest["files"])}
