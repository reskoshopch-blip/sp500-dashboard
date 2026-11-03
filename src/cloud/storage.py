"""Supabase Storage (privater Bucket) – Hoch- und Herunterladen der Dashboard-Ergebnisse.

Zugangsdaten nur aus der Umgebung: SUPABASE_URL, SUPABASE_SECRET_KEY (Secret bzw. service_role Key), optional
SUPABASE_BUCKET (Standard dashboard-data). Auf dem PC stehen sie in cloud.env, in der Streamlit Cloud unter
„Secrets“. Der Schlüssel wird nie protokolliert oder angezeigt.
"""
from __future__ import annotations

import os
import time

import httpx

from src.config import PROJECT_ROOT

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / "cloud.env", override=False)
except ImportError:  # pragma: no cover
    pass


class CloudError(RuntimeError):
    pass


def creds() -> tuple[str, str, str] | None:
    url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    bucket = (os.getenv("SUPABASE_BUCKET") or "dashboard-data").strip()
    return (url, key, bucket) if url and key else None


class Storage:
    def __init__(self, url: str, key: str, bucket: str, transport: httpx.BaseTransport | None = None):
        self.bucket = bucket
        self.client = httpx.Client(base_url=f"{url}/storage/v1", timeout=httpx.Timeout(120, connect=20),
                                   transport=transport,
                                   headers={"Authorization": f"Bearer {key}", "apikey": key})

    @classmethod
    def from_env(cls) -> "Storage":
        url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
        token = (os.getenv("DASHBOARD_SYNC_TOKEN") or "").strip()
        if url and token:
            return GatewayStorage(url, token)
        c = creds()
        if not c:
            raise CloudError("Keine Cloud-Zugangsdaten: SUPABASE_URL und DASHBOARD_SYNC_TOKEN in cloud.env eintragen.")
        return cls(*c)

    def _retry(self, fn):
        last = None
        for attempt in range(4):
            try:
                r = fn()
            except httpx.HTTPError as exc:
                last = str(exc) or type(exc).__name__
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code in (200, 201):
                return r
            if r.status_code in (401, 403):
                raise CloudError(f"Supabase verweigert den Zugriff (HTTP {r.status_code}) – Secret Key prüfen.")
            if r.status_code in (400, 404) and "not found" in r.text.lower():
                raise FileNotFoundError(r.text[:200])
            if r.status_code == 413:
                raise CloudError("Datei zu gross für Supabase (max. 50 MB).")
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            time.sleep(2 * (attempt + 1))
        raise CloudError(last or "unbekannter Fehler")

    def upload(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        self._retry(lambda: self.client.post(f"/object/{self.bucket}/{path}", content=data,
                                             headers={"Content-Type": content_type, "x-upsert": "true",
                                                      "cache-control": "no-cache"}))

    def download(self, path: str) -> bytes:
        return self._retry(lambda: self.client.get(f"/object/authenticated/{self.bucket}/{path}")).content

    def close(self) -> None:
        self.client.close()


class GatewayStorage(Storage):
    """Zugang über die Edge Function dashboard-storage: eigener Zugangscode statt Supabase-Secret-Key.
    Die Funktion gibt kurzlebige signierte URLs aus; Daten fliessen direkt zwischen PC/App und Supabase Storage."""

    def __init__(self, url: str, token: str, transport: httpx.BaseTransport | None = None):
        self.fn = f"{url}/functions/v1/dashboard-storage"
        self.token = token
        self.client = httpx.Client(timeout=httpx.Timeout(180, connect=20), transport=transport)

    def _sign(self, action: str, path: str) -> str:
        r = self._retry(lambda: self.client.post(self.fn, json={"action": action, "paths": [path]},
                                                  headers={"x-dashboard-token": self.token}))
        item = r.json()["urls"][0]
        if not item.get("signedUrl"):
            raise FileNotFoundError(item.get("error") or path)
        return item["signedUrl"]

    def upload(self, path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        url = self._sign("upload", path)
        self._retry(lambda: self.client.put(url, content=data, headers={"Content-Type": content_type,
                                                                        "x-upsert": "true"}))

    def download(self, path: str) -> bytes:
        url = self._sign("download", path)
        return self._retry(lambda: self.client.get(url)).content

    def download_many(self, paths: list[str], workers: int = 8) -> dict[str, bytes]:
        """Viele Dateien auf einmal: signierte URLs in Blöcken à 100, Downloads parallel."""
        from concurrent.futures import ThreadPoolExecutor
        urls: dict[str, str] = {}
        for i in range(0, len(paths), 100):
            chunk = paths[i:i + 100]
            r = self._retry(lambda: self.client.post(self.fn, json={"action": "download", "paths": chunk},
                                                      headers={"x-dashboard-token": self.token}))
            for item in r.json()["urls"]:
                if not item.get("signedUrl"):
                    raise FileNotFoundError(item.get("error") or item.get("path"))
                urls[item["path"]] = item["signedUrl"]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            data = list(ex.map(lambda p: self._retry(lambda: self.client.get(urls[p])).content, paths))
        return dict(zip(paths, data))
