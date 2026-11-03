"""Cloud-Sync: PC → Supabase (nachgebildet) → Cloud-Ansicht, nur geänderte Dateien, Passwortschutz."""
from __future__ import annotations

import json

import httpx
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.cloud import pull as P
from src.cloud import sync as S
from src.cloud.storage import Storage


def fake_storage(store: dict, log: list):
    def handler(req: httpx.Request):
        assert req.headers["Authorization"] == "Bearer sb_secret_test"
        path = req.url.path
        log.append((req.method, path))
        if req.method == "POST":
            store[path.split("/object/dashboard-data/")[1]] = req.content
            return httpx.Response(200, json={"Key": path})
        key = path.split("/object/authenticated/dashboard-data/")[1]
        if key not in store:
            return httpx.Response(400, json={"error": "not_found", "message": "Object not found"})
        return httpx.Response(200, content=store[key])
    return Storage("https://x.supabase.co", "sb_secret_test", "dashboard-data", transport=httpx.MockTransport(handler))


def _data(d):
    (d / "ml" / "models").mkdir(parents=True)
    (d / "gex" / "raw" / "SPY").mkdir(parents=True)
    (d / "fundamental" / "fred").mkdir(parents=True)
    (d / "sp500_balance_sheets.duckdb").write_bytes(b"DUCK" * 10000)
    pd.DataFrame({"a": [1, 2]}).to_parquet(d / "ml" / "health_SPY_daily.parquet")
    (d / "ml" / "models" / "m.joblib").write_bytes(b"model")
    (d / "gex" / "raw" / "SPY" / "20261101_220000.json").write_text("{}")
    (d / "gex" / "raw" / "SPY" / "20261102_220000.json").write_text('{"neu": 1}')
    (d / "fundamental" / "fred" / "DGS10.parquet").write_bytes(b"raw")   # Rohdaten: nicht hochladen
    (d / "pipeline.log").write_text("log")                                # Logs: nicht hochladen


def test_sync_and_pull_roundtrip(tmp_path, monkeypatch):
    src = tmp_path / "pc"
    src.mkdir()
    _data(src)
    monkeypatch.setenv("DATA_DIR", str(src))
    store, log = {}, []
    r = S.sync(storage=fake_storage(store, log), progress=lambda *_: None)
    assert "sp500_balance_sheets.duckdb.gz" in store and "manifest.json" in store
    assert not any("fred" in k or k.endswith(".log") for k in store)
    assert "gex/raw/SPY/20261102_220000.json" in store and "gex/raw/SPY/20261101_220000.json" not in store
    assert len(store["sp500_balance_sheets.duckdb.gz"]) < 40000          # komprimiert
    # zweiter Lauf: nichts geändert -> nur manifest
    log.clear()
    r2 = S.sync(storage=fake_storage(store, log), progress=lambda *_: None)
    assert r2["uploaded"] == [] and [p for m, p in log if m == "POST"] == ["/storage/v1/object/dashboard-data/manifest.json"]
    # Cloud-Seite
    dst = tmp_path / "cloud"
    res = P.pull(dst, storage=fake_storage(store, []))
    assert (dst / "sp500_balance_sheets.duckdb").read_bytes() == b"DUCK" * 10000
    assert (dst / "ml" / "models" / "m.joblib").read_bytes() == b"model"
    assert len(res["changed"]) == r["n_files"]
    # nur geänderte Dateien erneut laden
    (src / "ml" / "models" / "m.joblib").write_bytes(b"model2")
    S.sync(storage=fake_storage(store, []), progress=lambda *_: None)
    res2 = P.pull(dst, storage=fake_storage(store, []))
    assert res2["changed"] == ["ml/models/m.joblib"]
    assert "sb_secret_test" not in json.dumps(json.loads(store["manifest.json"]))


def test_password_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DASHBOARD_PASSWORD", "geheim")
    monkeypatch.delenv("DASHBOARD_CLOUD", raising=False)
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()
    assert at.title[0].value == "Anmelden"
    at.text_input[0].input("falsch")
    at.button[0].click().run()
    assert at.error
    at.text_input[0].input("geheim")
    at.button[0].click().run()
    assert not any(t.value == "Anmelden" for t in at.title)


def test_admin_page_in_cloud_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DASHBOARD_CLOUD", "1")
    (tmp_path / ".cloud_manifest.json").write_text(json.dumps({"synced_at": "2026-11-03T09:00:00+00:00", "files": []}))
    at = AppTest.from_string("from src.pages import data_admin\ndata_admin.page()\n", default_timeout=60)
    at.run()
    assert not at.exception and "Cloud-Ansicht" in at.info[0].value and not at.button


def test_gateway_storage_roundtrip():
    from src.cloud.storage import GatewayStorage
    store = {}

    def handler(req: httpx.Request):
        if req.url.path.endswith("/functions/v1/dashboard-storage"):
            if req.headers.get("x-dashboard-token") != "dsh_test":
                return httpx.Response(401, json={"error": "unauthorized"})
            body = json.loads(req.content)
            p = body["paths"][0]
            if body["action"] == "download" and p not in store:
                return httpx.Response(200, json={"urls": [{"path": p, "signedUrl": None, "error": "not found"}]})
            return httpx.Response(200, json={"urls": [{"path": p, "signedUrl": f"https://x.supabase.co/signed/{p}?t=1"}]})
        p = req.url.path.split("/signed/")[1]
        if req.method == "PUT":
            store[p] = req.content
            return httpx.Response(200, json={})
        return httpx.Response(200, content=store[p])

    g = GatewayStorage("https://x.supabase.co", "dsh_test", transport=httpx.MockTransport(handler))
    g.upload("a/b.parquet", b"123")
    assert g.download("a/b.parquet") == b"123"
    import pytest
    with pytest.raises(FileNotFoundError):
        g.download("fehlt.json")
    bad = GatewayStorage("https://x.supabase.co", "falsch", transport=httpx.MockTransport(handler))
    from src.cloud.storage import CloudError
    with pytest.raises(CloudError):
        bad.upload("x", b"1")


def test_gateway_download_many_and_pull():
    from src.cloud.storage import GatewayStorage
    store = {"manifest.json": json.dumps({"synced_at": "x", "files": [
        {"path": "a.txt", "remote": "a.txt", "sha256": "1", "gzip": False},
        {"path": "ml/b.txt", "remote": "ml/b.txt", "sha256": "2", "gzip": False}]}).encode(),
        "a.txt": b"A", "ml/b.txt": b"B"}
    calls = []

    def handler(req: httpx.Request):
        if req.url.path.endswith("/functions/v1/dashboard-storage"):
            body = json.loads(req.content)
            calls.append(len(body["paths"]))
            return httpx.Response(200, json={"urls": [{"path": p, "signedUrl": f"https://x.supabase.co/signed/{p}"}
                                                      for p in body["paths"]]})
        return httpx.Response(200, content=store[req.url.path.split("/signed/")[1]])

    g = GatewayStorage("https://x.supabase.co", "t", transport=httpx.MockTransport(handler))
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    res = P.pull(d, storage=g)
    assert sorted(res["changed"]) == ["a.txt", "ml/b.txt"] and (d / "ml" / "b.txt").read_bytes() == b"B"
    assert calls == [1, 2]          # 1× Manifest, 1× alle Dateien in einem Aufruf
