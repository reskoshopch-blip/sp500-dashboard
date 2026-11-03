"""Pizza-Index: tolerantes Lesen der PizzINT-Daten, HTML-Ausweichweg, Verlauf und Seite."""
from __future__ import annotations

import httpx
from streamlit.testing.v1 import AppTest

from src.geo import pizza as Z

JS = {"defcon_level": 2, "last_updated": "2026-11-03T02:00:00Z", "data": {"pizzerias": [
    {"name": "Extreme Pizza", "distance": "0.5 mi", "current_busyness": 90, "normal_busyness": 40, "is_open": True},
    {"name": "Domino's", "percentage_above_normal": "20%", "is_open": False}]}}


def client(json_ok=True):
    def h(req):
        if req.url.path.startswith("/api/") and json_ok:
            return httpx.Response(200, json=JS)
        if req.url.path.startswith("/api/"):
            return httpx.Response(404, text="nope")
        return httpx.Response(200, text="<div>DOUGHCON <span>4</span></div>", headers={"content-type": "text/html"})
    return httpx.Client(transport=httpx.MockTransport(h))


def test_parse_json_and_history(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    r = Z.collect(client())
    assert r["level"] == 2 and r["n_places"] == 2 and r["n_spikes"] == 1 and r["n_open"] == 1
    assert round(r["max_pct_above"]) == 125
    Z.collect(client())
    assert len(Z.load_history()) == 2


def test_html_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    r = Z.collect(client(json_ok=False))
    assert r["level"] == 4 and r["source"] == "html" and r["n_places"] == 0


def test_page(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DASHBOARD_CLOUD", raising=False)
    at = AppTest.from_string("from src.pages import pizza\npizza.page()\n", default_timeout=60)
    at.run()
    assert not at.exception and at.info
    Z.collect(client())
    Z.collect(client())
    at = AppTest.from_string("from src.pages import pizza\npizza.page()\n", default_timeout=60)
    at.run()
    assert not at.exception and at.error and at.metric[0].value == "2"
