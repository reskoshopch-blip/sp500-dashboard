"""Gemeinsame Test-Fixtures: vollständiger Pipeline-Lauf gegen einen simulierten SEC-/Wikipedia-Server
(httpx.MockTransport). Es werden keine echten Netzwerkzugriffe ausgeführt."""
from __future__ import annotations

import json

import httpx
import pytest

from tests.fixtures.builder import company_a, company_b

WIKI_HTML = """<html><body>
<table id="constituents"><thead><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th>
<th>Headquarters Location</th><th>Date added</th><th>CIK</th><th>Founded</th></tr></thead><tbody>
<tr><td>ALPA</td><td>Alpha Test Corp</td><td>Information Technology</td><td>Software</td><td>X</td><td>2010-01-01</td><td>0000001111</td><td>1990</td></tr>
<tr><td>BETA</td><td>Beta Retail Test Inc</td><td>Consumer Discretionary</td><td>Retail</td><td>Y</td><td>2023-06-01</td><td>0000002222</td><td>1980</td></tr>
<tr><td>BETA.B</td><td>Beta Retail Test Inc (Class B)</td><td>Consumer Discretionary</td><td>Retail</td><td>Y</td><td>2023-06-01</td><td>0000002222</td><td>1980</td></tr>
<tr><td>NOFC</td><td>No Facts Test Co</td><td>Utilities</td><td>Electric</td><td>Z</td><td>2000-01-01</td><td>0000003333</td><td>1970</td></tr>
</tbody></table>
<table id="changes"><thead>
<tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th rowspan="2">Reason</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr></thead><tbody>
<tr><td>June 1, 2023</td><td>BETA</td><td>Beta Retail Test Inc</td><td>OLDX</td><td>Old Co</td><td>Test</td></tr>
<tr><td>June 1, 2023</td><td>BETA.B</td><td>Beta Retail Test Inc (Class B)</td><td></td><td></td><td>Share class</td></tr>
</tbody></table></body></html>"""


def sec_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "company_tickers.json" in url:
        return httpx.Response(200, json={"0": {"cik_str": 1111, "ticker": "ALPA", "title": "Alpha"},
                                         "1": {"cik_str": 2222, "ticker": "BETA", "title": "Beta"}})
    if "companyfacts/CIK0000001111" in url:
        return httpx.Response(200, content=json.dumps(company_a()))
    if "companyfacts/CIK0000002222" in url:
        return httpx.Response(200, content=json.dumps(company_b()))
    if "companyconcept/CIK0000001111/us-gaap/Assets" in url:
        return httpx.Response(200, json={"units": {"USD": [
            {"end": "2023-12-31", "val": 1000, "form": "10-K", "filed": "2024-02-14", "accn": "a"}]}})
    return httpx.Response(404)


@pytest.fixture()
def pipeline_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SEC_USER_AGENT", "Pytest Runner pytest@testlauf.ch")
    monkeypatch.setenv("SEC_REQUESTS_PER_SECOND", "10")
    import src.data_sources.sec_client as sc
    import src.data_sources.sp500 as sp
    import src.pipeline as pl

    orig = sc.SecClient

    class MockClient(orig):
        def __init__(self, settings, transport=None):
            super().__init__(settings, transport=httpx.MockTransport(sec_handler))

    monkeypatch.setattr(pl, "SecClient", MockClient)
    monkeypatch.setattr(sp.httpx, "get", lambda *a, **k: httpx.Response(200, text=WIKI_HTML,
                                                                         request=httpx.Request("GET", a[0])))
    meta = pl.run()
    from src.config import get_settings
    return get_settings(), meta
