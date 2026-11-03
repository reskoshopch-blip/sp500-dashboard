"""Rendert jede Dashboard-Seite gegen die Test-Datenbank und prüft, dass keine Ausnahme auftritt."""
import pytest
from streamlit.testing.v1 import AppTest

PAGES = ["overview", "compare", "sectors", "profile", "table", "quality", "method", "data_admin"]


@pytest.mark.parametrize("name", PAGES)
def test_page_renders(pipeline_db, name):
    at = AppTest.from_string(f"from src.pages import {name}\n{name}.page()\n", default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_profile_quarter_mode_and_table_filters(pipeline_db):
    at = AppTest.from_string("from src.pages import profile\nprofile.page()\n", default_timeout=60)
    at.run()
    at.radio(key="prof_mode").set_value("Neuester Bilanzstichtag (10-Q/10-K)").run()
    assert not at.exception
    at = AppTest.from_string("from src.pages import table\ntable.page()\n", default_timeout=60)
    at.run()
    at.multiselect(key="tb_avail").set_value(["fehlt"]).run()
    assert not at.exception
    at.radio(key="tb_view").set_value("Kennzahlen").run()
    assert not at.exception


def test_app_entrypoint(pipeline_db):
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()
    assert not at.exception


@pytest.fixture(scope="session")
def ml_dir(tmp_path_factory):
    from src.config import get_settings
    from src.ml.run import train_and_evaluate
    from tests.fixtures.ml_synth import spy_series, synthetic_dataset
    d = tmp_path_factory.mktemp("mldata")
    mp = pytest.MonkeyPatch()
    mp.setenv("DATA_DIR", str(d))
    train_and_evaluate(get_settings(), synthetic_dataset(n_comp=80), spy_series())
    mp.undo()
    return d


ML_PAGES = ["ml_forecast", "ml_backtest", "ml_effects", "ml_regime"]


@pytest.mark.parametrize("name", ML_PAGES)
def test_ml_pages_without_results(tmp_path, monkeypatch, name):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    at = AppTest.from_string(f"from src.pages import {name}\n{name}.page()\n", default_timeout=60)
    at.run()
    assert not at.exception and at.info


@pytest.mark.parametrize("name", ML_PAGES)
def test_ml_pages_render(ml_dir, monkeypatch, name):
    monkeypatch.setenv("DATA_DIR", str(ml_dir))
    monkeypatch.setenv("SEC_USER_AGENT", "Pytest Runner pytest@testlauf.ch")
    at = AppTest.from_string(f"from src.pages import {name}\n{name}.page()\n", default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_ml_interactions(ml_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(ml_dir))
    at = AppTest.from_string("from src.pages import ml_forecast\nml_forecast.page()\n", default_timeout=120)
    at.run()
    at.selectbox(key="fc_t").set_value("drawdown").run()
    assert not at.exception
    ni = at.number_input
    assert len(ni) > 0
    ni[0].set_value(ni[0].value + 1.0).run()
    assert not at.exception, [e.value for e in at.exception]
    at = AppTest.from_string("from src.pages import ml_backtest\nml_backtest.page()\n", default_timeout=120)
    at.run()
    at.selectbox(key="bt_f").set_value("bilanz_kurs").run()
    assert not at.exception
    at = AppTest.from_string("from src.pages import ml_effects\nml_effects.page()\n", default_timeout=120)
    at.run()
    at.selectbox(key="fx_feat").set_value("current_ratio").run()
    at.radio(key="fx_t").set_value("drawdown").run()
    assert not at.exception, [e.value for e in at.exception]
