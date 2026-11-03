import json

import numpy as np
import pandas as pd
import pytest

from src.ml import model as M
from src.ml.features import availability_dates, build_dataset, pit_wide, price_block
from src.ml.prices import fetch_prices, load_prices
from src.normalization.facts import normalize_company
from tests.fixtures.builder import company_a
from tests.fixtures.ml_synth import SECTORS, spy_series, synthetic_dataset

YEARS = tuple(range(2020, 2026))


def lin_prices(start="2019-01-01", end="2026-10-20", p0=100.0, daily=0.0005):
    idx = pd.bdate_range(start, end)
    return pd.Series(p0 * np.exp(daily * np.arange(len(idx))), index=idx)


def test_price_block_point_in_time_and_labels():
    s = lin_prices()
    filed = pd.Timestamp("2024-02-14")
    pb = price_block(s, filed, 365)
    assert pb["t0"] > filed                                     # erst nach dem Filing handelbar
    assert pb["ret"] == pytest.approx(np.exp(0.0005 * 261) - 1, rel=0.02)
    assert pb["max_dd"] > 0                                     # steigender Kurs: kein Drawdown unter t0
    assert pb["mom_12m"] > 0
    late = price_block(s, pd.Timestamp("2026-03-01"), 365)
    assert pd.isna(late["ret"]) and pd.notna(late["t0"])        # Ergebnis noch offen -> Live-Prognose


def test_dataset_uses_first_reported_values_not_restatements():
    res = normalize_company(company_a(), YEARS)
    items = res.annual_items
    wide = pit_wide(items)
    a22 = wide[wide["fiscal_year"] == 2022]["total_assets"].iloc[0]
    assert a22 == 880                                           # nicht der spätere Wert 900
    comps = pd.DataFrame({"cik": [1111], "primary_ticker": ["ALPA"], "name": ["Alpha"], "sector": ["IT"]})
    spy = lin_prices(daily=0.0003)
    ds = build_dataset(items, res.periods, comps, {"ALPA": lin_prices(), "SPY": spy}, spy, YEARS)
    r22 = ds[ds["fiscal_year"] == 2022].iloc[0]
    assert r22["equity_ratio"] == pytest.approx(400 / 880)
    assert r22["t0"] > pd.Timestamp("2023-02-15")               # Filing-Datum des 10-K FY2022
    assert r22["excess_ret"] > 0 and r22["y_outperf"] == 1
    r21 = ds[ds["fiscal_year"] == 2021].iloc[0]                 # nur Vergleichsspalte im 10-K 2022
    assert r21["t0"] > pd.Timestamp("2023-02-15")
    av = availability_dates(res.periods)
    assert len(av) == av[["cik", "fiscal_year"]].drop_duplicates().shape[0]


def test_walk_forward_is_purged():
    ds = synthetic_dataset(n_comp=120)
    oos, folds = M.walk_forward(ds, M.FEATURE_SETS["bilanz"]["features"], "y_outperf", SECTORS, 2014)
    assert len(folds) >= 8
    for year, _, _, _ in folds:
        o = oos[oos["fold"] == year]
        assert (o["train_end"] < pd.Timestamp(year, 1, 1)).all()
        assert (pd.to_datetime(o["t0"]).dt.year == year).all()


def test_model_finds_planted_signal_and_not_noise():
    feats = M.FEATURE_SETS["bilanz"]["features"]
    sig = synthetic_dataset(signal=True)
    oos, _ = M.walk_forward(sig, feats, "y_outperf", SECTORS, 2014)
    assert M._auc(oos["y"], oos["prob"]) > 0.6
    noise = synthetic_dataset(signal=False, seed=5)
    oos2, _ = M.walk_forward(noise, feats, "y_outperf", SECTORS, 2014)
    assert abs(M._auc(oos2["y"], oos2["prob"]) - 0.5) < 0.05


def test_metrics_quintiles_and_buckets():
    ds = synthetic_dataset(n_comp=120)
    feats = M.FEATURE_SETS["bilanz"]["features"]
    oos, folds = M.walk_forward(ds, feats, "y_outperf", SECTORS, 2014)
    oos = M.add_quintiles(oos)
    assert set(oos["q"].dropna().unique()) == {1, 2, 3, 4, 5}
    fm = M.fold_metrics(oos)
    s = M.summary_metrics(oos, fm)
    assert s["n_folds"] == len(fm) and s["excess_spread_mean"] > 0
    cal, edges = M.calibration(oos)
    assert cal["n"].sum() == len(oos)
    b = M.empirical_buckets(ds, ["g_equity", "current_ratio"])
    ge = b[(b["feature"] == "g_equity") & (b["regime"] == "Alle")].set_index("q")
    assert ge.loc[5, "excess_mean"] > ge.loc[1, "excess_mean"]
    cr = b[(b["feature"] == "current_ratio") & (b["regime"] == "Alle")]
    assert 0 in set(cr["q"])                                    # 'nicht gemeldet' als eigene Gruppe


def test_train_and_evaluate_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from src.config import get_settings
    from src.ml.run import train_and_evaluate
    meta = train_and_evaluate(get_settings(), synthetic_dataset(n_comp=100), spy_series())
    out = tmp_path / "ml"
    for name in ("oos", "folds", "regimes", "calibration", "importance", "pdp", "live", "drivers", "buckets",
                 "regime", "dataset"):
        assert (out / f"{name}.parquet").exists(), name
    live = pd.read_parquet(out / "live.parquet")
    assert len(live) and live["prob"].between(0, 1).all()
    assert (pd.to_datetime(live["t0"]) > pd.Timestamp("2025-10-20")).all()
    assert json.loads((out / "meta.json").read_text(encoding="utf-8"))["summaries"]
    imp = pd.read_parquet(out / "importance.parquet")
    top = imp[(imp["target"] == "outperf") & (imp["fset"] == "bilanz")]["feature"].head(3).tolist()
    assert {"g_equity", "debt_to_assets"} & set(top)


def test_fetch_prices_cache_and_benchmark_required(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from src.config import get_settings
    st = get_settings()
    calls = []

    def dl(tickers, start):
        calls.append(list(tickers))
        return {t: lin_prices() for t in tickers if t != "BAD"}

    info = fetch_prices(st, ["AAA", "BAD"], downloader=dl)
    assert set(info["ok"]) == {"AAA", "SPY"} and info["failed"] == ["BAD"]
    info2 = fetch_prices(st, ["AAA"], downloader=dl)
    assert set(info2["skipped"]) == {"AAA", "SPY"}               # Cache frisch -> kein neuer Abruf
    assert set(load_prices(st, ["AAA"])) == {"AAA", "SPY"}
    with pytest.raises(RuntimeError):
        fetch_prices(st, ["X"], force=True, downloader=lambda t, s: {})


def test_empty_and_constant_features_in_early_years():
    """Reale Situation: Leasing-Tags erst ab 2019, Regime in frühen Jahren konstant."""
    ds = synthetic_dataset(n_comp=120)
    ds.loc[ds["fiscal_year"] < 2019, "lease_to_assets"] = np.nan
    ds.loc[ds["t0_year"] < 2015, "mkt_bull"] = 1.0
    ds["equity_negative"] = 0.0
    feats = M.FEATURE_SETS["bilanz"]["features"]
    oos, folds = M.walk_forward(ds, feats, "y_outperf", SECTORS, 2014)
    assert len(folds) >= 8
    first_used = folds[0][3]
    assert "lease_to_assets" not in first_used and "equity_negative" not in first_used
    assert "lease_to_assets" in folds[-1][3]
    imp = M.fold_importance(folds, feats, "y_outperf", SECTORS)
    assert set(imp["feature"]) == set(feats) - {"equity_negative"} or "equity_negative" in set(imp["feature"])
    model, X, med, used = M.fit_final(ds, feats, "y_outperf", SECTORS)
    assert "equity_negative" not in used and "lease_to_assets" in used
