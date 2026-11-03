"""ML-Lauf: Kurse laden -> Datensatz -> Walk-forward-Backtest -> Endmodelle -> Prognosen -> Wirkungsanalyse.

Aufruf:
  python -m src.ml.run                 # Kurse aktualisieren und alles trainieren
  python -m src.ml.run --skip-prices   # nur neu trainieren (vorhandene Kurse)
  python -m src.ml.run --force-prices  # Kurse komplett neu laden
Voraussetzung: Die Bilanzdatenbank existiert (python -m src.pipeline).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from src.config import ConfigError, Settings, get_settings
from src.utils import atomic_write_text
from src.database.store import connect_readonly
from src.ml import model as M
from src.ml.features import BS_FEATURES, FEATURE_LABELS, build_dataset, market_frame
from src.ml.health import compute_health
from src.ml.market_top import EXTRA_TICKERS, run_top, run_top_market
from src.ml.markets import MARKETS, fetch_market_data, market_inputs
from src.ml.prices import BENCHMARK, fetch_prices, load_prices

log = logging.getLogger("ml")
FIRST_TEST_YEAR = 2014   # davor zu wenige abgeschlossene Trainingsjahre


def status(settings: Settings, **kw) -> None:
    settings.ml_dir.mkdir(parents=True, exist_ok=True)
    kw["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    atomic_write_text(settings.ml_dir / "status.json", json.dumps(kw, ensure_ascii=False))


def load_db(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not settings.db_path.exists():
        raise RuntimeError("Keine Bilanzdatenbank gefunden – zuerst 'python -m src.pipeline' ausführen.")
    con = connect_readonly(settings.db_path)
    try:
        items = con.execute("SELECT cik, fiscal_year, period_end, item_key, value, original_value, unit, tag, accn, "
                            "form, filed, restated, is_derived FROM balance_items").df()
        periods = con.execute("SELECT cik, fiscal_year, period_end, filed FROM periods").df()
        comps = con.execute("SELECT cik, primary_ticker, name, sector FROM companies WHERE fetch_status = 'ok'").df()
    finally:
        con.close()
    for d in (items, periods):
        d["cik"] = d["cik"].astype(int)
        d["fiscal_year"] = d["fiscal_year"].astype(int)
    comps["cik"] = comps["cik"].astype(int)
    return items, periods, comps


def train_and_evaluate(settings: Settings, ds: pd.DataFrame, spy: pd.Series,
                       first_test_year: int = FIRST_TEST_YEAR) -> dict:
    out = settings.ml_dir
    (out / "models").mkdir(parents=True, exist_ok=True)
    sectors = sorted(ds["sector"].dropna().unique().tolist())
    last_price = spy.index.max()
    ds.to_parquet(out / "dataset.parquet", index=False)

    mf = market_frame(spy).reset_index().rename(columns={"index": "date"})
    mf["regime"] = np.where(mf["vs_sma200"].isna(), "unbekannt", np.where(mf["vs_sma200"] > 0, "Bull", "Bear"))
    mf.to_parquet(out / "regime.parquet", index=False)

    tables = {k: [] for k in ("oos", "folds", "regimes", "calibration", "importance", "pdp", "live", "drivers")}
    summaries = {}
    combos = [(t, f) for t in M.TARGETS for f in M.FEATURE_SETS]
    for i, (tkey, fkey) in enumerate(combos, 1):
        ycol, feats = M.TARGETS[tkey]["y"], M.FEATURE_SETS[fkey]["features"]
        status(settings, state="läuft", step=f"Training {M.TARGETS[tkey]['short']} / {M.FEATURE_SETS[fkey]['label']}",
               done=i - 1, total=len(combos))
        tag = {"target": tkey, "fset": fkey}
        oos, folds = M.walk_forward(ds, feats, ycol, sectors, first_test_year)
        if oos.empty:
            summaries[f"{tkey}|{fkey}"] = {"error": "zu wenige Daten für einen Walk-forward-Backtest"}
            continue
        oos = M.add_quintiles(oos)
        fm = M.fold_metrics(oos)
        summaries[f"{tkey}|{fkey}"] = M.summary_metrics(oos, fm)
        cal, edges = M.calibration(oos)
        imp = M.fold_importance(folds, feats, ycol, sectors)
        for name, df in (("oos", oos), ("folds", fm), ("regimes", M.regime_metrics(oos)), ("calibration", cal),
                         ("importance", imp)):
            tables[name].append(df.assign(**tag))

        model, X, medians, used = M.fit_final(ds, feats, ycol, sectors)
        joblib.dump({"model": model, "features": used, "sectors": sectors, "medians": medians,
                     "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "n_train": int(len(X))}, out / "models" / f"{tkey}_{fkey}.joblib")
        lab = ds[ds[ycol].notna()]
        top_num = [f for f in imp["feature"] if f != "sector" and f in used][:12]
        tables["pdp"].append(M.pdp_table(model, X, lab["regime"], top_num).assign(**tag))

        live = M.live_rows(ds, last_price, settings.ml_horizon_days)
        if len(live):
            XL = M.prepare_X(live, used, sectors)
            p = model.predict_proba(XL)[:, 1]
            b = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
            calm = cal.set_index("bin")
            lv = live[["cik", "fiscal_year", "primary_ticker", "name", "sector", "t0", "regime", "filed"]].assign(
                prob=p, pct_rank=pd.Series(p).rank(pct=True).values,
                hist_rate=calm["observed"].reindex(b).values, hist_excess=calm["excess"].reindex(b).values,
                hist_dd=calm["max_dd"].reindex(b).values, hist_n=calm["n"].reindex(b).values, **tag)
            tables["live"].append(lv)
            dr = M.drivers(model, XL.reset_index(drop=True), medians)
            dr["cik"] = live["cik"].values[dr["row"]]
            dr = dr.drop(columns="row")
            dr = dr[dr["delta"].abs() > 1e-6]
            tables["drivers"].append(dr.assign(**tag))

    status(settings, state="läuft", step="Wirkungsanalyse (modellfrei)", done=len(combos), total=len(combos))
    buckets = M.empirical_buckets(ds, list(BS_FEATURES))
    buckets.to_parquet(out / "buckets.parquet", index=False)
    for name, parts in tables.items():
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        df.to_parquet(out / f"{name}.parquet", index=False)
    meta = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_samples": int(len(ds)), "n_labeled": int(ds["excess_ret"].notna().sum()),
            "n_companies": int(ds["cik"].nunique()), "t0_min": str(pd.to_datetime(ds["t0"]).min().date()),
            "t0_max": str(pd.to_datetime(ds["t0"]).max().date()), "last_price": str(last_price.date()),
            "horizon_days": settings.ml_horizon_days, "first_test_year": first_test_year,
            "summaries": summaries, "feature_labels": FEATURE_LABELS}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return meta


def _write_top(settings: Settings, key: str, res: dict, extra_meta: dict) -> dict:
    out = settings.ml_dir
    out.mkdir(parents=True, exist_ok=True)
    res["daily"].rename_axis("date").reset_index().to_parquet(out / f"top_{key}_daily.parquet", index=False)
    for name in ("episodes", "coef", "red"):
        df = res[name].copy()
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype("string")
        df.to_parquet(out / f"top_{key}_{name}.parquet", index=False)
    hd, hbt, hcomps = compute_health(res["daily"])
    hd.rename_axis("date").reset_index().to_parquet(out / f"health_{key}_daily.parquet", index=False)
    hbt.to_parquet(out / f"health_{key}_bt.parquet", index=False)
    extra_meta = dict(extra_meta, health_components=hcomps)
    meta = dict(res["meta"], **extra_meta, key=key, built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                last_date=str(res["daily"]["close"].dropna().index.max().date()))
    (out / f"top_{key}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str),
                                              encoding="utf-8")
    return meta


def save_top(settings: Settings, prices: dict, spy: pd.Series, key: str = "SPY") -> dict:
    """S&P 500 aus einem Kurs-Dict (für Tests und als Rückfall ohne lange Historie)."""
    res = run_top(prices, spy)
    return _write_top(settings, key, res, {"label": MARKETS[key]["label"], "short": MARKETS[key]["short"]})


def save_top_markets(settings: Settings, sp500_tickers: list[str], market_info: dict | None = None) -> dict:
    metas = {}
    for key, m in MARKETS.items():
        status(settings, state="läuft", step=f"Top-Warnung {m['short']}", done=0, total=0)
        inp = market_inputs(settings, key, sp500_tickers)
        if inp is None:
            metas[key] = {"error": "keine Kursdaten"}
            continue
        try:
            res = run_top_market(inp["main"], inp["stocks"], inp["vol"], inp["hyg"], inp["ief"], inp["fx"],
                                 inp["first_test_year"], inp.get("extras"))
        except Exception as exc:  # noqa: BLE001 – ein Markt darf die anderen nicht verhindern
            log.exception("Top-Warnung %s fehlgeschlagen", key)
            metas[key] = {"error": str(exc)}
            continue
        extra = {"label": m["label"], "short": m["short"], "main_ticker": m["main"],
                 "nikkei_source": (market_info or {}).get("nikkei_source") if m["stocks"] == "nikkei" else None}
        metas[key] = _write_top(settings, key, res, extra)
    (settings.ml_dir / "top_markets.json").write_text(json.dumps(
        {k: {"label": MARKETS[k]["label"], "ok": "error" not in v, "error": v.get("error")} for k, v in metas.items()},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return metas


def run(skip_prices: bool = False, force_prices: bool = False, downloader=None, top_only: bool = False) -> dict:
    settings = get_settings()
    try:
        status(settings, state="läuft", step="Bilanzdaten lesen", done=0, total=0)
        items, periods, comps = load_db(settings)
        tickers = comps["primary_ticker"].dropna().tolist()
        price_info = {}
        if not skip_prices:
            status(settings, state="läuft", step="Kurse laden (Yahoo Finance)", done=0, total=len(tickers) + 1)
            price_info = fetch_prices(settings, tickers + EXTRA_TICKERS, force=force_prices, downloader=downloader,
                                      progress=lambda d, t: status(settings, state="läuft", step="Kurse laden",
                                                                   done=d, total=t))
        market_info = {}
        if not skip_prices:
            market_info = fetch_market_data(settings, downloader=downloader, force=force_prices,
                                            progress=lambda m: status(settings, state="läuft", step=f"Kurse: {m}",
                                                                      done=0, total=0))
        prices = load_prices(settings, tickers + EXTRA_TICKERS)
        if BENCHMARK not in prices:
            raise RuntimeError("Keine SPY-Kurse vorhanden – zuerst ohne --skip-prices ausführen.")
        top_metas = save_top_markets(settings, tickers, market_info)
        if top_only:
            ok = [k for k, v in top_metas.items() if "error" not in v]
            status(settings, state="fertig", step=f"fertig (Top-Warnung: {', '.join(ok) or 'keine'})", done=1,
                   total=1, n_samples=len(ok), prices_failed=len(price_info.get("failed", [])))
            return {"top": top_metas, "prices_failed": price_info.get("failed", [])}
        status(settings, state="läuft", step="Datensatz aufbauen", done=0, total=0)
        ds = build_dataset(items, periods, comps, prices, prices[BENCHMARK], settings.ml_years,
                           settings.ml_horizon_days)
        if ds.empty:
            raise RuntimeError("Datensatz ist leer – passen Kurse und Bilanzen zusammen?")
        meta = train_and_evaluate(settings, ds, prices[BENCHMARK])
        meta["prices_failed"] = price_info.get("failed", [])
        (settings.ml_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str),
                                                   encoding="utf-8")
        status(settings, state="fertig", step="fertig", done=1, total=1, n_samples=meta["n_samples"],
               prices_failed=len(meta["prices_failed"]))
        return meta
    except Exception as exc:
        status(settings, state="Fehler", step=str(exc), done=0, total=0)
        raise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ML-Modelle trainieren und historisch testen")
    ap.add_argument("--skip-prices", action="store_true")
    ap.add_argument("--force-prices", action="store_true")
    ap.add_argument("--top-only", action="store_true", help="Nur die Top-Warnung für den Gesamtmarkt berechnen")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        meta = run(a.skip_prices, a.force_prices, top_only=a.top_only)
    except ConfigError as exc:
        print(f"KONFIGURATIONSFEHLER: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 3
    if "top" in meta and "n_samples" not in meta:
        for k, t in meta["top"].items():
            if "error" in t:
                print(f"{k}: FEHLER {t['error']}")
            else:
                print(f"{k}: AUC {t['auc_oos']:.3f}, Ampel erkannt {t['ampel']['detected']}/{t['ampel']['episodes']}")
        return 0
    print(f"Fertig: {meta['n_samples']} Firmenjahre ({meta['n_labeled']} mit abgeschlossenem 12-Monats-Ergebnis).")
    for k, s in meta["summaries"].items():
        if "auc_mean" in s:
            print(f"  {k}: AUC {s['auc_mean']:.3f} über {s['n_folds']} Testjahre")
    if meta.get("prices_failed"):
        print(f"  Keine Kurse für {len(meta['prices_failed'])} Ticker: {', '.join(meta['prices_failed'][:20])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
