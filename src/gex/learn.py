"""GEX-KI: lernt aus den gesammelten Schnappschüssen, wie sich Gamma-Lage und Levels auf den nächsten Tag auswirken.

Ziele (je Schnappschuss, mit dem nächsten Börsenschluss nach dem Abruf):
* Bewegungsgrösse: |Rendite bis zum nächsten Schluss| im Verhältnis zur normalen Tagesschwankung (20 Tage)
* Richtung: nächster Schluss höher als der Spot beim Abruf?
* Range: schliesst der nächste Tag zwischen Put-Wall und Call-Wall?

Zwei Stufen
1. Regelmodell (sofort): Positive Gamma → Bewegung ~0.8 × normal, negative Gamma → ~1.25 × normal; die Walls als
   Grenzen der erwarteten Range. Das ist Marktlogik, nicht gelernt – und wird genauso mitgetestet.
2. Gelerntes Modell (ab MIN_SAMPLES abgeschlossenen Tagen): regularisierte Regressionen über alle Symbole, jeden Tag
   neu trainiert nur mit Tagen, deren Ergebnis schon feststand (walk-forward). Die Treffsicherheit wird nur aus diesen
   echten Vorhersagen berechnet und gegen das Regelmodell gestellt.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import get_settings
from src.gex.collect import gex_dir, load_snapshots
from src.ml.markets import LONG_DIR, LONG_START
from src.ml.prices import fetch_prices, load_prices
from src.utils import atomic_write_text

PRICE_TICKER = {"SPX": "^GSPC", "XSP": "^GSPC", "NDX": "^NDX", "RUT": "^RUT", "VIX": "^VIX"}
MIN_SAMPLES = 30
FEATS = ["regime_pos", "dist_flip", "net_ratio", "concentration", "wall_pos", "wall_width", "dist_call_wall",
         "dist_put_wall", "dist_max_gamma", "zeta_risk", "zeta_drift", "rv20", "mom5"]
FEAT_LABEL = {"regime_pos": "Positive Gamma (über Flip)", "dist_flip": "Abstand zum Gamma-Flip",
              "net_ratio": "Netto-Gamma (−1 … +1)", "concentration": "Gamma-Konzentration ±2 %",
              "wall_pos": "Lage zwischen Put- und Call-Wall", "wall_width": "Abstand Put-Wall–Call-Wall",
              "dist_call_wall": "Abstand zur Call-Wall", "dist_put_wall": "Abstand zur Put-Wall",
              "dist_max_gamma": "Abstand zu Max Gamma", "zeta_risk": "Zeta-Risiko", "zeta_drift": "Zeta-Drift",
              "rv20": "Normale Tagesschwankung (20 T.)", "mom5": "Kursveränderung 5 Tage"}
CLOSE_UTC_HOUR = 21     # US-Schluss 16:00 New York ≈ 20–21 Uhr UTC (konservativ 21)


def ticker(sym: str) -> str:
    return PRICE_TICKER.get(sym, sym)


def build_dataset(snaps: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Je Symbol und Tag der letzte Schnappschuss + Ziele aus den Schlusskursen danach."""
    if snaps.empty:
        return pd.DataFrame()
    s = snaps.copy()
    s["taken_at"] = pd.to_datetime(s["taken_at"], utc=True)
    s["day"] = s["taken_at"].dt.tz_convert("America/New_York").dt.date
    s = s.sort_values("taken_at").groupby(["symbol", "day"]).tail(1)
    rows = []
    for r in s.itertuples(index=False):
        r = r._asdict()
        px = prices.get(ticker(r["symbol"]))
        if px is not None and len(px):
            close_ts = pd.DatetimeIndex(px.index).tz_localize("UTC") + pd.Timedelta(hours=CLOSE_UTC_HOUR)
            before = px[close_ts <= r["taken_at"]]
            after = px[close_ts > r["taken_at"]]
            rets = before.pct_change().dropna()
            r["rv20"] = float(rets.tail(20).std()) if len(rets) >= 15 else np.nan
            r["mom5"] = float(before.iloc[-1] / before.iloc[-6] - 1) if len(before) >= 6 else np.nan
            if len(after) >= 1 and np.isfinite(r["spot"]):
                c1 = float(after.iloc[0])
                r["next_date"] = after.index[0]
                r["ret1"] = c1 / r["spot"] - 1
                r["inside"] = float(r.get("put_wall", np.nan) <= c1 <= r.get("call_wall", np.nan)) \
                    if np.isfinite(r.get("put_wall", np.nan)) and np.isfinite(r.get("call_wall", np.nan)) else np.nan
                r["toward_maxg"] = float(np.sign(c1 - r["spot"]) == np.sign(r.get("max_gamma", np.nan) - r["spot"])) \
                    if np.isfinite(r.get("max_gamma", np.nan)) and r.get("max_gamma") != r["spot"] else np.nan
            if len(after) >= 5 and np.isfinite(r["spot"]):
                r["ret5"] = float(after.iloc[4]) / r["spot"] - 1
        rows.append(r)
    d = pd.DataFrame(rows)
    for col in ("next_date", "ret1", "ret5", "inside", "toward_maxg", "rv20", "mom5"):   # erster Tag: noch keine Ergebnisse
        if col not in d:
            d[col] = np.nan
    d["regime_pos"] = (d["regime"] == "positiv").astype(float)
    if "ret1" in d:
        d["move_ratio"] = d["ret1"].abs() / d["rv20"]
        d["up"] = (d["ret1"] > 0).astype(float).where(d["ret1"].notna())
    for c in FEATS:
        if c not in d:
            d[c] = np.nan
    return d.sort_values("taken_at").reset_index(drop=True)


def rule_predict(row) -> dict:
    rv = row.get("rv20")
    mult = 0.8 if row.get("regime_pos") == 1 else 1.25
    return {"move_ratio": mult, "exp_move": rv * mult if rv and np.isfinite(rv) else np.nan}


def _X(d: pd.DataFrame) -> pd.DataFrame:
    X = d[FEATS].astype(float).copy()
    return X


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    out = {}
    X = _X(train)
    med = X.median()
    Xf, Xt = X.fillna(med).fillna(0), _X(test).fillna(med).fillna(0)
    keep = [c for c in FEATS if Xf[c].std() > 0]
    if not keep:
        return out
    t = train.dropna(subset=["move_ratio"])
    if len(t) >= MIN_SAMPLES:
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xf.loc[t.index, keep], np.log(t["move_ratio"] + 0.05))
        out["move_ratio"] = np.exp(m.predict(Xt[keep])) - 0.05
        out["move_model"] = (m, keep)
    for target in ("up", "inside"):
        t = train.dropna(subset=[target])
        if len(t) >= MIN_SAMPLES and t[target].nunique() == 2:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500)).fit(Xf.loc[t.index, keep],
                                                                                             t[target].astype(int))
            out[target] = m.predict_proba(Xt[keep])[:, 1]
            out[f"{target}_model"] = (m, keep)
    return out


def walk_forward(d: pd.DataFrame) -> pd.DataFrame:
    """Jeder Schnappschuss wird mit einem Modell vorhergesagt, das nur Tage kannte, deren nächster Schluss
    vor dem Abruf feststand."""
    preds = []
    for i, row in d.iterrows():
        known = d[(d["next_date"].notna()) & (pd.to_datetime(d["next_date"]).dt.tz_localize("UTC")
                                              + pd.Timedelta(hours=CLOSE_UTC_HOUR) <= row["taken_at"])]
        p = {"idx": i, "rule_move": rule_predict(row)["move_ratio"]}
        if len(known) >= MIN_SAMPLES:
            fp = _fit_predict(known, d.loc[[i]])
            for k in ("move_ratio", "up", "inside"):
                if k in fp:
                    p[f"ml_{k}"] = float(fp[k][0])
        preds.append(p)
    return pd.DataFrame(preds).set_index("idx")


def evaluate(d: pd.DataFrame) -> dict:
    ev = {}
    m = d.dropna(subset=["move_ratio"])
    if len(m) >= 10:
        ev["rule_move_corr"] = float(m["rule_move"].corr(m["move_ratio"], method="spearman"))
        mm = m.dropna(subset=["ml_move_ratio"]) if "ml_move_ratio" in m else m.iloc[0:0]
        if len(mm) >= 10:
            ev["ml_move_corr"] = float(mm["ml_move_ratio"].corr(mm["move_ratio"], method="spearman"))
            ev["ml_move_corr_rule_same"] = float(mm["rule_move"].corr(mm["move_ratio"], method="spearman"))
            ev["n_ml"] = int(len(mm))
    for t in ("up", "inside"):
        col = f"ml_{t}"
        if col in d:
            x = d.dropna(subset=[col, t])
            if len(x) >= 10 and x[t].nunique() == 2:
                ev[f"ml_{t}_auc"] = float(roc_auc_score(x[t], x[col]))
                ev[f"ml_{t}_hit"] = float(((x[col] >= 0.5) == (x[t] == 1)).mean())
                ev[f"{t}_base"] = float(x[t].mean())
    return ev


def regime_stats(d: pd.DataFrame) -> list[dict]:
    rows = []
    x = d.dropna(subset=["ret1"]) if "ret1" in d else pd.DataFrame()
    for reg, g in x.groupby("regime"):
        rows.append({"Regime": f"{reg}e Gamma", "Tage": len(g), "Bewegung vs. normal": float(g["move_ratio"].median()),
                     "Ø |Bewegung|": float(g["ret1"].abs().mean()), "Anteil Plus-Tage": float((g["ret1"] > 0).mean()),
                     "Schluss zwischen den Walls": float(g["inside"].mean()) if g["inside"].notna().any() else None,
                     "Richtung Max Gamma": float(g["toward_maxg"].mean()) if g["toward_maxg"].notna().any() else None})
    return rows


def current(d: pd.DataFrame) -> dict:
    """Prognose für den jeweils neuesten Schnappschuss je Symbol (Modell mit allen abgeschlossenen Tagen)."""
    out = {}
    known = d.dropna(subset=["move_ratio"]) if "move_ratio" in d else d.iloc[0:0]
    for sym, g in d.groupby("symbol"):
        row = g.iloc[-1]
        r = {"taken_at": str(row["taken_at"]), "spot": row["spot"], "regime": row.get("regime"),
             "flip": row.get("flip"), "call_wall": row.get("call_wall"), "put_wall": row.get("put_wall"),
             "max_gamma": row.get("max_gamma"), "rv20": row.get("rv20")}
        rule = rule_predict(row)
        r["rule_exp_move"] = rule["exp_move"]
        if len(known) >= MIN_SAMPLES:
            fp = _fit_predict(known, g.iloc[[-1]])
            if "move_ratio" in fp and np.isfinite(row.get("rv20", np.nan)):
                r["ml_exp_move"] = float(fp["move_ratio"][0] * row["rv20"])
            for t in ("up", "inside"):
                if t in fp:
                    r[f"ml_{t}"] = float(fp[t][0])
            if "move_model" in fp:
                m, keep = fp["move_model"]
                coef = m[-1].coef_
                r["move_drivers"] = sorted([{"key": k, "coef": float(c)} for k, c in zip(keep, coef)],
                                           key=lambda z: -abs(z["coef"]))[:8]
        out[sym] = r
    return out


def run(fetch: bool = True) -> dict:
    snaps = load_snapshots()
    s = get_settings()
    tickers = sorted({ticker(x) for x in snaps["symbol"].unique()}) if not snaps.empty else []
    if fetch and tickers:
        try:
            fetch_prices(s, tickers, start=LONG_START, subdir=LONG_DIR, require_benchmark=False, max_age_hours=6)
        except Exception:  # noqa: BLE001 - ohne Kursupdate mit vorhandenen Kursen weiter
            pass
    prices = load_prices(s, tickers, subdir=LONG_DIR, include_benchmark=False)
    d = build_dataset(snaps, prices)
    res = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_snapshots": int(len(snaps)),
           "n_days": int(len(d)), "n_done": int(d["ret1"].notna().sum()) if "ret1" in d else 0,
           "min_samples": MIN_SAMPLES}
    if not d.empty:
        wf = walk_forward(d)
        d = d.join(wf)
        res["eval"] = evaluate(d)
        res["regimes"] = regime_stats(d)
        res["current"] = current(d)
        d.drop(columns=[c for c in d.columns if d[c].dtype == object and c not in ("symbol", "regime", "zeta_regime",
                                                                                  "zeta_computed")],
               errors="ignore").to_parquet(gex_dir() / "dataset.parquet", index=False)
    atomic_write_text(gex_dir() / "model.json", json.dumps(res, ensure_ascii=False, indent=1, default=_json))
    return res


def _json(o):
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return str(o)


if __name__ == "__main__":
    print(json.dumps(run(), default=_json, indent=1)[:2000])
