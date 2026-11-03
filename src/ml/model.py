"""Lernende Modelle, Walk-forward-Backtest und Auswertungen.

Modell: Gradient Boosting (sklearn HistGradientBoostingClassifier). Es verarbeitet fehlende Werte nativ –
'nicht gemeldet' wird also nicht als 0 behandelt, sondern als eigene Information gelernt.

Walk-forward mit Purging: Für das Testjahr Y wird nur auf Beobachtungen trainiert, deren Label-Zeitraum
(t0 bis t0+H) vor dem 1.1.Y abgeschlossen war. Damit kennt das Modell nie Kurse, die im Testjahr liegen.
Jede Prognose im Backtest ist echt 'out of sample'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import partial_dependence
from sklearn.metrics import roc_auc_score

from src.ml.features import BS_FEATURES, MARKET_FEATURES, PRICE_FEATURES

TARGETS = {
    "outperf": {"y": "y_outperf", "label": "Schlägt den S&P 500 in 12 Monaten",
                "short": "Outperformance", "positive": "schlägt Index"},
    "drawdown": {"y": "y_dd30", "label": "Kursrückgang über 30 % innerhalb 12 Monaten",
                 "short": "Absturzrisiko", "positive": "Rückgang > 30 %"},
}
FEATURE_SETS = {
    "bilanz": {"label": "Bilanz + Marktregime",
               "features": list(BS_FEATURES) + list(MARKET_FEATURES) + ["sector"]},
    "bilanz_kurs": {"label": "Bilanz + Marktregime + Kursverlauf",
                    "features": list(BS_FEATURES) + list(MARKET_FEATURES) + list(PRICE_FEATURES) + ["sector"]},
}
ID_COLS = ["cik", "fiscal_year", "primary_ticker", "name", "sector", "t0", "t0_year", "regime",
           "excess_ret", "ret", "max_dd"]


def make_model(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=250, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0,
        categorical_features="from_dtype", random_state=seed)


def prepare_X(df: pd.DataFrame, features: list[str], sectors: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for f in features:
        if f == "sector":
            X[f] = pd.Categorical(df["sector"], categories=sectors)
        else:
            X[f] = pd.to_numeric(df[f], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    return X


def usable_features(train: pd.DataFrame, features: list[str]) -> list[str]:
    """Nur Merkmale mit mindestens zwei unterschiedlichen Werten im Trainingszeitraum.
    Beispiel: Leasingverbindlichkeiten gibt es in XBRL erst ab 2019 – in frühen Trainingsjahren ist die Spalte leer.
    Leere oder konstante Spalten tragen keine Information (und lassen den Binning-Schritt scheitern)."""
    out = []
    for f in features:
        if f == "sector":
            if train["sector"].nunique() >= 2:
                out.append(f)
            continue
        v = pd.to_numeric(train[f], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if v.nunique() >= 2:
            out.append(f)
    return out


# ------------------------------------------------------------------ Walk-forward
def walk_forward(df: pd.DataFrame, features: list[str], ycol: str, sectors: list[str],
                 first_test_year: int, min_train: int = 400, min_test: int = 20):
    lab = df[df[ycol].notna()]
    oos, folds = [], []
    for year in sorted(lab["t0_year"].unique()):
        if year < first_test_year:
            continue
        test = lab[lab["t0_year"] == year]
        train = lab[pd.to_datetime(lab["label_end"]) < pd.Timestamp(int(year), 1, 1)]
        if len(train) < min_train or len(test) < min_test or train[ycol].nunique() < 2:
            continue
        used = usable_features(train, features)
        if not used:
            continue
        model = make_model().fit(prepare_X(train, used, sectors), train[ycol].astype(int))
        prob = model.predict_proba(prepare_X(test, used, sectors))[:, 1]
        oos.append(test[ID_COLS + [ycol]].rename(columns={ycol: "y"}).assign(
            prob=prob, fold=int(year), n_train=len(train),
            train_end=pd.to_datetime(train["label_end"]).max()))
        folds.append((int(year), model, test, used))
    return (pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()), folds


def _auc(y: pd.Series, p: pd.Series) -> float:
    return roc_auc_score(y, p) if y.nunique() == 2 else np.nan


def _ic(p: pd.Series, r: pd.Series) -> float:
    ok = r.notna()
    if ok.sum() < 10:
        return np.nan
    return float(spearmanr(p[ok], r[ok]).statistic)


def add_quintiles(oos: pd.DataFrame) -> pd.DataFrame:
    """Quintil der Prognose innerhalb jedes Testjahres (5 = höchste Wahrscheinlichkeit)."""
    o = oos.copy()
    o["q"] = o.groupby("fold")["prob"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1 if len(s) >= 5 else np.nan)
    return o


def fold_metrics(oos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in oos.groupby("fold"):
        top, bot = g[g["q"] == 5], g[g["q"] == 1]
        rows.append({"fold": year, "n": len(g), "n_train": int(g["n_train"].iloc[0]),
                     "base_rate": g["y"].mean(), "auc": _auc(g["y"], g["prob"]), "ic": _ic(g["prob"], g["excess_ret"]),
                     "top_rate": top["y"].mean(), "bottom_rate": bot["y"].mean(),
                     "top_excess": top["excess_ret"].mean(), "bottom_excess": bot["excess_ret"].mean(),
                     "all_excess": g["excess_ret"].mean(),
                     "top_dd": top["max_dd"].mean(), "bottom_dd": bot["max_dd"].mean()})
    return pd.DataFrame(rows)


def summary_metrics(oos: pd.DataFrame, fm: pd.DataFrame) -> dict:
    if oos.empty:
        return {}
    spread = fm["top_rate"] - fm["bottom_rate"]
    exc = fm["top_excess"] - fm["bottom_excess"]

    def tstat(x: pd.Series) -> float:
        x = x.dropna()
        return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan

    return {"n": int(len(oos)), "n_folds": int(len(fm)), "first_fold": int(fm["fold"].min()),
            "last_fold": int(fm["fold"].max()), "base_rate": float(oos["y"].mean()),
            "auc_pooled": float(_auc(oos["y"], oos["prob"])), "auc_mean": float(fm["auc"].mean()),
            "ic_mean": float(fm["ic"].mean()), "rate_spread_mean": float(spread.mean()),
            "rate_spread_t": tstat(spread), "excess_spread_mean": float(exc.mean()),
            "excess_spread_t": tstat(exc), "share_years_auc_gt_05": float((fm["auc"] > 0.5).mean())}


def regime_metrics(oos: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reg, g in [("Alle", oos)] + [(r, g) for r, g in oos.groupby("regime")]:
        top, bot = g[g["q"] == 5], g[g["q"] == 1]
        rows.append({"regime": reg, "n": len(g), "base_rate": g["y"].mean(), "auc": _auc(g["y"], g["prob"]),
                     "top_rate": top["y"].mean(), "bottom_rate": bot["y"].mean(),
                     "top_excess": top["excess_ret"].mean(), "bottom_excess": bot["excess_ret"].mean()})
    return pd.DataFrame(rows)


def calibration(oos: pd.DataFrame, bins: int = 10) -> tuple[pd.DataFrame, np.ndarray]:
    edges = np.unique(np.quantile(oos["prob"], np.linspace(0, 1, bins + 1)))
    o = oos.assign(bin=np.clip(np.digitize(oos["prob"], edges[1:-1]), 0, len(edges) - 2))
    cal = o.groupby("bin").agg(prob_mean=("prob", "mean"), observed=("y", "mean"), excess=("excess_ret", "mean"),
                               max_dd=("max_dd", "mean"), n=("y", "size"), prob_min=("prob", "min"),
                               prob_max=("prob", "max")).reset_index()
    return cal, edges


def fold_importance(folds, features: list[str], ycol: str, sectors: list[str], repeats: int = 2) -> pd.DataFrame:
    """Permutation Importance auf den Testjahren: AUC-Verlust, wenn ein Merkmal zufällig vertauscht wird.
    Merkmale, die in einem Testjahr nicht im Modell waren, zählen dort als 0."""
    rng = np.random.default_rng(0)
    res = []
    for year, model, test, used in folds:
        y = test[ycol].astype(int).values
        if len(np.unique(y)) < 2:
            continue
        for f in set(features) - set(used):
            res.append({"feature": f, "imp": 0.0, "fold": year})
        features_fold = used
        X = prepare_X(test, used, sectors)
        base = roc_auc_score(y, model.predict_proba(X)[:, 1])
        for f in features_fold:
            drops = []
            for _ in range(repeats):
                Xp = X.copy()
                Xp[f] = X[f].values[rng.permutation(len(X))]
                if f == "sector":
                    Xp[f] = pd.Categorical(Xp[f], categories=sectors)
                drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
            res.append({"feature": f, "imp": float(np.mean(drops)), "fold": year})
    if not res:
        return pd.DataFrame(columns=["feature", "importance", "std", "n_folds"])
    r = pd.DataFrame(res)
    return r.groupby("feature")["imp"].agg(importance="mean", std="std", n_folds="count") \
        .reset_index().sort_values("importance", ascending=False)


# ------------------------------------------------------------------ Endmodell, Prognosen, Erklärungen
def fit_final(df: pd.DataFrame, features: list[str], ycol: str, sectors: list[str]):
    lab = df[df[ycol].notna()]
    used = usable_features(lab, features)
    X = prepare_X(lab, used, sectors)
    model = make_model().fit(X, lab[ycol].astype(int))
    medians = X.drop(columns=["sector"], errors="ignore").median()
    return model, X, medians, used


def live_rows(df: pd.DataFrame, last_price_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Neueste Bilanz je Unternehmen, deren 12-Monats-Ausgang noch offen ist."""
    open_ = df[(pd.to_datetime(df["t0"]) > last_price_date - pd.Timedelta(days=horizon))]
    return open_.sort_values("t0").groupby("cik").tail(1)


def drivers(model, X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """Beitrag je Merkmal: Wahrscheinlichkeit minus Wahrscheinlichkeit mit Merkmal auf Median gesetzt."""
    base = model.predict_proba(X)[:, 1]
    rows = []
    for f in medians.index:
        X2 = X.copy()
        X2[f] = medians[f]
        delta = base - model.predict_proba(X2)[:, 1]
        rows.append(pd.DataFrame({"row": X.index, "feature": f, "value": X[f].values, "delta": delta}))
    return pd.concat(rows, ignore_index=True)


def pdp_table(model, X: pd.DataFrame, regimes: pd.Series, features: list[str], grid: int = 20) -> pd.DataFrame:
    rows = []
    for f in features:
        if X[f].notna().sum() < 50:
            continue
        for reg in ["Alle", "Bull", "Bear"]:
            sub = X if reg == "Alle" else X[regimes.values == reg]
            if len(sub) < 100:
                continue
            try:
                pdp = partial_dependence(model, sub, [f], kind="average", grid_resolution=grid,
                                         percentiles=(0.05, 0.95))
            except ValueError:
                continue
            rows.append(pd.DataFrame({"feature": f, "regime": reg, "x": pdp["grid_values"][0],
                                      "prob": pdp["average"][0]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["feature", "regime", "x", "prob"])


def empirical_buckets(df: pd.DataFrame, features: list[str], dd_col: str = "y_dd30") -> pd.DataFrame:
    """Modellfrei: Wie entwickelten sich Aktien je Quintil eines Merkmals (Quintil je Jahr gebildet)?"""
    lab = df[df["excess_ret"].notna()]
    rows = []
    for f in features:
        v = pd.to_numeric(lab[f], errors="coerce")
        q = v.groupby(lab["t0_year"]).transform(
            lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1 if s.notna().sum() >= 25 else np.nan)
        q = q.where(v.notna(), 0)  # 0 = nicht gemeldet
        tmp = lab.assign(q=q, v=v)
        for reg, g in [("Alle", tmp)] + [(r, g) for r, g in tmp.groupby("regime") if r in ("Bull", "Bear")]:
            agg = g.groupby("q").agg(n=("excess_ret", "size"), excess_mean=("excess_ret", "mean"),
                                     excess_median=("excess_ret", "median"), outperf_rate=("y_outperf", "mean"),
                                     dd_rate=(dd_col, "mean"), value_median=("v", "median")).reset_index()
            agg["feature"], agg["regime"] = f, reg
            rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
