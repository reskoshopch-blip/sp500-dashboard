"""Lernender Fundamental-Score, ML-Prognose und „Was ist eingepreist?“ – alles walk-forward.

1. Score je Indikator: Perzentil gegenüber den letzten 10 Jahren (nur Vergangenheit, mind. 3 Jahre) → 0–100.
2. Lernen (jedes Jahr am 1. Januar neu): Für jeden Indikator wird gemessen, wie sein Score mit der Rendite der
   folgenden 6 Monate zusammenhing (Rangkorrelation, nur Zeiträume, die am Lerntag schon abgeschlossen waren).
   Daraus folgen Richtung (hoch = gut oder schlecht) und Gewicht (statistische Stärke, t-Wert, gedeckelt).
   Schwache Indikatoren (|t| < 1) behalten die ökonomische Grundrichtung mit kleinem Gewicht oder fallen weg.
3. Kategorie- und Gesamtscore = gewichteter Durchschnitt der ausgerichteten Scores (50 = neutral).
4. ML-Prognose: Gradient Boosting schätzt die Wahrscheinlichkeit einer positiven Rendite in 3 und 12 Monaten,
   jährlich neu trainiert nur mit abgeschlossenen Zeiträumen (Purging), getestet Jahr für Jahr.
5. Eingepreist: Kursmomentum (Perzentil) minus Fundamental-Score. Stark positiv = der Kurs ist den
   Fundamentaldaten vorausgelaufen (viel eingepreist), stark negativ = Fundamentaldaten besser als der Kurs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.fundamental.catalog import CATEGORIES, IND_BY_KEY

WINDOW, MIN_HIST = 2520, 756
H_LEARN = 126
CAT_COL = {c: "C_" + c.split(" ")[0].replace("&", "").lower() for c in CATEGORIES}
LEVELS = [(0, 20, "sehr bärisch"), (20, 40, "bärisch"), (40, 60, "neutral"), (60, 80, "bullisch"),
          (80, 101, "sehr bullisch")]
DIV_BUCKETS = [(-200, -25, "Fundamental besser als Kurs (nicht eingepreist)"),
               (-25, 25, "Kurs und Fundamentaldaten im Einklang"),
               (25, 200, "Kurs vorausgelaufen (viel eingepreist)")]


def level(v) -> str:
    if v is None or pd.isna(v):
        return "keine Daten"
    for lo, hi, n in LEVELS:
        if lo <= v < hi:
            return n
    return LEVELS[-1][2]


def percentile_scores(raw: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in raw.columns:
        x = raw[c]
        if x.notna().sum() < MIN_HIST:
            continue
        out[c] = x.rolling(WINDOW, min_periods=MIN_HIST).rank(pct=True) * 100
    return pd.DataFrame(out, index=raw.index)


def forward(close: pd.Series, h: int) -> tuple[pd.Series, pd.Series]:
    c = close.values
    n = len(c)
    fwd = np.full(n, np.nan)
    mdd = np.full(n, np.nan)
    for i in range(n - h):
        fwd[i] = c[i + h] / c[i] - 1
        mdd[i] = c[i + 1:i + 1 + h].min() / c[i] - 1
    return pd.Series(fwd, index=close.index), pd.Series(mdd, index=close.index)


def learn_weights(scores: pd.DataFrame, fwd: pd.Series, years: list[int], h: int = H_LEARN) -> pd.DataFrame:
    """Richtung und Gewicht je Indikator und Lernjahr (nur abgeschlossene Zeiträume vor dem 1. Januar)."""
    idx = scores.index
    rows = []
    weekly = np.arange(0, len(idx), 5)
    for y in years:
        cut = idx.searchsorted(pd.Timestamp(f"{y}-01-01"))
        last = cut - h - 1
        sel = weekly[weekly <= last]
        for c in scores.columns:
            prior = IND_BY_KEY[c].prior if c in IND_BY_KEY else 0
            s = scores[c].iloc[sel]
            r = fwd.iloc[sel]
            ok = s.notna() & r.notna()
            n = int(ok.sum())
            ic, t = np.nan, 0.0
            if n >= 150:
                ic = float(s[ok].corr(r[ok], method="spearman"))
                n_eff = n / (h / 5)
                t = ic * np.sqrt(max(n_eff, 1)) if pd.notna(ic) else 0.0
            if abs(t) >= 1:
                sign, w, src = int(np.sign(t)), float(min(abs(t), 3.0)), "gelernt"
            elif prior != 0:
                sign, w, src = prior, 0.5, "Grundrichtung"
            else:
                sign, w, src = 1, 0.0, "kein Signal"
            rows.append({"year": y, "key": c, "n": n, "ic": ic, "t": t, "sign": sign, "weight": w, "source": src,
                         "prior": prior})
    return pd.DataFrame(rows)


def apply_weights(scores: pd.DataFrame, learned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ausgerichtete Scores, Gewichte je Tag, Kategorie- und Gesamtscores."""
    idx = scores.index
    years = sorted(learned["year"].unique())
    W = pd.DataFrame(np.nan, index=idx, columns=scores.columns)
    S = pd.DataFrame(np.nan, index=idx, columns=scores.columns)
    for k, y in enumerate(years):
        start = pd.Timestamp(f"{y}-01-01")
        end = pd.Timestamp(f"{years[k + 1]}-01-01") if k + 1 < len(years) else idx.max() + pd.Timedelta(days=1)
        m = (idx >= start) & (idx < end)
        ly = learned[learned["year"] == y].set_index("key")
        for c in scores.columns:
            if c not in ly.index:
                continue
            W.loc[m, c] = ly.loc[c, "weight"]
            S.loc[m, c] = ly.loc[c, "sign"]
    oriented = scores.where(S >= 0, 100 - scores)
    oriented = oriented.where(S.notna())
    comp = pd.DataFrame(index=idx)
    Wv = W.where(oriented.notna(), 0).fillna(0)
    for cat, col in CAT_COL.items():
        keys = [c for c in scores.columns if c in IND_BY_KEY and IND_BY_KEY[c].category == cat]
        if not keys:
            continue
        w = Wv[keys]
        num = (oriented[keys].fillna(0) * w).sum(axis=1)
        den = w.sum(axis=1)
        comp[col] = (num / den).where(den > 0)
    num = (oriented.fillna(0) * Wv).sum(axis=1)
    den = Wv.sum(axis=1)
    comp["F_gesamt"] = (num / den).where(den > 0)
    return oriented, Wv, comp


def bucket_test(score: pd.Series, close: pd.Series, start: str, horizons=(63, 126, 252),
                buckets=None, label="Bereich") -> pd.DataFrame:
    buckets = buckets or [(lo, hi, f"{lo}–{min(hi, 100)} {n}") for lo, hi, n in LEVELS]
    rows = []
    for h in horizons:
        fwd, mdd = forward(close, h)
        df = pd.DataFrame({"s": score, "f": fwd, "d": mdd})
        df = df[df.index >= start].dropna()
        if df.empty:
            continue
        base_f, base_p, base_d = df["f"].mean(), (df["f"] > 0).mean(), (df["d"] <= -0.10).mean()
        for lo, hi, name in buckets:
            g = df[(df["s"] >= lo) & (df["s"] < hi)]
            rows.append({"h": h, label: name, "lo": lo, "Tage": len(g),
                         "Ø Rendite": g["f"].mean() if len(g) else np.nan,
                         "Anteil positiv": (g["f"] > 0).mean() if len(g) else np.nan,
                         "Rückgang ≥ 10 %": (g["d"] <= -0.10).mean() if len(g) else np.nan,
                         "Basis Ø Rendite": base_f, "Basis positiv": base_p, "Basis ≥10%": base_d,
                         "IC": float(df["s"].corr(df["f"], method="spearman"))})
    return pd.DataFrame(rows)


def effects(scores: pd.DataFrame, close: pd.Series, start: str) -> pd.DataFrame:
    """Modellfrei: Rendite der nächsten 6/12 Monate je Quintil jedes Indikators (ab start)."""
    rows = []
    f126, d126 = forward(close, 126)
    f252, _ = forward(close, 252)
    for c in scores.columns:
        df = pd.DataFrame({"s": scores[c], "f6": f126, "d6": d126, "f12": f252})
        df = df[df.index >= start].dropna(subset=["s", "f6"])
        if len(df) < 500:
            continue
        df["q"] = pd.cut(df["s"], [-0.1, 20, 40, 60, 80, 100.1], labels=["0–20", "20–40", "40–60", "60–80", "80–100"])
        for q, g in df.groupby("q", observed=True):
            rows.append({"key": c, "Quintil": str(q), "Tage": len(g), "Ø 6 M.": g["f6"].mean(),
                         "Ø 12 M.": g["f12"].mean(), "Anteil positiv 6 M.": (g["f6"] > 0).mean(),
                         "Rückgang ≥ 10 % (6 M.)": (g["d6"] <= -0.10).mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- ML-Prognose
def _model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=0)


def ml_forecast(X: pd.DataFrame, close: pd.Series, h: int, first_test_year: int) -> dict:
    """Walk-forward: jährlich neu trainieren (nur abgeschlossene Zeiträume), Prognose für jeden Tag des Jahres.
    Auswertung auf Wochenstichproben."""
    fwd, _ = forward(close, h)
    y = (fwd > 0).astype(float).where(fwd.notna())
    idx = X.index
    weekly = np.arange(0, len(idx), 5)
    prob = pd.Series(np.nan, index=idx)
    years = sorted({d.year for d in idx if d.year >= first_test_year})
    per_year = []
    used_cols = None
    for yr in years:
        cut = idx.searchsorted(pd.Timestamp(f"{yr}-01-01"))
        tr = weekly[weekly <= cut - h - 1]
        tr = tr[y.iloc[tr].notna().values]
        if len(tr) < 260:
            continue
        cols = [c for c in X.columns if X.iloc[tr][c].notna().mean() > 0.3]
        if not cols:
            continue
        ytr = y.iloc[tr]
        if ytr.nunique() < 2:
            continue
        m = _model().fit(X.iloc[tr][cols], ytr)
        te = (idx >= pd.Timestamp(f"{yr}-01-01")) & (idx < pd.Timestamp(f"{yr + 1}-01-01"))
        prob[te] = m.predict_proba(X.loc[te, cols])[:, 1]
        used_cols = cols
    ev = pd.DataFrame({"p": prob, "y": y, "f": fwd}).iloc[weekly].dropna()
    res = {"h": h, "n": int(len(ev))}
    if len(ev) >= 50 and ev["y"].nunique() == 2:
        res["auc"] = float(roc_auc_score(ev["y"], ev["p"]))
        res["base"] = float(ev["y"].mean())
        res["hit"] = float(((ev["p"] >= 0.5) == (ev["y"] == 1)).mean())
        res["brier"] = float(((ev["p"] - ev["y"]) ** 2).mean())
        for yr, g in ev.groupby(ev.index.year):
            auc = float(roc_auc_score(g["y"], g["p"])) if g["y"].nunique() == 2 and len(g) >= 20 else np.nan
            per_year.append({"Jahr": int(yr), "AUC": auc, "Ø Prognose": float(g["p"].mean()),
                             "Anteil positiv": float(g["y"].mean()), "Ø Rendite": float(g["f"].mean()), "n": len(g)})
        bins = [0, 0.4, 0.5, 0.6, 0.7, 1.01]
        ev["b"] = pd.cut(ev["p"], bins, right=False)
        res["calibration"] = [{"Prognose": f"{b.left:.0%}–{min(b.right, 1):.0%}", "n": len(g),
                               "Ø Prognose": float(g["p"].mean()), "Trefferquote": float(g["y"].mean()),
                               "Ø Rendite": float(g["f"].mean())}
                              for b, g in ev.groupby("b", observed=True) if len(g)]
    res["per_year"] = per_year
    # finales Modell mit allem Abgeschlossenen -> heutige Prognose und Treiber
    tr = weekly[weekly <= len(idx) - h - 1]
    tr = tr[y.iloc[tr].notna().values]
    cols = [c for c in X.columns if X.iloc[tr][c].notna().mean() > 0.3] if len(tr) else []
    if len(tr) >= 260 and cols and y.iloc[tr].nunique() == 2:
        m = _model().fit(X.iloc[tr][cols], y.iloc[tr])
        x_now = X[cols].iloc[[-1]]
        p_now = float(m.predict_proba(x_now)[:, 1][0])
        res["p_raw"] = p_now
        res["p_now"] = calibrate(ev, p_now)
        med = X.iloc[tr][cols].median()
        drv = []
        for c in cols:
            if pd.isna(x_now[c].iloc[0]):
                continue
            x2 = x_now.copy()
            x2[c] = med[c]
            drv.append({"key": c, "delta": p_now - float(m.predict_proba(x2)[:, 1][0])})
        res["drivers"] = sorted(drv, key=lambda d: -abs(d["delta"]))[:12]
        res["used_cols"] = cols
    return {"res": res, "prob": prob}


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def calibrate(ev: pd.DataFrame, p: float) -> float:
    """Heutige Wahrscheinlichkeit an den echten Testergebnissen kalibrieren (Platt-Skalierung auf den
    Out-of-sample-Prognosen). Ist das Modell überzuversichtlich, wird die Zahl Richtung Basisrate gezogen;
    ohne Vorhersagekraft landet sie bei der Basisrate."""
    if ev is None or len(ev) < 150 or ev["y"].nunique() < 2:
        return p
    lr = LogisticRegression(C=1.0).fit(_logit(ev["p"]).reshape(-1, 1), ev["y"].astype(int))
    if lr.coef_[0][0] <= 0:      # keine positive Beziehung -> Basisrate
        return float(ev["y"].mean())
    return float(lr.predict_proba(_logit([p]).reshape(-1, 1))[0, 1])


def momentum_pct(close: pd.Series) -> pd.Series:
    mom = close / close.shift(126) - 1
    return mom.rolling(1260, min_periods=504).rank(pct=True) * 100


def run_market(raw: pd.DataFrame, close: pd.Series, first_test_year: int) -> dict:
    """Komplette Auswertung eines Marktes. raw: Rohwerte der Indikatoren auf den Handelstagen von close."""
    close = close.reindex(raw.index).ffill()
    scores = percentile_scores(raw)
    fwd, _ = forward(close, H_LEARN)
    first_learn = max(int(raw.index.min().year) + 6, 1995)
    years = list(range(first_learn, int(raw.index.max().year) + 1))
    learned = learn_weights(scores, fwd, years)
    oriented, weights, comp = apply_weights(scores, learned)
    daily = pd.concat([close.rename("close"), raw.add_prefix("raw_"), scores.add_prefix("s_"),
                       oriented.add_prefix("o_"), weights.add_prefix("w_"), comp], axis=1)
    daily["mom_pct"] = momentum_pct(close)
    daily["divergence"] = daily["mom_pct"] - daily["F_gesamt"]
    start = f"{max(first_test_year, first_learn)}-01-01"
    bt = bucket_test(daily["F_gesamt"], close, start)
    bt_div = bucket_test(daily["divergence"], close, start, buckets=DIV_BUCKETS, label="Lage")
    eff = effects(scores, close, start)
    # ML: Scores + Kategorien + Veränderung der Kategorien über 3 Monate
    X = pd.concat([scores.add_prefix("s_"), comp], axis=1)
    for c in comp.columns:
        X[f"{c}_chg"] = comp[c] - comp[c].shift(63)
    X["mom_pct"] = daily["mom_pct"]
    ml = {}
    for h, name in ((63, "3M"), (252, "12M")):
        out = ml_forecast(X, close, h, max(first_test_year, first_learn))
        daily[f"p_{name}"] = out["prob"]
        ml[name] = out["res"]
    return {"daily": daily.reset_index().rename(columns={"index": "date"}), "learned": learned, "bt": bt,
            "bt_div": bt_div, "effects": eff, "ml": ml, "start": start}
