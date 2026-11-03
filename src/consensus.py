"""Modell-Konsens S&P 500: Wie oft lag man richtig, wenn alle (bzw. k von n) Modelle bullisch waren?

Modelle (alle 0–100, hoch = bullisch, je Tag nur mit damals bekannten Daten berechnet):
* Markt-Terminal kurz / mittel / lang (Gesundheitsindex)
* Fundamental gesamt (gelernter Score) und Fundamental-ML (Wahrscheinlichkeit Plus in 3 Monaten × 100)
* COT-Score E-mini S&P 500 (wöchentlich, gilt ab dem Montag nach Veröffentlichung)
* Top-Warnung: 100 − 25 × Anzahl Ampel-Warnzeichen (100 = keine Warnzeichen)
GEX hat noch keine Historie und fliesst nur als aktuelle Information ein.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import get_settings

MODELS = {"H_kurz": "Markt-Terminal kurzfristig", "H_mittel": "Markt-Terminal mittelfristig",
          "H_lang": "Markt-Terminal langfristig", "F_gesamt": "Fundamental gesamt",
          "ML3M": "Fundamental-ML (Chance 3 M.)", "COT": "COT-Score S&P 500", "TOP": "Top-Warnung (keine Ampel)",
          "CHAMP": "Champagner-Index"}
DEFAULT = ["H_kurz", "H_mittel", "H_lang", "F_gesamt", "ML3M", "COT"]


def load_frame() -> pd.DataFrame:
    """Tägliche Tabelle mit Kurs (SPY) und allen verfügbaren Modell-Scores."""
    s = get_settings()
    d = s.data_dir
    p = d / "ml" / "health_SPY_daily.parquet"
    if not p.exists():
        return pd.DataFrame()
    h = pd.read_parquet(p)
    h["date"] = pd.to_datetime(h["date"])
    out = h.set_index("date")[["close", "H_kurz", "H_mittel", "H_lang"]]
    f = d / "fundamental" / "fund_SPY_daily.parquet"
    if f.exists():
        fd = pd.read_parquet(f)
        fd["date"] = pd.to_datetime(fd["date"])
        fd = fd.set_index("date")
        out = out.join(fd[["F_gesamt"]])
        if "p_3M" in fd:
            out["ML3M"] = fd["p_3M"].reindex(out.index) * 100
    c = d / "cot" / "SPX_weekly.parquet"
    if c.exists():
        cw = pd.read_parquet(c)
        cs = cw.set_index(pd.to_datetime(cw["trade_date"]))["score"].dropna().sort_index()
        out["COT"] = cs.reindex(out.index, method="ffill", tolerance=pd.Timedelta(days=10))
    t = d / "ml" / "top_SPY_daily.parquet"
    if t.exists():
        td = pd.read_parquet(t, columns=["date", "ampel_score"])
        td["date"] = pd.to_datetime(td["date"])
        out["TOP"] = (100 - 25 * td.set_index("date")["ampel_score"]).clip(0, 100).reindex(out.index)
    g = d / "geo" / "champagne_daily.parquet"
    if g.exists():
        gd = pd.read_parquet(g, columns=["date", "score_learned"])
        gd["date"] = pd.to_datetime(gd["date"])
        out["CHAMP"] = gd.set_index("date")["score_learned"].reindex(out.index, method="ffill",
                                                                      tolerance=pd.Timedelta(days=5))
    return out[out["close"].notna()]


def forward(close: pd.Series, h: int) -> pd.Series:
    return close.shift(-h) / close - 1


def _hit(r: pd.Series) -> dict:
    r = r.dropna()
    return {"Tage": int(len(r)), "Trefferquote": float((r > 0).mean()) if len(r) else np.nan,
            "Ø Rendite": float(r.mean()) if len(r) else np.nan}


def by_count(d: pd.DataFrame, models: list[str], th: float, h: int) -> pd.DataFrame:
    x = d.dropna(subset=models)
    r = forward(d["close"], h).reindex(x.index)
    k = (x[models] >= th).sum(axis=1)
    rows = [{"Bullische Modelle": f"{i} von {len(models)}", "k": i, **_hit(r[k == i])} for i in range(len(models) + 1)]
    rows.append({"Bullische Modelle": "Alle Tage (Basis)", "k": -1, **_hit(r)})
    return pd.DataFrame(rows)


def per_model(d: pd.DataFrame, models: list[str], th: float, h: int) -> pd.DataFrame:
    x = d.dropna(subset=models)
    r = forward(d["close"], h).reindex(x.index)
    rows = []
    for m in models:
        rows.append({"Modell": MODELS[m], "Bedingung": f"≥ {th:.0f}", **_hit(r[x[m] >= th])})
        rows.append({"Modell": MODELS[m], "Bedingung": f"≤ {100 - th:.0f}", **_hit(r[x[m] <= 100 - th])})
    return pd.DataFrame(rows)


def trades(d: pd.DataFrame, cond: pd.Series, h: int, side: int = 1, cost: float = 0.0005) -> pd.DataFrame:
    """Nicht überlappende Trades: Einstieg am ersten Tag, an dem die Bedingung gilt, Ausstieg nach h Tagen."""
    c = d["close"].values
    idx = d.index
    cv = cond.reindex(idx).fillna(False).values
    rows, i = [], 0
    while i < len(c) - h:
        if cv[i]:
            ret = (c[i + h] / c[i] - 1) * side - 2 * cost
            rows.append({"Einstieg": idx[i], "Ausstieg": idx[i + h], "Rendite": ret})
            i += h
        else:
            i += 1
    return pd.DataFrame(rows)


def trade_stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"n": 0}
    r = t["Rendite"]
    w, l = r[r > 0], r[r <= 0]
    return {"n": len(r), "winrate": float((r > 0).mean()), "avg": float(r.mean()),
            "pf": float(w.sum() / -l.sum()) if len(l) and l.sum() < 0 else np.inf, "worst": float(r.min())}


def learned_combo(d: pd.DataFrame, models: list[str], h: int, first_year: int = 2008) -> dict:
    """Gelernte Kombination (logistische Regression, jährlich neu, nur abgeschlossene Zeiträume)."""
    x = d.dropna(subset=models).copy()
    if len(x) < 1500:
        return {}
    y = (forward(d["close"], h).reindex(x.index) > 0).astype(float).where(forward(d["close"], h).reindex(x.index).notna())
    r = forward(d["close"], h).reindex(x.index)
    p = pd.Series(np.nan, index=x.index)
    for yr in range(max(first_year, x.index.min().year + 4), x.index.max().year + 1):
        cut = pd.Timestamp(f"{yr}-01-01")
        pos = x.index.searchsorted(cut)
        tr = x.iloc[:max(pos - h - 1, 0)].iloc[::5]
        ytr = y.reindex(tr.index).dropna()
        if len(ytr) < 200 or ytr.nunique() < 2:
            continue
        m = LogisticRegression(C=0.1, max_iter=500).fit(tr.loc[ytr.index, models] / 100, ytr)
        te = (x.index >= cut) & (x.index < pd.Timestamp(f"{yr + 1}-01-01"))
        p[te] = m.predict_proba(x.loc[te, models] / 100)[:, 1]
    e = pd.DataFrame({"p": p, "y": y, "r": r}).dropna()
    if len(e) < 200 or e["y"].nunique() < 2:
        return {}
    rows = []
    for t in (0.5, 0.6, 0.65, 0.7):
        g = e[e["p"] >= t]
        rows.append({"KI-Wahrscheinlichkeit": f"≥ {t:.0%}", **_hit(g["r"])})
    rows.append({"KI-Wahrscheinlichkeit": "Alle Tage (Basis)", **_hit(e["r"])})
    return {"auc": float(roc_auc_score(e["y"], e["p"])), "table": pd.DataFrame(rows), "p_now": None,
            "start": str(e.index.min().date())}
