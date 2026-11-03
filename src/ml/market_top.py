"""Top-Warnung für den Gesamtmarkt (S&P 500 / SPY).

Ziel: Phasen erkennen, in denen ein Rückgang von mindestens 10 % innerhalb der nächsten ~3 Monate
(63 Handelstage) wahrscheinlicher ist als üblich – also VOR dem Fall, nicht danach wie die 200-Tage-Regel.

Zwei Bausteine:
  1. Ampel (regelbasiert, transparent): vier klassische Warnzeichen
     - Breiten-Divergenz: Index nahe Hoch, aber immer weniger Aktien über ihrem 200-Tage-Schnitt
     - VIX steigt aus tiefem Niveau
     - Kreditmarkt schwächelt (Hochzinsanleihen HYG vs. Staatsanleihen IEF) während Aktien nahe Hoch
     - Überdehnung (SPY weit über dem 200-Tage-Schnitt)
  2. ML-Top-Risiko: logistische Regression auf ~15 Marktmerkmalen, Walk-forward getestet.
     Bewusst ein einfaches Modell: Seit 2010 gab es nur rund ein Dutzend Rückgänge ≥ 10 % – ein komplexes
     Modell würde diese wenigen Fälle auswendig lernen.

Kein Blick in die Zukunft: Merkmale nur aus Kursen bis zum jeweiligen Tag; für ein Testjahr wird nur auf Tage
trainiert, deren 63-Tage-Ergebnis vor Beginn des Testjahres feststand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXTRA_TICKERS = ["^VIX", "HYG", "IEF"]
HORIZON = 63            # Handelstage (~3 Monate)
DD_THRESHOLD = -0.10    # Rückgang, der als "Einbruch" zählt
FIRST_TEST_YEAR = 2014

TOP_FEATURES = {
    "breadth_200": "Anteil Aktien über 200-Tage-Schnitt",
    "breadth_50": "Anteil Aktien über 50-Tage-Schnitt",
    "d_breadth_200_63": "Δ Marktbreite (200T) über 3 Monate",
    "breadth_divergence": "Breiten-Divergenz (Index nahe Hoch, Breite fällt)",
    "hi_lo": "Neue 52W-Hochs minus -Tiefs (Anteil)",
    "spy_vs_sma200": "Index Abstand zu 200-Tage-Schnitt",
    "spy_vs_sma50": "Index Abstand zu 50-Tage-Schnitt",
    "spy_mom_63": "Index Rendite 3 Monate",
    "spy_mom_252": "Index Rendite 12 Monate",
    "spy_dist_high": "Index Abstand zum 52W-Hoch",
    "spy_vol_21": "Index Volatilität 1 Monat",
    "d_spy_vol": "Veränderung Volatilität (1M vs. 6M)",
    "vix": "Volatilitätsindex (VIX / Nikkei VI)",
    "vix_vs_min63": "Volatilität über 3-Monats-Tief",
    "credit_vs_sma50": "Kredit (USA): HYG/IEF vs. 50-Tage-Schnitt",
    "credit_mom_63": "Kredit (USA): HYG/IEF 3-Monats-Veränderung",
    "fx_mom_20": "USD/JPY Veränderung 1 Monat",
    "fx_vs_sma50": "USD/JPY Abstand zu 50-Tage-Schnitt",
}
AMPEL = {
    "a_breadth": "Breiten-Divergenz",
    "a_vix": "Volatilität steigt aus Tief",
    "a_credit": "Kreditmarkt schwächelt",
    "a_stretch": "Überdehnung",
    "a_fx": "Yen wertet stark auf",
    "a_ew": "Schwergewichte tragen den Index",
    "a_div": "Tages-Divergenzen (Index fällt, Mehrheit steigt)",
    "a_below20": "Viele Aktien ≥ 20 % unter Hoch",
}


# ------------------------------------------------------------------ Merkmale
BREADTH_COLS = ["breadth_200", "breadth_50", "breadth_20", "hi_lo", "pct_below20", "adv_share", "ad_line",
                "mcclellan", "n_stocks"]


def breadth_panel(prices: dict[str, pd.Series], index: pd.DatetimeIndex, exclude: set[str]) -> pd.DataFrame:
    """Marktbreite aus allen Einzelaktien (nur heutige Mitglieder -> leichte Verzerrung in frühen Jahren).
    breadth_X   Anteil über X-Tage-Schnitt          pct_below20  Anteil ≥ 20 % unter 52-Wochen-Hoch
    adv_share   Anteil Aktien mit Tagesplus          ad_line      kumulierte Advance-Decline-Linie (in Aktien)
    mcclellan   McClellan-Oszillator (EMA19 − EMA39 der verhältnisbereinigten Netto-Advances)"""
    cols = {t: s.reindex(index).ffill(limit=5) for t, s in prices.items() if t not in exclude}
    if not cols:
        return pd.DataFrame(index=index, columns=BREADTH_COLS, dtype=float)
    panel = pd.DataFrame(cols)
    sma200 = panel.rolling(200, min_periods=200).mean()
    sma50 = panel.rolling(50, min_periods=50).mean()
    sma20 = panel.rolling(20, min_periods=20).mean()
    hi = panel.rolling(252, min_periods=200).max()
    lo = panel.rolling(252, min_periods=200).min()
    out = pd.DataFrame(index=index)

    def share(cond: pd.DataFrame, valid: pd.DataFrame) -> pd.Series:
        n = valid.sum(axis=1)
        return (cond & valid).sum(axis=1) / n.where(n >= 30)

    v200, v50, v20, vhl = (sma200.notna() & panel.notna(), sma50.notna() & panel.notna(),
                           sma20.notna() & panel.notna(), hi.notna() & panel.notna())
    out["breadth_200"] = share(panel > sma200, v200)
    out["breadth_50"] = share(panel > sma50, v50)
    out["breadth_20"] = share(panel > sma20, v20)
    nhl = vhl.sum(axis=1).where(vhl.sum(axis=1) >= 30)
    out["hi_lo"] = (((panel >= hi * 0.999) & vhl).sum(axis=1) - ((panel <= lo * 1.001) & vhl).sum(axis=1)) / nhl
    out["pct_below20"] = share(panel <= hi * 0.80, vhl)
    ret = panel.pct_change(fill_method=None)
    vr = ret.notna()
    nr = vr.sum(axis=1)
    adv, dec = (ret > 0).sum(axis=1), (ret < 0).sum(axis=1)
    out["adv_share"] = (adv / nr.where(nr >= 30))
    net = ((adv - dec) / (adv + dec).where(nr >= 30))
    out["ad_line"] = (adv - dec).where(nr >= 30).fillna(0).cumsum().where(nr >= 30)
    ratio = net * 1000
    out["mcclellan"] = ratio.ewm(span=19, adjust=False).mean() - ratio.ewm(span=39, adjust=False).mean()
    out["mcclellan"] = out["mcclellan"].where(nr >= 30)
    out["n_stocks"] = v200.sum(axis=1)
    return out


def market_features(main: pd.Series, stocks: dict[str, pd.Series] | None = None, vol: pd.Series | None = None,
                    hyg: pd.Series | None = None, ief: pd.Series | None = None,
                    fx: pd.Series | None = None, extras: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Merkmale für einen Index. Fehlende Quellen (z. B. kein Vol-Index) bleiben NaN bzw. nutzen Ersatzgrössen."""
    if isinstance(stocks, dict) and vol is None and hyg is None and "^VIX" in stocks:   # alte Aufrufform (prices-Dict)
        prices = stocks
        stocks = {k: v for k, v in prices.items() if k not in {"SPY", *EXTRA_TICKERS}}
        vol, hyg, ief = prices.get("^VIX"), prices.get("HYG"), prices.get("IEF")
    s = main.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    idx = s.index
    f = pd.DataFrame(index=idx)
    f["close"] = s
    sma200, sma50 = s.rolling(200, min_periods=200).mean(), s.rolling(50, min_periods=50).mean()
    f["sma200"] = sma200
    f["sma50"] = sma50
    f["spy_vs_sma200"] = s / sma200 - 1
    f["spy_vs_sma50"] = s / sma50 - 1
    f["spy_mom_63"] = s / s.shift(63) - 1
    f["spy_mom_252"] = s / s.shift(252) - 1
    f["spy_dist_high"] = s / s.rolling(252, min_periods=200).max() - 1
    r = np.log(s).diff()
    f["spy_vol_21"] = r.rolling(21).std() * np.sqrt(252)
    f["d_spy_vol"] = f["spy_vol_21"] / (r.rolling(126).std() * np.sqrt(252)) - 1
    f = f.join(breadth_panel(stocks or {}, idx, exclude=set()))
    f["d_breadth_200_63"] = f["breadth_200"] - f["breadth_200"].shift(63)
    near_high = (f["spy_dist_high"] > -0.03).astype(float)
    f["breadth_divergence"] = (near_high * (-f["d_breadth_200_63"]).clip(lower=0)).where(f["d_breadth_200_63"].notna())
    if vol is not None and len(vol):
        v = vol.reindex(idx).ffill(limit=5)
        f["vix"] = v
        f["vix_vs_min63"] = v / v.rolling(63, min_periods=40).min() - 1
        f["vol_source"] = "index"
    else:   # Ersatz: realisierte Volatilität des Index
        f["vix"] = np.nan
        f["vix_vs_min63"] = f["spy_vol_21"] / f["spy_vol_21"].rolling(63, min_periods=40).min() - 1
        f["vol_source"] = "realisiert"
    if hyg is not None and ief is not None and len(hyg) and len(ief):
        ratio = (hyg.reindex(idx).ffill(limit=5) / ief.reindex(idx).ffill(limit=5))
        f["credit_vs_sma50"] = ratio / ratio.rolling(50, min_periods=50).mean() - 1
        f["credit_mom_63"] = ratio / ratio.shift(63) - 1
    else:
        f["credit_vs_sma50"] = np.nan
        f["credit_mom_63"] = np.nan
    if fx is not None and len(fx):
        x = fx.reindex(idx).ffill(limit=5)
        f["fx_mom_20"] = x / x.shift(20) - 1
        f["fx_vs_sma50"] = x / x.rolling(50, min_periods=50).mean() - 1
    else:
        f["fx_mom_20"] = np.nan
        f["fx_vs_sma50"] = np.nan
    f["ret_1d"] = s.pct_change()
    # Divergenz-Tage: Index fällt ≥ 0.5 %, aber mehr als 55 % der Aktien steigen (Zählung der letzten 20 Tage)
    div_day = ((f["ret_1d"] <= -0.005) & (f["adv_share"] > 0.55)).astype(float).where(f["adv_share"].notna())
    f["div_days_20"] = div_day.rolling(20, min_periods=1).sum().where(f["adv_share"].notna())
    f["ad_63"] = (f["ad_line"] - f["ad_line"].shift(63)) / f["n_stocks"].where(f["n_stocks"] > 0)
    f["ad_divergence"] = ((f["spy_dist_high"] > -0.03) & (f["ad_63"] < 0)).astype(float).where(f["ad_63"].notna())
    ex = {k: v.reindex(idx).ffill(limit=5) for k, v in (extras or {}).items() if v is not None and len(v)}

    def rel(a: str, b_: str | None, n: int) -> pd.Series:
        if a not in ex or (b_ is not None and b_ not in ex and b_ != "MAIN"):
            return pd.Series(np.nan, index=idx)
        base = s if b_ == "MAIN" else (ex[b_] if b_ else 1.0)
        r = ex[a] / base
        return r / r.shift(n) - 1
    f["ew_vs_cap_63"] = rel("RSP", "MAIN", 63)
    f["small_vs_large_63"] = rel("IWM", "MAIN", 63)
    f["cyc_vs_def_63"] = rel("XLY", "XLP", 63)
    f["copper_gold_126"] = rel("HG=F", "GC=F", 126)
    f["vix_term"] = (ex["^VIX"] / ex["^VIX3M"]) if "^VIX" in ex and "^VIX3M" in ex else np.nan
    f["yield_curve"] = (ex["^TNX"] - ex["^IRX"]) if "^TNX" in ex and "^IRX" in ex else np.nan
    f["credit_mom_252"] = (hyg.reindex(idx).ffill(limit=5) / ief.reindex(idx).ffill(limit=5)).pct_change(252) \
        if hyg is not None and ief is not None and len(hyg) and len(ief) else np.nan
    f["regime"] = np.where(f["spy_vs_sma200"].isna(), "unbekannt", np.where(f["spy_vs_sma200"] > 0, "Bull", "Bear"))
    return f


def ampel(f: pd.DataFrame) -> pd.DataFrame:
    near = f["spy_dist_high"] > -0.03
    vol_thr = 0.30 if (f.get("vol_source") == "index").any() else 0.50   # realisierte Vol schwankt stärker
    a = pd.DataFrame(index=f.index)
    a["a_breadth"] = (near & (f["d_breadth_200_63"] <= -0.10)).astype(int)
    a["a_vix"] = ((f["vix_vs_min63"] >= vol_thr) & (f["spy_dist_high"] > -0.05)).astype(int)
    a["a_credit"] = (near & (f["credit_vs_sma50"] < -0.005)).astype(int)
    a["a_stretch"] = (f["spy_vs_sma200"] > 0.12).astype(int)
    a["a_fx"] = (f["fx_mom_20"] <= -0.03).astype(int)      # USD/JPY fällt > 3 % in 1 Monat = Yen-Stärke
    a["a_ew"] = (near & (f["ew_vs_cap_63"] <= -0.03)).astype(int)          # Gleichgewichtet hinkt deutlich hinterher
    a["a_div"] = (near & (f["div_days_20"] >= 2)).astype(int)              # Index fällt, Mehrheit steigt
    a["a_below20"] = ((f["spy_dist_high"] > -0.05) & (f["pct_below20"] >= 0.35)).astype(int)
    a["ampel_score"] = a[list(AMPEL)].sum(axis=1)
    return a


def available_signs(f: pd.DataFrame) -> list[str]:
    out = []
    if f["breadth_200"].notna().any():
        out.append("a_breadth")
    out.append("a_vix")
    if f["credit_vs_sma50"].notna().any():
        out.append("a_credit")
    out.append("a_stretch")
    if f["fx_mom_20"].notna().any():
        out.append("a_fx")
    if f["ew_vs_cap_63"].notna().any():
        out.append("a_ew")
    if f["div_days_20"].notna().any():
        out.append("a_div")
    if f["pct_below20"].notna().any():
        out.append("a_below20")
    return out


def forward_labels(spy: pd.Series, horizon: int = HORIZON, threshold: float = DD_THRESHOLD) -> pd.DataFrame:
    """Tiefster Schlusskurs der nächsten `horizon` Handelstage relativ zu heute."""
    s = spy.sort_index()
    vals = s.values
    n = len(vals)
    fwd_min = np.full(n, np.nan)
    for i in range(n - horizon):
        fwd_min[i] = vals[i + 1:i + 1 + horizon].min()
    out = pd.DataFrame(index=s.index)
    out["fwd_dd"] = fwd_min / vals - 1
    out["y_top"] = (out["fwd_dd"] <= threshold).astype(float).where(out["fwd_dd"].notna())
    idx = s.index
    end = [idx[i + horizon] if i + horizon < n else pd.NaT for i in range(n)]
    out["label_end"] = end
    return out


# ------------------------------------------------------------------ Modell
def make_top_model():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000))


def usable(train: pd.DataFrame, feats: list[str]) -> list[str]:
    return [f for f in feats if train[f].notna().sum() >= 100 and train[f].nunique() >= 2]


def walk_forward_top(df: pd.DataFrame, feats: list[str], first_test_year: int = FIRST_TEST_YEAR,
                     step: int = 5) -> tuple[pd.DataFrame, list]:
    lab = df[df["y_top"].notna()]
    rows, folds = [], []
    years = sorted(set(df.index.year))
    for y in years:
        if y < first_test_year:
            continue
        test = df[df.index.year == y]
        train = lab[pd.to_datetime(lab["label_end"]) < pd.Timestamp(y, 1, 1)].iloc[::step]
        if len(train) < 150 or train["y_top"].nunique() < 2 or test.empty:
            continue
        used = usable(train, feats)
        m = make_top_model().fit(train[used], train["y_top"].astype(int))
        p = m.predict_proba(test[used])[:, 1]
        rows.append(pd.DataFrame({"prob": p, "fold": y, "base_rate": train["y_top"].mean()}, index=test.index))
        folds.append((y, m, used, float(train["y_top"].mean())))
    return (pd.concat(rows) if rows else pd.DataFrame(columns=["prob", "fold", "base_rate"])), folds


def drawdown_episodes(close: pd.Series, threshold: float = DD_THRESHOLD) -> pd.DataFrame:
    """Einbrüche ≥ |threshold| nach dem Zickzack-Prinzip: Ein Hoch gilt als Top, wenn danach ein Rückgang
    ≥ 10 % folgt; die Episode endet, wenn der Kurs vom Tief wieder ≥ 10 % steigt. So zählen auch Einbrüche
    in langen Bärenmärkten (z. B. Nikkei 1990–2012) einzeln."""
    s = close.dropna().sort_index()
    if s.empty:
        return pd.DataFrame(columns=["peak", "trough", "max_dd", "recovered"])
    up, ext_val, ext_date = True, s.iloc[0], s.index[0]
    peak_val = peak_date = None
    eps = []
    for d, v in s.items():
        if up:
            if v > ext_val:
                ext_val, ext_date = v, d
            elif v <= ext_val * (1 + threshold):
                peak_val, peak_date = ext_val, ext_date
                up, ext_val, ext_date = False, v, d
        else:
            if v < ext_val:
                ext_val, ext_date = v, d
            elif v >= ext_val / (1 + threshold):
                eps.append({"peak": peak_date, "trough": ext_date, "max_dd": ext_val / peak_val - 1, "recovered": d})
                up, ext_val, ext_date = True, v, d
    if not up:
        eps.append({"peak": peak_date, "trough": ext_date, "max_dd": ext_val / peak_val - 1, "recovered": pd.NaT})
    return pd.DataFrame(eps)


def evaluate_warnings(df: pd.DataFrame, warn: pd.Series, episodes: pd.DataFrame, start: pd.Timestamp,
                      lookback: int = HORIZON) -> tuple[pd.DataFrame, dict]:
    """Erkannte Einbrüche (Warnung in den `lookback` Handelstagen vor dem Hoch oder bis 10 %-Marke) und Fehlalarme."""
    idx = df.index
    ep = episodes[pd.to_datetime(episodes["peak"]) >= start].copy()
    rows = []
    for e in ep.itertuples():
        i_peak = idx.get_indexer([e.peak])[0]
        # Bis zu dem Tag, an dem -10 % erstmals erreicht wurden
        after = df["close"].iloc[i_peak:]
        hit = after[after / after.iloc[0] - 1 <= DD_THRESHOLD]
        i_hit = idx.get_indexer([hit.index[0]])[0] if len(hit) else i_peak
        window = warn.iloc[max(0, i_peak - lookback):i_hit + 1]
        first = window[window == 1]
        rows.append({"Hoch": e.peak.date(), "Tief": e.trough.date(), "Max. Rückgang": e.max_dd,
                     "Gewarnt": bool(len(first)),
                     "Erste Warnung": first.index[0].date() if len(first) else None,
                     "Vorlauf zum Hoch (Handelstage)": int(i_peak - idx.get_indexer([first.index[0]])[0])
                     if len(first) else None})
    tab = pd.DataFrame(rows)
    w = warn[warn.index >= start]
    yt = df.loc[w.index, "y_top"]
    hit_rate = float(yt[(w == 1) & yt.notna()].mean()) if ((w == 1) & yt.notna()).any() else np.nan
    # Fehlalarme: Warnphasen (Lücken < 10 Handelstage zusammengefasst), nach denen kein Einbruch kam
    wf = w.rolling(10, min_periods=1).max().shift(-9).fillna(w).astype(int) | w
    runs = (wf != wf.shift()).cumsum()
    false_alarms, n_runs = 0, 0
    for _, g in w[wf == 1].groupby(runs[wf == 1]):
        n_runs += 1
        if not (df.loc[g.index, "y_top"] == 1).any():
            false_alarms += 1
    stats = {"episodes": int(len(tab)), "detected": int(tab["Gewarnt"].sum()) if len(tab) else 0,
             "warning_days_share": float((w == 1).mean()) if len(w) else np.nan,
             "precision": hit_rate, "base_rate": float(yt.mean()), "warning_phases": n_runs,
             "false_alarm_phases": false_alarms}
    return tab, stats


def red_phase_stats(f: pd.DataFrame, merge_gap: int = 10) -> pd.DataFrame:
    """Was passierte nach Beginn einer Bear-Phase (Index unter 200-Tage-Schnitt)?
    Phasen, die weniger als `merge_gap` Handelstage auseinanderliegen, gelten als eine."""
    bear = (f["regime"] == "Bear").values
    close = f["close"].values
    idx = f.index
    starts, last_end = [], -10_000
    i = 0
    while i < len(bear):
        if bear[i]:
            j = i
            while j + 1 < len(bear) and bear[j + 1]:
                j += 1
            if i - last_end > merge_gap:
                starts.append(i)
            last_end = j
            i = j + 1
        else:
            i += 1
    rows = []
    hi252 = f["close"].rolling(252, min_periods=50).max().values
    for i in starts:
        r = {"Beginn Rot": idx[i].date(), "Kurs": close[i],
             "Schon vom Hoch gefallen": close[i] / hi252[i] - 1 if not np.isnan(hi252[i]) else np.nan}
        for lab, h in (("1 Monat", 21), ("3 Monate", 63), ("6 Monate", 126), ("12 Monate", 252)):
            r[f"Rendite {lab}"] = close[i + h] / close[i] - 1 if i + h < len(close) else np.nan
        r["Weiterer max. Rückgang (12M)"] = close[i + 1:i + 253].min() / close[i] - 1 if i + 1 < len(close) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def run_top_market(main: pd.Series, stocks: dict | None = None, vol: pd.Series | None = None,
                   hyg: pd.Series | None = None, ief: pd.Series | None = None, fx: pd.Series | None = None,
                   first_test_year: int = FIRST_TEST_YEAR, extras: dict | None = None) -> dict:
    f = market_features(main, stocks, vol, hyg, ief, fx, extras)
    f = f.join(ampel(f)).join(forward_labels(f["close"]))
    feats = [x for x in TOP_FEATURES if f[x].notna().any()]
    oos, folds = walk_forward_top(f, feats, first_test_year)
    f["prob_oos"] = oos["prob"].reindex(f.index)
    lab = f[f["y_top"].notna()]
    used = usable(lab.iloc[::5], feats)
    final = make_top_model().fit(lab[used].iloc[::5], lab["y_top"].iloc[::5].astype(int))
    f["prob_live"] = final.predict_proba(f[used])[:, 1]
    # Für die Anzeige: Out-of-sample, wo vorhanden; sonst (offene letzte 63 Tage) das Endmodell
    raw = f["prob_oos"].where(f["prob_oos"].notna() & f["y_top"].notna(), f["prob_live"])
    raw[f.index < pd.Timestamp(first_test_year, 1, 1)] = np.nan
    f["prob_raw"] = raw
    f["prob"] = raw.rolling(5, min_periods=1).mean().where(raw.notna())   # glättet tägliches Flackern
    base = float(lab["y_top"].mean())
    thr = 2 * base   # Warnung, wenn das Risiko mindestens doppelt so hoch ist wie im Durchschnitt
    f["warn_ml"] = (f["prob"] >= thr).astype(int).where(f["prob"].notna(), 0)
    start = pd.Timestamp(first_test_year, 1, 1)
    eps = drawdown_episodes(f["close"])
    ep_ml, st_ml = evaluate_warnings(f, f["warn_ml"], eps, start)
    ep_amp, st_amp = evaluate_warnings(f, (f["ampel_score"] >= 2).astype(int), eps, start)
    test = f[f["prob_oos"].notna() & f["y_top"].notna()]
    auc = float(roc_auc_score(test["y_top"], test["prob_oos"])) if test["y_top"].nunique() == 2 else np.nan
    names = list(final[0].get_feature_names_out(used))
    coef = pd.DataFrame({"feature": names, "coef": final[-1].coef_[0]})
    coef = coef[~coef["feature"].str.startswith("missingindicator")]
    coef = coef.sort_values("coef", key=np.abs, ascending=False)
    episodes_tab = ep_ml.merge(ep_amp[["Hoch", "Gewarnt", "Vorlauf zum Hoch (Handelstage)"]]
                               .rename(columns={"Gewarnt": "Ampel gewarnt",
                                                "Vorlauf zum Hoch (Handelstage)": "Ampel Vorlauf"}),
                               on="Hoch", how="left") if len(ep_ml) else ep_ml
    return {"daily": f.drop(columns=["vol_source"]), "episodes": episodes_tab, "coef": coef, "red": red_phase_stats(f),
            "meta": {"auc_oos": auc, "base_rate": base, "threshold": thr, "ml": st_ml, "ampel": st_amp,
                     "n_folds": len(folds), "horizon_days": HORIZON, "dd_threshold": DD_THRESHOLD,
                     "first_test_year": first_test_year, "first_date": str(f.index.min().date()),
                     "ampel_signs": available_signs(f),
                     "vol_source": str(f["vol_source"].iloc[-1]),
                     "has_vix": bool(f["vix"].notna().any()), "has_credit": bool(f["credit_vs_sma50"].notna().any()),
                     "has_fx": bool(f["fx_mom_20"].notna().any()),
                     "n_stocks_breadth": int(f["n_stocks"].max()) if pd.notna(f["n_stocks"].max()) else 0}}


def run_top(prices: dict[str, pd.Series], spy: pd.Series, first_test_year: int = FIRST_TEST_YEAR) -> dict:
    """Kompatibilität: S&P 500 aus einem Kurs-Dict (Aktien + ^VIX, HYG, IEF)."""
    stocks = {k: v for k, v in prices.items() if k not in {"SPY", *EXTRA_TICKERS}}
    return run_top_market(spy, stocks, prices.get("^VIX"), prices.get("HYG"), prices.get("IEF"), None,
                          first_test_year)
