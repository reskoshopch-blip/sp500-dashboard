"""Merkmale aus einem GEX-Schnappschuss (Antwort von /gex/{symbol}).

Begriffe
* GEX je Strike: Gamma-Exposure der Market Maker (Calls positiv, Puts negativ, Summe = totalGex).
* Gamma-Flip: Kurs, bei dem die Netto-Gamma von positiv nach negativ kippt. Über dem Flip (positive Gamma) sichern
  Market Maker gegen die Bewegung ab → Kurse werden gedämpft, eher Seitwärts/Mean-Reversion. Darunter (negative
  Gamma) sichern sie mit der Bewegung ab → grössere, trendige Bewegungen.
* Call-Wall: Strike mit der grössten Call-Gamma (oft Widerstand). Put-Wall: Strike mit der grössten Put-Gamma
  (oft Unterstützung). Max Gamma: Strike mit der grössten absoluten Netto-Gamma (Magnet).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


def _arr(d: dict, key: str) -> np.ndarray:
    v = d.get(key)
    if v is None:
        return np.array([], dtype=float)
    return np.asarray([np.nan if x is None else x for x in v], dtype=float)


def profile(resp: dict) -> pd.DataFrame:
    d = resp.get("data", resp)
    k, c, p, t = _arr(d, "strikes"), _arr(d, "callGex"), _arr(d, "putGex"), _arr(d, "totalGex")
    n = min(len(k), len(c) or len(k), len(p) or len(k))
    if n == 0:
        return pd.DataFrame(columns=["strike", "call", "put", "total"])
    c = c[:n] if len(c) else np.full(n, np.nan)
    p = p[:n] if len(p) else np.full(n, np.nan)
    t = t[:n] if len(t) >= n else np.nan_to_num(c) + np.nan_to_num(p)
    return pd.DataFrame({"strike": k[:n], "call": c, "put": p, "total": t}).sort_values("strike").reset_index(drop=True)


def _flip_from_profile(pr: pd.DataFrame, spot: float) -> float | None:
    """Kumulierte Netto-Gamma über die Strikes; Nullstelle am nächsten zum Spot (Rückfall, falls die API keinen
    Flip liefert)."""
    if pr.empty:
        return None
    cum = pr["total"].cumsum().values
    s = pr["strike"].values
    cands = []
    for i in range(1, len(cum)):
        if np.sign(cum[i - 1]) != np.sign(cum[i]) and cum[i - 1] != 0:
            x = s[i - 1] + (s[i] - s[i - 1]) * (-cum[i - 1]) / (cum[i] - cum[i - 1])
            cands.append(x)
    return float(min(cands, key=lambda x: abs(x - spot))) if cands else None


def features(resp: dict, symbol: str, taken_at: datetime) -> dict:
    d = resp.get("data", resp)
    pr = profile(resp)
    kl = d.get("keyLevels") or {}
    spot = d.get("spotPrice") or kl.get("spotPrice")
    spot = float(spot) if spot is not None else np.nan
    out = {"symbol": symbol, "taken_at": pd.Timestamp(taken_at), "spot": spot,
           "n_strikes": len(pr), "n_exp": len(d.get("expirations") or [])}
    exps = sorted(str(e) for e in (d.get("expirations") or []))
    if exps:
        try:
            first = pd.Timestamp(datetime.strptime(exps[0], "%Y%m%d"))
            out["days_to_exp"] = float((first - pd.Timestamp(taken_at).tz_localize(None).normalize()).days)
        except ValueError:
            pass
    if pr.empty or not np.isfinite(spot):
        return out
    out["call_gex"] = float(np.nansum(pr["call"]))
    out["put_gex"] = float(np.nansum(pr["put"]))
    out["total_gex"] = float(np.nansum(pr["total"]))
    gross = np.nansum(np.abs(pr["call"])) + np.nansum(np.abs(pr["put"]))
    out["net_ratio"] = out["total_gex"] / gross if gross else np.nan          # -1 … +1
    near = pr[(pr["strike"] >= spot * 0.99) & (pr["strike"] <= spot * 1.01)]
    out["gex_near_spot"] = float(near["total"].sum()) if len(near) else np.nan
    band = pr[(pr["strike"] >= spot * 0.98) & (pr["strike"] <= spot * 1.02)]
    tot_abs = np.nansum(np.abs(pr["total"]))
    out["concentration"] = float(np.nansum(np.abs(band["total"])) / tot_abs) if tot_abs else np.nan
    flip = kl.get("gammaFlip")
    flip = float(flip) if flip is not None else _flip_from_profile(pr, spot)
    maxg = kl.get("maxGamma")
    maxg = float(maxg) if maxg is not None else float(pr.loc[pr["total"].abs().idxmax(), "strike"])
    call_wall = float(pr.loc[pr["call"].idxmax(), "strike"]) if pr["call"].notna().any() else np.nan
    put_wall = float(pr.loc[pr["put"].idxmin(), "strike"]) if pr["put"].notna().any() else np.nan
    out.update({"flip": flip if flip is not None else np.nan, "max_gamma": maxg, "call_wall": call_wall,
                "put_wall": put_wall})
    for k in ("flip", "max_gamma", "call_wall", "put_wall"):
        v = out[k]
        out[f"dist_{k}"] = (spot / v - 1) if v and np.isfinite(v) else np.nan
    out["regime"] = ("positiv" if spot >= out["flip"] else "negativ") if np.isfinite(out["flip"]) else \
        ("positiv" if out["total_gex"] >= 0 else "negativ")
    if np.isfinite(call_wall) and np.isfinite(put_wall) and call_wall > put_wall:
        out["wall_pos"] = (spot - put_wall) / (call_wall - put_wall)            # 0 = an Put-Wall, 1 = an Call-Wall
        out["wall_width"] = call_wall / put_wall - 1
    return out


def zeta_features(resp: dict) -> dict:
    d = resp.get("data", resp)
    now = (d.get("data") or {}).get("now") or {}
    return {"zeta_regime": now.get("regime"), "zeta_drift": now.get("drift"), "zeta_risk": now.get("risk"),
            "zeta_computed": d.get("computedAt")}
