"""Plotly-Diagramme mit validierter Farbpalette (dataviz-Referenzpalette, hell/dunkel separat gestuft).

Regeln: Farben folgen der Entität (nicht dem Rang), nie zwei y-Achsen, dünne Linien (2px),
Legende ab 2 Serien, Hover auf jedem Diagramm, Text immer in Text-Farben.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PALETTE = {
    "light": {"series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
              "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "missing": "#e1e0d9",
              "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
              "div": ["#e34948", "#f0efec", "#2a78d6"]},
    "dark": {"series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
             "surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "missing": "#383835",
             "seq": ["#0d366b", "#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"],
             "div": ["#e66767", "#383835", "#3987e5"]},
}
STATUS = {"ok": "#0ca30c", "teilweise": "#fab219", "NA": "#d03b3b"}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
# Feste Farbzuordnung der Bilanzseiten (Slots 1–3 sind auch paarweise unterscheidbar)
SECTION_SLOT = {"Aktiva": 0, "Verbindlichkeiten": 1, "Eigenkapital": 2, "Mezzanine": 3}
REGIME_ORDER = ["Alle", "Bull", "Bear"]   # feste Farben: Alle = Blau, Bull = Orange, Bear = Grün


def fmt_usd(v, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return "Nicht verfügbar"
    a = abs(v)
    for div, unit in ((1e12, "Bio."), (1e9, "Mrd."), (1e6, "Mio."), (1e3, "Tsd.")):
        if a >= div:
            return f"{v / div:,.{digits}f} {unit} USD".replace(",", "'")
    return f"{v:,.0f} USD".replace(",", "'")


def fmt_metric(v, kind: str) -> str:
    if v is None or pd.isna(v):
        return "NA"
    if kind == "pct":
        return f"{v * 100:,.1f} %"
    if kind == "usd":
        return fmt_usd(v)
    return f"{v:,.2f}"


def base_layout(fig: go.Figure, mode: str, title: str = "", height: int = 420, y_title: str = "") -> go.Figure:
    p = PALETTE[mode]
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=p["text"]), x=0.01, xanchor="left",
                   y=0.985, yref="container", yanchor="top"),
        paper_bgcolor=p["surface"], plot_bgcolor=p["surface"], height=height,
        font=dict(family=FONT, color=p["text2"], size=12),
        margin=dict(l=12, r=16, t=(84 if len(fig.data) > 1 else 52) if title else 24, b=12, autoexpand=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(color=p["text2"]),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=p["surface"], font=dict(color=p["text"], family=FONT), bordercolor=p["axis"]),
        hovermode="x unified" if fig.data and fig.data[0].type == "scatter" else "closest",
        bargap=0.25, bargroupgap=0.06,
    )
    fig.update_xaxes(showgrid=False, linecolor=p["axis"], tickfont=dict(color=p["muted"]), zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=p["grid"], gridwidth=1, linecolor=p["axis"], tickfont=dict(color=p["muted"]), automargin=True,
                     zerolinecolor=p["axis"], title_font=dict(color=p["muted"]))
    fig.update_xaxes(title_font=dict(color=p["muted"]))
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return fig


def color_map(entities: list, mode: str, stable_order: list | None = None) -> dict:
    """Farbe hängt an der Entität (Position in der stabilen Reihenfolge), nicht am Rang.
    Mehr als 8 Serien erhalten Grau – Diagramme sollten dann gefiltert werden."""
    order = stable_order or entities
    s = PALETTE[mode]["series"]
    out = {}
    for e in entities:
        i = order.index(e) if e in order else len(s)
        out[e] = s[i] if i < len(s) else PALETTE[mode]["muted"]
    return out


def line_chart(df: pd.DataFrame, x: str, y: str, series: str, mode: str, title: str, kind: str = "usd",
               stable_order: list | None = None, y_title: str = "") -> go.Figure:
    fig = go.Figure()
    ents = list(dict.fromkeys(df[series]))
    cmap = color_map(ents, mode, stable_order)
    for e in ents:
        d = df[df[series] == e]
        xnum = pd.to_numeric(d[x], errors="coerce")
        if xnum.notna().all():
            d = d.iloc[np.argsort(xnum.values, kind="stable")]
        fig.add_trace(go.Scatter(
            x=d[x], y=_scale(d[y], kind), name=str(e), mode="lines+markers", line=dict(width=2, color=cmap[e]),
            marker=dict(size=8, color=cmap[e], line=dict(width=2, color=PALETTE[mode]["surface"])),
            connectgaps=False, customdata=[fmt_metric(v, kind) for v in d[y]],
            hovertemplate=f"{e}: %{{customdata}}<extra></extra>"))
    fig.update_xaxes(type="category")
    _y_format(fig, kind)
    return base_layout(fig, mode, title, y_title=y_title)


def bar_chart(df: pd.DataFrame, x: str, y: str, series: str | None, mode: str, title: str, kind: str = "usd",
              horizontal: bool = False, stable_order: list | None = None, height: int = 420,
              stack: bool = False) -> go.Figure:
    fig = go.Figure()
    p = PALETTE[mode]
    groups = list(dict.fromkeys(df[series])) if series else [None]
    cmap = color_map(groups, mode, stable_order) if series else {None: p["series"][0]}
    for g in groups:
        d = df if g is None else df[df[series] == g]
        kw = dict(x=_scale(d[y], kind), y=d[x], orientation="h") if horizontal else dict(x=d[x], y=_scale(d[y], kind))
        fig.add_trace(go.Bar(name=str(g) if g is not None else y, marker=dict(
            color=cmap[g], line=dict(width=2 if stack else 0, color=p["surface"]), cornerradius=4),
            customdata=[fmt_metric(v, kind) for v in d[y]],
            hovertemplate=("%{y}" if horizontal else "%{x}") + ": %{customdata}<extra>" + (str(g) if g else "") + "</extra>",
            showlegend=series is not None, **kw))
    fig.update_layout(barmode="stack" if stack else "group")
    if horizontal:
        fig.update_yaxes(autorange="reversed", type="category")
        _x_format(fig, kind)
        fig.update_xaxes(gridcolor=p["grid"], showgrid=True)
        fig.update_yaxes(showgrid=False)
    else:
        fig.update_xaxes(type="category")
        _y_format(fig, kind)
    return base_layout(fig, mode, title, height=height)


def balance_structure(df: pd.DataFrame, mode: str, title: str) -> go.Figure:
    """Gestapelte Bilanzstruktur pro Jahr: links Aktiva, rechts Passiva (Verbindlichkeiten + Mezzanine + EK).
    df: fiscal_year, total_assets, total_liabilities, temporary_equity, equity_total."""
    p = PALETTE[mode]
    fig = go.Figure()
    years = [str(int(y)) for y in df["fiscal_year"]]
    parts = [("Aktiva", "total_assets", "Aktiva"), ("Verbindlichkeiten", "total_liabilities", "Passiva"),
             ("Mezzanine", "temporary_equity", "Passiva"), ("Eigenkapital", "equity_total", "Passiva")]
    for name, col, group in parts:
        if col not in df or df[col].isna().all():
            continue
        vals = df[col]
        fig.add_trace(go.Bar(
            x=[years, [group] * len(years)], y=_scale(vals, "usd"), name=name,
            marker=dict(color=p["series"][SECTION_SLOT[name]], line=dict(width=2, color=p["surface"]), cornerradius=4),
            customdata=[fmt_usd(v) for v in vals], hovertemplate=f"{name}: %{{customdata}}<extra></extra>"))
    fig.update_layout(barmode="relative")
    _y_format(fig, "usd")
    fig = base_layout(fig, mode, title)
    fig.update_layout(bargap=0.12)
    fig.update_xaxes(tickangle=0)
    return fig


def heatmap(z: pd.DataFrame, text: pd.DataFrame, mode: str, title: str, color_by_column_rank: bool = True,
            height: int | None = None) -> go.Figure:
    """Werte als Text, Farbe als Perzentil je Spalte (Kennzahlen haben unterschiedliche Skalen).
    Fehlende Werte erscheinen neutral mit 'NA'."""
    p = PALETTE[mode]
    zc = z.rank(pct=True) if color_by_column_rank else z
    seq = p["seq"]
    scale = [[i / (len(seq) - 1), c] for i, c in enumerate(seq)]
    fig = go.Figure(go.Heatmap(
        z=zc.values, x=list(z.columns), y=list(z.index), text=text.values, texttemplate="%{text}",
        textfont=dict(size=11), colorscale=scale, showscale=False, xgap=2, ygap=2,
        hovertemplate="%{y} · %{x}: %{text}<extra></extra>", zmin=0, zmax=1))
    na = z.isna()
    if na.values.any():
        fig.add_trace(go.Heatmap(z=np.where(na.values, 1, np.nan), x=list(z.columns), y=list(z.index),
                                 colorscale=[[0, p["missing"]], [1, p["missing"]]], showscale=False, xgap=2, ygap=2,
                                 hoverinfo="skip"))
        fig.data = (fig.data[1], fig.data[0])
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(side="top", type="category")
    return base_layout(fig, mode, title, height=height or max(320, 26 * len(z.index) + 120))


def coverage_heatmap(cov: pd.DataFrame, mode: str, title: str) -> go.Figure:
    """cov: Index Zeilen, Spalten Jahre, Werte 2=gemeldet, 1=berechnet, 0=fehlend."""
    p = PALETTE[mode]
    labels = {2: "gemeldet", 1: "berechnet", 0: "fehlt"}
    colors = [[0, p["missing"]], [0.33, p["missing"]], [0.34, p["series"][2]], [0.66, p["series"][2]],
              [0.67, p["series"][0]], [1, p["series"][0]]]
    text = cov.map(lambda v: labels.get(int(v), "") if pd.notna(v) else "")
    fig = go.Figure(go.Heatmap(z=cov.values, x=[str(c) for c in cov.columns], y=list(cov.index), zmin=0, zmax=2,
                               colorscale=colors, showscale=False, xgap=2, ygap=2, text=text.values,
                               hovertemplate="%{y} · %{x}: %{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(side="top", type="category")
    return base_layout(fig, mode, title, height=max(300, 22 * len(cov.index) + 120))


def _scale(values, kind: str):
    """USD-Werte werden in Mrd. USD dargestellt (Achsentitel weist darauf hin); Hover zeigt den exakten Wert."""
    return pd.Series(values, dtype="float64") / 1e9 if kind == "usd" else values


def _y_format(fig: go.Figure, kind: str) -> None:
    if kind == "pct":
        fig.update_yaxes(tickformat=".1~%")
    elif kind == "usd":
        fig.update_yaxes(tickformat=",.0f", title_text="Mrd. USD")


def _x_format(fig: go.Figure, kind: str) -> None:
    if kind == "pct":
        fig.update_xaxes(tickformat=".1~%")
    elif kind == "usd":
        fig.update_xaxes(tickformat=",.0f", title_text="Mrd. USD")


def _runs(flags: np.ndarray, dates: np.ndarray):
    """Zusammenhängende Abschnitte, in denen flags True ist -> (start, ende)."""
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = dates[i]
        if start is not None and (not f or i == len(flags) - 1):
            yield start, dates[i]
            start = None


WARN_COLORS = {"ml": "#ec835a", "ampel": "#fab219"}   # Status "serious" / "warning"


def regime_chart(mf: pd.DataFrame, mode: str, title: str, warn: dict | None = None,
                 main_label: str = "SPY (inkl. Dividenden)", y_title: str = "Kurs in USD (log. Skala)") -> go.Figure:
    """SPY mit 200-Tage-Durchschnitt; Bull-Phasen grün, Bear-Phasen rot hinterlegt.
    warn: optionale Warnstreifen am unteren Rand, z. B. {"Top-Warnung (ML)": ("ml", Serie 0/1)}."""
    p = PALETTE[mode]
    green, red = STATUS["ok"], STATUS["NA"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mf["date"], y=mf["close"], name=main_label, mode="lines",
                             line=dict(width=2, color=p["series"][0]),
                             hovertemplate="Kurs: %{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=mf["date"], y=mf["sma200"], name="200-Tage-Durchschnitt", mode="lines",
                             line=dict(width=2, color=p["text2"], dash="dot"),
                             hovertemplate="SMA200: %{y:,.2f}<extra></extra>"))
    dates = mf["date"].values
    op_bull, op_bear = (0.10, 0.22) if mode == "dark" else (0.10, 0.18)
    for regime, color, op in (("Bull", green, op_bull), ("Bear", red, op_bear)):
        for x0, x1 in _runs((mf["regime"] == regime).values, dates):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=op, line_width=0, layer="below")
        # Legendeneintrag für die Hinterlegung (Farbe nie allein: Name im Legendentext)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"{regime}-Phase",
                                 marker=dict(symbol="square", size=12, color=color, opacity=0.6)))
    for k, (label, (ckey, series)) in enumerate((warn or {}).items()):
        sr = series.reindex(pd.to_datetime(mf["date"])).fillna(0).astype(bool).values
        y0, y1 = 0.0 + k * 0.045, 0.04 + k * 0.045
        for x0, x1 in _runs(sr, dates):
            fig.add_shape(type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=y0, y1=y1,
                          fillcolor=WARN_COLORS[ckey], opacity=0.95, line_width=0, layer="above")
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                 marker=dict(symbol="square", size=12, color=WARN_COLORS[ckey])))
    lo, hi = float(mf["close"].min()), float(mf["close"].max())
    cand = [m * 10 ** k for k in range(0, 6) for m in (1, 1.5, 2, 3, 5, 7)]
    ticks = [v for v in cand if lo * 0.9 <= v <= hi * 1.1]
    if len(ticks) > 9:
        ticks = [v for v in ticks if str(int(v))[0] in "125"]
    ticks = [int(v) if v >= 1 else v for v in ticks]
    fig.update_yaxes(type="log", title_text=y_title, tickvals=ticks,
                     ticktext=[f"{v:,}".replace(",", "'") for v in ticks])
    fig = base_layout(fig, mode, title, height=440)
    fig.update_layout(hovermode="x unified")
    return fig


def driver_chart(df: pd.DataFrame, mode: str, title: str, up_is_good: bool = True) -> go.Figure:
    """Horizontale Balken: Beitrag je Merkmal in Prozentpunkten (positiv = erhöht die Wahrscheinlichkeit)."""
    p = PALETTE[mode]
    d = df.sort_values("delta")
    good, bad = p["div"][2], p["div"][0]
    colors = [(good if up_is_good else bad) if v > 0 else (bad if up_is_good else good) for v in d["delta"]]
    fig = go.Figure(go.Bar(x=d["delta"] * 100, y=d["label"], orientation="h",
                           marker=dict(color=colors, cornerradius=4),
                           customdata=d["value_txt"],
                           hovertemplate="%{y}<br>Wert: %{customdata}<br>Beitrag: %{x:+.1f} Pp.<extra></extra>"))
    fig.update_xaxes(title_text="Beitrag zur Wahrscheinlichkeit (Prozentpunkte)", zeroline=True, showgrid=True,
                     gridcolor=p["grid"])
    fig.update_yaxes(type="category")
    return base_layout(fig, mode, title, height=max(300, 30 * len(d) + 110))


def date_lines(df: pd.DataFrame, x: str, cols: dict[str, str], mode: str, title: str, kind: str = "pct",
               height: int = 320, hline: float | None = None, hline_label: str = "") -> go.Figure:
    """Zeitreihen mit Datumsachse; cols = {Spalte: Legendenname}."""
    p = PALETTE[mode]
    fig = go.Figure()
    for i, (c, name) in enumerate(cols.items()):
        fig.add_trace(go.Scatter(x=df[x], y=df[c], name=name, mode="lines", line=dict(width=2, color=p["series"][i]),
                                 hovertemplate=f"{name}: %{{y:{'.1%' if kind == 'pct' else ',.2f'}}}<extra></extra>"))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color=p["text2"], width=1.5, dash="dash"),
                      annotation_text=hline_label, annotation_position="top left",
                      annotation_font=dict(color=p["text2"], size=11))
    if kind == "pct":
        fig.update_yaxes(tickformat=".0%")
    fig = base_layout(fig, mode, title, height=height)
    fig.update_layout(hovermode="x unified", showlegend=len(cols) > 1)
    return fig


GAUGE_STEPS = [(0, 20, "rgba(208,59,59,0.85)"), (20, 40, "rgba(236,131,90,0.85)"), (40, 60, "rgba(250,178,25,0.75)"),
               (60, 80, "rgba(12,163,12,0.55)"), (80, 100, "rgba(12,163,12,0.95)")]


def gauge(value: float, title: str, subtitle: str, mode: str, ref: float | None = None) -> go.Figure:
    """Tacho 0–100 mit fünf Farbzonen (rot = sehr schwach … grün = sehr gesund)."""
    p = PALETTE[mode]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta" if ref is not None else "gauge+number",
        value=None if value is None or pd.isna(value) else round(float(value), 0),
        number=dict(font=dict(size=40, color=p["text"]), valueformat=".0f"),
        delta=dict(reference=ref, valueformat=".0f", increasing=dict(color=STATUS["ok"]),
                   decreasing=dict(color=STATUS["NA"])) if ref is not None and pd.notna(ref) else None,
        title=dict(text=f"<b>{title}</b><br><span style='font-size:12px;color:{p['text2']}'>{subtitle}</span>",
                   font=dict(size=16, color=p["text"])),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 20, 40, 60, 80, 100], tickcolor=p["muted"],
                             tickfont=dict(color=p["muted"], size=10)),
                   bar=dict(color=p["text"], thickness=0.22),
                   bgcolor=p["surface"], borderwidth=0,
                   steps=[dict(range=[a, b], color=c) for a, b, c in GAUGE_STEPS],
                   threshold=dict(line=dict(color=p["text"], width=3), thickness=0.8,
                                  value=None if value is None or pd.isna(value) else float(value)))))
    fig.update_layout(paper_bgcolor=p["surface"], height=250, margin=dict(l=24, r=24, t=70, b=10),
                      font=dict(family=FONT, color=p["text2"]))
    return fig


def _step_scale(steps) -> list:
    sc = []
    for a, b, col in steps:
        sc += [[a / 100, col], [b / 100, col]]
    return sc


REGIME_CLASSES = [(0, 40, STATUS["NA"], "schwach (< 40)"), (40, 60, STATUS["teilweise"], "neutral (40–60)"),
                  (60, 101, STATUS["ok"], "gesund (≥ 60)")]


def terminal_chart(d: pd.DataFrame, mode: str, label: str, trades: pd.DataFrame | None = None,
                   background: str | None = "lang", lines: dict | None = None,
                   bands: dict[str, str] | None = None, bg_label: str | None = None,
                   bg_names: tuple[str, str] | None = None) -> go.Figure:
    """Kurs (log) mit Regime-Hinterlegung eines Horizonts, darunter drei Score-Bänder (kurz/mittel/lang)
    und optional die Trades eines Regel-Backtests (Dreieck = Einstieg, grün = Gewinn, rot = Verlust)."""
    from plotly.subplots import make_subplots

    p = PALETTE[mode]
    names = {"kurz": "Kurzfristig", "mittel": "Mittelfristig", "lang": "Langfristig"}
    bands = bands or {names[h]: f"H_{h}" for h in ("kurz", "mittel", "lang")}
    nb = len(bands)
    bh = 0.30 / 3 if nb <= 3 else 0.42 / nb
    fig = make_subplots(rows=nb + 1, cols=1, shared_xaxes=True, vertical_spacing=0.012,
                        row_heights=[1 - bh * nb] + [bh] * nb)
    x = d.index
    fig.add_trace(go.Scatter(x=x, y=d["close"], name=label, mode="lines", line=dict(width=2, color=p["series"][0]),
                             hovertemplate="Kurs: %{y:,.0f}<extra></extra>"), row=1, col=1)
    for i, (col, nm) in enumerate((lines or {}).items()):
        fig.add_trace(go.Scatter(x=x, y=d[col], name=nm, mode="lines",
                                 line=dict(width=1.5, color=[p["text2"], p["series"][3]][i % 2], dash="dot"),
                                 hovertemplate=f"{nm}: %{{y:,.0f}}<extra></extra>"), row=1, col=1)
    # Regime-Hinterlegung (Farbe + Legendentext)
    bg_col = None
    if background:
        bg_col = background if background in d.columns else f"H_{background}"
    if bg_col:
        s = d[bg_col].values
        dates = x.values
        op = 0.16 if mode == "dark" else 0.13
        runs = []
        for lo, hi, colr, _ in REGIME_CLASSES:
            if colr == STATUS["teilweise"]:
                continue
            runs += [(a, b, colr) for a, b in _runs((s >= lo) & (s < hi), dates)]
        if len(runs) <= 600:
            for a, b, colr in runs:
                fig.add_vrect(x0=a, x1=b, fillcolor=colr, opacity=op, line_width=0, layer="below", row=1, col=1)
    # Score-Bänder
    scale = _step_scale([(0, 20, "#b8322f"), (20, 40, "#ec835a"), (40, 60, "#fab219"), (60, 80, "#3fb33f"),
                         (80, 100, "#0a8a0a")])
    for r, (nm, col) in enumerate(bands.items(), start=2):
        fig.add_trace(go.Heatmap(x=x, y=[nm], z=[d[col].values], zmin=0, zmax=100, colorscale=scale,
                                 showscale=False, name=nm,
                                 hovertemplate=f"{nm}: %{{z:.0f}}<extra></extra>"), row=r, col=1)
        fig.update_yaxes(showticklabels=True, tickfont=dict(size=11, color=p["text2"]), showgrid=False,
                         fixedrange=True, row=r, col=1)
    # Trades
    if trades is not None and not trades.empty:
        t = trades.copy()
        t["Ergebnis"] = np.where(t["Status"] == "offen", "offen", np.where(t["Rendite"] > 0, "Gewinn", "Verlust"))
        for res, colr, sym in (("Gewinn", STATUS["ok"], "triangle-up"), ("Verlust", STATUS["NA"], "triangle-up"),
                               ("offen", p["text"], "triangle-up-open")):
            g = t[t["Ergebnis"] == res]
            if g.empty:
                continue
            fig.add_trace(go.Scatter(
                x=g["Einstieg"], y=g["Kurs Einstieg"] * 0.97, mode="markers", name=f"Signal: {res}",
                marker=dict(symbol=sym, size=11, color=colr, line=dict(width=1, color=p["surface"])),
                customdata=np.c_[g["Rendite"].values, g["Max. Rückgang"].values, g["Kurz"].values, g["Lang"].values],
                hovertemplate=(f"<b>{res}</b><br>Rendite: %{{customdata[0]:+.1%}}<br>Max. Rückgang: "
                               "%{customdata[1]:.1%}<br>Kurz %{customdata[2]:.0f} · Lang %{customdata[3]:.0f}"
                               "<extra></extra>")), row=1, col=1)
            done = g.dropna(subset=["Ausstieg"])
            if len(done):
                fig.add_trace(go.Scatter(x=done["Ausstieg"], y=done["Kurs Ausstieg"], mode="markers",
                                         name=f"Ausstieg ({res})", showlegend=False,
                                         marker=dict(symbol="x-thin", size=9, line=dict(width=2, color=colr)),
                                         hovertemplate="Ausstieg: %{y:,.0f}<extra></extra>"), row=1, col=1)
    bg_name = bg_label or (names.get(background, background) if background else "")
    for i, (_, _, colr, nm) in enumerate(REGIME_CLASSES if bg_col else []):
        if colr != STATUS["teilweise"]:
            label = (bg_names[0] if i == 0 else bg_names[1]) if bg_names else f"{bg_name} {nm}"
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                     marker=dict(symbol="square", size=12, color=colr, opacity=0.5)), row=1, col=1)
    lo, hi = float(d["close"].min()), float(d["close"].max())
    cand = [m * 10 ** k for k in range(0, 7) for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8)]
    ticks = [v for v in cand if lo * 0.9 <= v <= hi * 1.1]
    while len(ticks) > 10:
        ticks = ticks[::2]
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"{int(v):,}".replace(",", "'") for v in ticks],
                     title_text="Kurs (log.)", row=1, col=1)
    fig = base_layout(fig, mode, "", height=700 if nb <= 3 else 700 + 40 * (nb - 3))
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.04, x=0),
                      margin=dict(t=60))
    fig.update_xaxes(showgrid=False, rangeslider=dict(visible=False))
    return fig


def cot_chart(v: pd.DataFrame, mode: str, label: str, lines: dict[str, str], lo: float, hi: float,
              net: dict[str, str] | None = None) -> go.Figure:
    """COT-Diagramm: Kurs mit Signal-Hintergrund (sig100: 100 Long, 0 Short), darunter Index-Linien 0–100 mit
    Schwellen und optional Netto-Positionen in Kontrakten."""
    from plotly.subplots import make_subplots

    p = PALETTE[mode]
    rows = 3 if net else 2
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=[0.48, 0.30, 0.22] if net else [0.6, 0.4])
    x = v.index
    fig.add_trace(go.Scatter(x=x, y=v["close"], name=label, mode="lines", line=dict(width=2, color=p["text"]),
                             hovertemplate="Kurs: %{y:,.2f}<extra></extra>"), row=1, col=1)
    op = 0.18 if mode == "dark" else 0.14
    dates = x.values
    for flag, colr in ((v["sig100"].values >= 99, STATUS["ok"]), (v["sig100"].values <= 1, STATUS["NA"])):
        for a, b in _runs(flag, dates):
            fig.add_vrect(x0=a, x1=b + np.timedelta64(7, "D"), fillcolor=colr, opacity=op, line_width=0, layer="below",
                          row=1, col=1)
    for nm, colr in (("Long-Signal", STATUS["ok"]), ("Short-Signal", STATUS["NA"])):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=nm,
                                 marker=dict(symbol="square", size=12, color=colr, opacity=0.6)), row=1, col=1)
    def colour(nm: str, i: int) -> str:   # gleiche Gruppe = gleiche Farbe in allen Diagrammen
        if "Kleine" in nm:
            return p["series"][6]
        if "Commercials" in nm:
            return p["series"][1]
        if "Spekulanten" in nm:
            return p["series"][2]
        return p["series"][0]

    for i, (nm, col) in enumerate(lines.items()):
        dash = "dot" if "26 W." in nm else ("dash" if "52 W." in nm else None)
        fig.add_trace(go.Scatter(x=x, y=v[col], name=nm, mode="lines",
                                 line=dict(width=3 if i == 0 else 1.8, color=colour(nm, i), dash=dash),
                                 hovertemplate=f"{nm}: %{{y:.0f}}<extra></extra>"), row=2, col=1)
    for y, txt, colr in ((hi, f"Long ab {hi:.0f}", STATUS["ok"]), (lo, f"Short bis {lo:.0f}", STATUS["NA"])):
        fig.add_hline(y=y, line=dict(color=colr, width=1.5, dash="dash"), row=2, col=1,
                      annotation_text=txt, annotation_position="top left",
                      annotation_font=dict(color=colr, size=11))
    fig.update_yaxes(range=[-3, 103], tickvals=[0, 20, 40, 60, 80, 100], title_text="Index 0–100", row=2, col=1)
    if net:
        for i, (nm, col) in enumerate(net.items()):
            fig.add_trace(go.Scatter(x=x, y=v[col], name=nm, mode="lines",
                                     line=dict(width=1.8, color=colour(nm, i)),
                                     hovertemplate=f"{nm}: %{{y:,.0f}}<extra></extra>"), row=3, col=1)
        fig.add_hline(y=0, line=dict(color=p["muted"], width=1), row=3, col=1)
        fig.update_yaxes(title_text="Netto (Kontrakte)", row=3, col=1)
    fig.update_yaxes(type="log", title_text="Kurs (log.)", row=1, col=1)
    fig = base_layout(fig, mode, "", height=820 if net else 680)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.06, x=0), margin=dict(t=70))
    fig.update_xaxes(showgrid=False)
    return fig
