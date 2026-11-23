"""Mit Hyblock Capital getestete BTC-Strategien (Stand 23.11.2026).

Diese Strategien brauchen Hyblock-Indikatoren (Tier 3, Professional) und lassen sich deshalb hier nicht live
berechnen – das Dashboard zeigt die Testergebnisse; live laufen sie als Alarm oder Strategie in Hyblock.
Backtest: BTC (Binance USDⓈ-M Perp), 2 Jahre, Einstieg Market, Gebühr 0,045 % pro Seite, TP/SL fest.
"Summe" = Summe der Trade-Ergebnisse in % (bei voller Positionsgrösse), nach Gebühren.
Jahr 1 = Nov. 2024 – Nov. 2025, Jahr 2 = Nov. 2025 – Nov. 2026 (Out-of-Sample-Kontrolle).
"""
from __future__ import annotations

import pandas as pd

STRATEGIES = [
    {"Name": "Imbalance-Short", "Richtung": "Short", "Zeiteinheit": "4 Std",
     "Regel": "Market Imbalance Index (Binance) unter dem 10. Perzentil der letzten 90 Tage",
     "TP / SL": "4 % / 4 %", "Trades": 104, "Treffer %": 59, "Summe %": 66.6,
     "Jahr 1": "56 % · +22.9 %", "Jahr 2": "62 % · +43.8 %", "Bewertung": "Beste Kandidatin – in beiden Jahren im Plus"},
    {"Name": "Imbalance-Short (1 Std)", "Richtung": "Short", "Zeiteinheit": "1 Std",
     "Regel": "Market Imbalance Index unter dem 10. Perzentil (90 Tage), nur 1 Jahr getestet",
     "TP / SL": "3 % / 3 %", "Trades": 107, "Treffer %": 58, "Summe %": 44.4,
     "Jahr 1": "–", "Jahr 2": "58 % · +44.4 %", "Bewertung": "Gewinn fast nur aus dem letzten Halbjahr (Bärenmarkt)"},
    {"Name": "Imbalance-Long mit Trend", "Richtung": "Long", "Zeiteinheit": "4 Std",
     "Regel": "Market Imbalance Index über dem 90. Perzentil UND Preis höher als vor 15 Tagen",
     "TP / SL": "4 % / 4 %", "Trades": 46, "Treffer %": 57, "Summe %": 19.9,
     "Jahr 1": "50 % · −2.3 %", "Jahr 2": "65 % · +22.2 %", "Bewertung": "Positiv, aber schwankend"},
    {"Name": "Imbalance-Long ohne Filter", "Richtung": "Long", "Zeiteinheit": "4 Std",
     "Regel": "Market Imbalance Index über dem 90. Perzentil",
     "TP / SL": "4 % / 4 %", "Trades": 109, "Treffer %": 51, "Summe %": 2.2,
     "Jahr 1": "56 % · +22.9 %", "Jahr 2": "46 % · −20.7 %", "Bewertung": "Nur im Bullenmarkt – ohne Trendfilter nicht nutzen"},
    {"Name": "Sentiment-Gap-Long", "Richtung": "Long", "Zeiteinheit": "4 Std",
     "Regel": "Trader Sentiment Gap unter dem 10. Perzentil (90 Tage)",
     "TP / SL": "4 % / 4 %", "Trades": 19, "Treffer %": 53, "Summe %": 2.3,
     "Jahr 1": "50 % · −0.7 %", "Jahr 2": "55 % · +3.0 %", "Bewertung": "Keine Kante"},
]

# Signal-Statistik (4 Std, 2 Jahre): Anteil Signale mit Kursanstieg nach 3 Tagen; Vergleich: alle Kerzen 49.6 %
SCAN = [
    ("Market Imbalance Index > 90. Perzentil", "Long", 453, 62.0, "+0.84 %"),
    ("Market Imbalance Index < 10. Perzentil", "Short", 480, 55.4, "−0.72 %"),
    ("Trader Sentiment Gap < 10. Perzentil", "Long", 189, 57.7, "+0.39 %"),
    ("Squeeze Intensity (Short) > 90. Perzentil", "Long", 474, 49.8, "+0.12 %"),
    ("Whale Position Dominance > 90. Perzentil", "Long", 387, 46.8, "+0.17 %"),
    ("Squeeze Intensity (Long) > 90. Perzentil", "Short", 518, 49.0, "−0.18 %"),
    ("Spot-CVD > 80. Perzentil + Preis −2 % in 24h", "Long", 84, 54.8, "−0.39 %"),
    ("Long-Liquidationen > 95. Perz. + Sentiment Gap < 30.", "Long", 28, 50.0, "−0.76 %"),
]


def strategies_frame() -> pd.DataFrame:
    return pd.DataFrame(STRATEGIES)


def scan_frame() -> pd.DataFrame:
    return pd.DataFrame(SCAN, columns=["Bedingung", "Richtung", "Signale", "Treffer nach 3 T %", "Ø 3 T"])
