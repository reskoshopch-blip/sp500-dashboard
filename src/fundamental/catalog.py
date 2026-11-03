"""Was das Fundamental-Terminal misst: Datenreihen, Veröffentlichungsverzug und Indikatoren.

Veröffentlichungsverzug (lag, Kalendertage): ein Wert gilt erst ab Datum + lag als bekannt. Beispiel: Die
US-Inflation für September (FRED-Datum 1. September) erscheint Mitte Oktober → lag 45. So nutzt jeder
historische Tag nur Daten, die damals schon veröffentlicht waren. Spätere Revisionen der Behörden (FRED zeigt
den heutigen Stand) werden nicht nachgebildet – bei Zinsen/Kursen ohne Belang, bei Konjunkturdaten klein.
"""
from __future__ import annotations

from dataclasses import dataclass

# id: (Beschreibung, Verzug in Tagen, maximale Gültigkeit in Tagen)
FRED: dict[str, tuple[str, int, int]] = {
    # USA Zinsen / Notenbank
    "DGS10": ("US-Rendite 10 Jahre", 1, 10), "DGS2": ("US-Rendite 2 Jahre", 1, 10),
    "DGS3MO": ("US-Rendite 3 Monate", 1, 10), "DFF": ("Fed Funds Rate (effektiv)", 1, 10),
    "DFII10": ("US-Realrendite 10 J. (TIPS)", 1, 10), "T10YIE": ("Inflationserwartung 10 J. (Breakeven)", 1, 10),
    # Liquidität / Kredit / Finanzierungsbedingungen
    "WALCL": ("Fed-Bilanzsumme", 2, 14), "WTREGEN": ("US-Treasury-Konto bei der Fed (TGA)", 2, 14),
    "RRPONTSYD": ("Reverse-Repo-Fazilität", 1, 10), "NFCI": ("Chicago Fed Financial Conditions", 6, 21),
    "BAA10Y": ("Kreditspread Baa vs. 10 J.", 1, 10), "BAMLH0A0HYM2": ("High-Yield-Spread (OAS)", 1, 10),
    "M2SL": ("Geldmenge M2", 30, 75),
    # Konjunktur USA
    "ICSA": ("Erstanträge Arbeitslosenhilfe", 5, 21), "UNRATE": ("Arbeitslosenquote", 36, 75),
    "PAYEMS": ("Beschäftigte (Payrolls)", 36, 75), "INDPRO": ("Industrieproduktion", 47, 80),
    "CPILFESL": ("Kerninflation (CPI ohne Energie/Nahrung)", 45, 80), "UMCSENT": ("Konsumentenstimmung Michigan", 30, 75),
    "PERMIT": ("Baugenehmigungen", 48, 80), "DCOILWTICO": ("Ölpreis WTI", 1, 10),
    "DCOILBRENTEU": ("Ölpreis Brent", 1, 10), "PCOPPUSDM": ("Kupferpreis (IMF, monatlich)", 20, 60),
    "DTWEXBGS": ("US-Dollar (breiter Index)", 1, 10),
    # Japan
    "DEXJPUS": ("USD/JPY", 1, 10), "IRLTLT01JPM156N": ("Japan Rendite 10 J. (JGB)", 40, 100),
    "IRSTCI01JPM156N": ("Japan Tagesgeldsatz (BoJ)", 40, 100), "JPNASSETS": ("BoJ-Bilanzsumme", 30, 75),
    "JPNPROINDMISMEI": ("Japan Industrieproduktion", 60, 110), "XTEXVA01JPM664S": ("Japan Exporte", 60, 110),
    "LRUN64TTJPM156S": ("Japan Arbeitslosenquote", 60, 110), "JPNCPIALLMINMEI": ("Japan Inflation", 60, 110),
    "JPNLOLITOAASTSAM": ("Japan OECD-Frühindikator", 75, 120), "USALOLITOAASTSAM": ("USA OECD-Frühindikator", 75, 120),
}

# COT: Futures je Markt (CFTC-Code). E-mini S&P 500 (CME) seit 1997, Nikkei 225 in USD (CME) seit 1990.
COT_CODES = {"SPY": ["13874A"], "N225": ["240741"]}
COT_LAG = 3   # Stichtag Dienstag, Veröffentlichung Freitag

CATEGORIES = ["Zinsen & Notenbank", "Liquidität & Kredit", "Konjunktur", "Rohstoffe", "Gewinne & Bewertung",
              "Währung", "Positionierung (COT)"]
# Rohstoffe täglich von Yahoo (Futures-Fortschreibung, ab 2000); davor Monatswerte von FRED/IMF
COMMODITY_TICKERS = {"copper": "HG=F", "gold": "GC=F", "oil": "CL=F"}


@dataclass(frozen=True)
class Ind:
    key: str
    label: str
    category: str
    markets: tuple[str, ...]
    prior: int          # +1: hohe Werte ökonomisch günstig, -1: ungünstig, 0: offen (Modell lernt Richtung)
    fmt: str            # pct | pp | num | x
    explain: str


B = ("SPY", "N225")
INDICATORS: tuple[Ind, ...] = (
    # Zinsen & Notenbank
    Ind("us10y_chg", "US-Rendite 10 J., Veränderung 6 M.", "Zinsen & Notenbank", B, -1, "pp",
        "Schnell steigende Langfristzinsen belasten Bewertungen"),
    Ind("us2y_chg", "US-Rendite 2 J., Veränderung 6 M.", "Zinsen & Notenbank", B, -1, "pp",
        "Die 2-jährige Rendite spiegelt den erwarteten Fed-Kurs – steigend = Straffung"),
    Ind("policy_exp", "Markt-Zinserwartung (2 J. minus Fed Funds)", "Zinsen & Notenbank", B, 0, "pp",
        "Negativ = der Markt preist Zinssenkungen ein, positiv = Zinserhöhungen"),
    Ind("curve_10y2y", "Zinskurve 10 J. – 2 J.", "Zinsen & Notenbank", B, 0, "pp",
        "Invertiert (negativ) gilt als Rezessionswarnung; die Wiederversteilung ist oft der gefährliche Moment"),
    Ind("curve_10y3m", "Zinskurve 10 J. – 3 M.", "Zinsen & Notenbank", B, 0, "pp", "Klassischer Rezessionsindikator"),
    Ind("real10", "US-Realrendite 10 J.", "Zinsen & Notenbank", B, -1, "pp",
        "Hohe Realzinsen machen Aktien relativ unattraktiv"),
    Ind("real10_chg", "US-Realrendite, Veränderung 6 M.", "Zinsen & Notenbank", B, -1, "pp", "Tempo der Realzinsänderung"),
    Ind("breakeven", "Inflationserwartung 10 J.", "Zinsen & Notenbank", B, 0, "pp", "Am Anleihemarkt eingepreiste Inflation"),
    Ind("jp10y_chg", "Japan-Rendite 10 J., Veränderung 6 M.", "Zinsen & Notenbank", ("N225",), -1, "pp",
        "Steigende JGB-Renditen = BoJ-Normalisierung, stärkerer Yen"),
    Ind("us_jp_spread", "Zinsdifferenz USA – Japan (10 J.)", "Zinsen & Notenbank", ("N225",), 1, "pp",
        "Grosse Differenz stützt einen schwachen Yen (gut für Exporteure im Nikkei)"),
    Ind("us_jp_spread_chg", "Zinsdifferenz USA – Japan, Veränderung 6 M.", "Zinsen & Notenbank", ("N225",), 1, "pp",
        "Schrumpfende Differenz → Yen-Aufwertung, Carry-Trade-Abbau"),
    Ind("jp_short_chg", "BoJ-Leitzins, Veränderung 12 M.", "Zinsen & Notenbank", ("N225",), -1, "pp",
        "Zinserhöhungen der Bank of Japan"),
    # Liquidität & Kredit
    Ind("net_liq_chg", "Netto-Liquidität Fed, Veränderung 3 M.", "Liquidität & Kredit", B, 1, "pct",
        "Fed-Bilanz minus Treasury-Konto minus Reverse Repo – mehr Liquidität stützt Risikoanlagen"),
    Ind("nfci", "Finanzierungsbedingungen (NFCI)", "Liquidität & Kredit", B, -1, "num",
        "Über 0 = straffer als im Durchschnitt"),
    Ind("credit_baa", "Kreditspread Baa", "Liquidität & Kredit", B, -1, "pp", "Risikoaufschlag für Unternehmensanleihen"),
    Ind("credit_baa_chg", "Kreditspread Baa, Veränderung 3 M.", "Liquidität & Kredit", B, -1, "pp",
        "Steigende Spreads = Stress im Kreditmarkt, oft vor Aktien"),
    Ind("hy_spread", "High-Yield-Spread", "Liquidität & Kredit", B, -1, "pp", "Risikoaufschlag für Hochzinsanleihen"),
    Ind("m2_yoy", "Geldmenge M2, Wachstum 12 M.", "Liquidität & Kredit", B, 1, "pct", "Geldmengenwachstum"),
    Ind("boj_assets_yoy", "BoJ-Bilanz, Wachstum 12 M.", "Liquidität & Kredit", ("N225",), 1, "pct",
        "Anleihekäufe der Bank of Japan"),
    # Konjunktur
    Ind("claims_chg", "Erstanträge (4-Wochen-Schnitt), Veränderung 6 M.", "Konjunktur", B, -1, "pct",
        "Steigende Arbeitslosenanträge sind ein früher Rezessionshinweis"),
    Ind("sahm", "Arbeitslosenquote über 12-Monats-Tief", "Konjunktur", B, -1, "pp",
        "Sahm-Regel: Anstieg ≥ 0.5 Pp. ging jeder US-Rezession voraus"),
    Ind("payrolls_3m", "Beschäftigung, Wachstum 3 M. (annualisiert)", "Konjunktur", B, 1, "pct", "Jobwachstum"),
    Ind("indpro_yoy", "Industrieproduktion USA, 12 M.", "Konjunktur", B, 1, "pct", "Industriezyklus"),
    Ind("core_cpi_3m", "Kerninflation, 3 M. annualisiert", "Konjunktur", B, -1, "pct",
        "Hohe/steigende Inflation zwingt die Notenbank zur Straffung"),
    Ind("sentiment", "Konsumentenstimmung", "Konjunktur", B, 0, "num",
        "Extrem tiefe Stimmung war oft eher eine Kaufgelegenheit (Kontraindikator)"),
    Ind("permits_yoy", "Baugenehmigungen, 12 M.", "Konjunktur", B, 1, "pct", "Frühzyklischer Immobilienindikator"),
    Ind("us_cli", "USA OECD-Frühindikator, Veränderung 6 M.", "Konjunktur", B, 1, "num", "Richtung des Konjunkturzyklus"),
    Ind("jp_indpro_yoy", "Japan Industrieproduktion, 12 M.", "Konjunktur", ("N225",), 1, "pct", "Japans Industriezyklus"),
    Ind("jp_exports_yoy", "Japan Exporte, 12 M.", "Konjunktur", ("N225",), 1, "pct", "Exportnachfrage"),
    Ind("jp_cli", "Japan OECD-Frühindikator, Veränderung 6 M.", "Konjunktur", ("N225",), 1, "num",
        "Richtung des japanischen Zyklus"),
    Ind("jp_cpi_yoy", "Japan Inflation, 12 M.", "Konjunktur", ("N225",), 0, "pct",
        "Inflation in Japan – Grundlage für BoJ-Zinserhöhungen"),
    # Rohstoffe
    Ind("oil_yoy", "Ölpreis, 12 M.", "Rohstoffe", B, 0, "pct",
        "Ölschock (belastet, v. a. Importland Japan) vs. Nachfragestärke"),
    Ind("oil_3m", "Ölpreis, 3 M.", "Rohstoffe", B, 0, "pct", "Kurzfristiger Ölpreisschub"),
    Ind("copper_yoy", "Kupferpreis, 12 M.", "Rohstoffe", B, 1, "pct",
        "„Dr. Copper“: steigender Kupferpreis = robuste Industrienachfrage (China, Bau, Elektrifizierung)"),
    Ind("copper_3m", "Kupferpreis, 3 M.", "Rohstoffe", B, 1, "pct", "Kurzfristige Industrienachfrage"),
    Ind("copper_gold", "Kupfer/Gold, Veränderung 6 M.", "Rohstoffe", B, 1, "pct",
        "Wachstum (Kupfer) gegen Absicherung (Gold) – steigend = Risikoappetit"),
    Ind("gold_yoy", "Goldpreis, 12 M.", "Rohstoffe", B, 0, "pct",
        "Starker Gold-Anstieg = Flucht in Sicherheit oder Geldentwertung"),
    # Gewinne & Bewertung (S&P 500 aus SEC-Daten; für Japan als globaler Gewinnzyklus)
    Ind("ni_growth", "S&P-500-Gewinnwachstum (TTM, like-for-like)", "Gewinne & Bewertung", B, 1, "pct",
        "Gewinne der letzten 12 Monate gegenüber Vorjahr, nur Firmen mit beiden Werten (SEC-Daten)"),
    Ind("ni_growth_chg", "Gewinnwachstum beschleunigt/verlangsamt (3 M.)", "Gewinne & Bewertung", B, 1, "pp",
        "Zweite Ableitung: Märkte reagieren oft auf die Veränderung des Wachstums"),
    Ind("rev_growth", "S&P-500-Umsatzwachstum (TTM)", "Gewinne & Bewertung", B, 1, "pct", "Umsatzwachstum like-for-like"),
    Ind("margin_chg", "Gewinnmarge, Veränderung 12 M.", "Gewinne & Bewertung", B, 1, "pp", "Steigende Margen"),
    Ind("pe", "KGV S&P 500 (Näherung)", "Gewinne & Bewertung", ("SPY",), -1, "x",
        "Hohe Bewertung = viel Optimismus eingepreist, tiefere langfristige Renditen"),
    Ind("erp", "Gewinnrendite minus Realrendite", "Gewinne & Bewertung", ("SPY",), 1, "pp",
        "Risikoprämie: Wie viel mehr bringen Aktien als inflationsgeschützte Anleihen?"),
    # Währung
    Ind("usdjpy_chg", "USD/JPY, Veränderung 3 M.", "Währung", ("N225",), 1, "pct",
        "Schwacher Yen (steigender USD/JPY) hebt die Gewinne der Exporteure"),
    Ind("usdjpy_vs_1y", "USD/JPY vs. 12-Monats-Schnitt", "Währung", ("N225",), 1, "pct", "Yen-Trend"),
    Ind("dollar_chg", "US-Dollar (breit), Veränderung 3 M.", "Währung", B, -1, "pct",
        "Starker Dollar belastet US-Auslandsgewinne und globale Liquidität"),
    # Positionierung
    Ind("cot_nc", "COT Spekulanten netto (% Open Interest)", "Positionierung (COT)", B, 0, "pct",
        "Netto-Position der grossen Spekulanten (Non-Commercials)"),
    Ind("cot_index", "COT-Index Spekulanten (3 Jahre)", "Positionierung (COT)", B, 0, "num",
        "0 = so short wie seit 3 Jahren nicht, 100 = so long wie nie – Extreme gelten als überfüllt"),
    Ind("cot_comm_index", "COT-Index Commercials (3 Jahre)", "Positionierung (COT)", B, 0, "num",
        "Positionierung der Absicherer – oft Gegenposition zu den Spekulanten"),
    Ind("cot_chg", "COT Spekulanten, Veränderung 8 Wochen", "Positionierung (COT)", B, 0, "pp",
        "Bauen Spekulanten gerade Positionen auf oder ab?"),
)
IND_BY_KEY = {i.key: i for i in INDICATORS}


def for_market(key: str) -> list[Ind]:
    return [i for i in INDICATORS if key in i.markets]
