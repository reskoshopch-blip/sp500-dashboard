### Datenquellen

| Quelle | Verwendung | Endpunkt |
|---|---|---|
| SEC EDGAR XBRL Company Facts | Alle Bilanzwerte | `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` |
| SEC XBRL Company Concept | Unabhängige Stichproben-Verifikation | `https://data.sec.gov/api/xbrl/companyconcept/CIK##########/us-gaap/<Tag>.json` |
| SEC Filing-Archiv | Links zu den Originalberichten (10-K) | `https://www.sec.gov/Archives/edgar/data/<CIK>/<Accession>/` |
| SEC Ticker-Datei | Abgleich Ticker ↔ CIK | `https://www.sec.gov/files/company_tickers.json` |
| Wikipedia „List of S&P 500 companies“ | Mitgliederliste, GICS-Sektor, Änderungshistorie | Tabellen `constituents` und `changes` |

Die offizielle Mitgliederliste von S&P Dow Jones Indices ist lizenzpflichtig. Die Wikipedia-Liste ist öffentlich,
verweist auf die Pressemitteilungen von S&P DJI und wird gegen die SEC-Ticker-Datei geprüft (Abweichungen werden
pro Unternehmen vermerkt). Fallback: eigene CSV unter `data/sp500_constituents.csv` bzw. der letzte erfolgreiche Snapshot.
Alle SEC-Abrufe verwenden den konfigurierten User-Agent, höchstens 10 (Standard 5) Anfragen pro Sekunde, Retries mit
Backoff bei 429/5xx und einen komprimierten lokalen Cache mit `If-Modified-Since`.

### Normalisierung

1. **Nur Instant-Facts.** Bilanzwerte sind Stichtagswerte (`end` ohne `start`). Duration-Facts (GuV, Cashflow) werden
   verworfen. Dadurch können Quartalswerte auch nicht zu Jahreswerten aufsummiert werden.
2. **Stichtag eines Jahresberichts** = `end` der Assets-Facts dieses Berichts (Fallback: LiabilitiesAndStockholdersEquity,
   StockholdersEquity). Formulare: 10-K, 10-K/A, 10-KT, 20-F, 40-F.
3. **Geschäftsjahr** = `fy` (DocumentFiscalYearFocus) des Jahresberichts mit diesem Stichtag. Unplausible Werte werden
   durch die Regel „Stichtag im Jan./Feb. → Vorjahr, sonst Kalenderjahr“ ersetzt und protokolliert. Damit werden
   Geschäftsjahre konsistent verglichen (z. B. Einzelhändler mit Jahresende Anfang Februar = Geschäftsjahr des Vorjahres),
   nicht Kalenderjahre. Fehlt für ein Vorjahr ein eigener Bericht (z. B. Jahr vor dem Börsengang), wird der Stichtag aus
   der Vergleichsspalte des Folgeberichts übernommen und so gekennzeichnet.
4. **Restatements und Mehrfachmeldungen.** Derselbe Stichtag erscheint in mehreren Berichten. Verwendet wird der zuletzt
   eingereichte Wert; Erstmeldung, Anzahl Meldungen und ein Restatement-Flag (Differenz > 0,50 USD) werden gespeichert.
5. **Quartalsmodus.** Der neueste Bilanzstichtag aus 10-Q/10-K wird separat geführt und nie mit Geschäftsjahren vermischt.
6. **Einheiten.** Nur Währungseinheiten. Die SEC liefert Werte in ganzen Einheiten (keine Tausender), daher ist der
   normalisierte Wert gleich dem Originalwert. Es findet keine Währungsumrechnung statt; Nicht-USD-Werte werden
   gekennzeichnet und nicht aggregiert.
7. **Fehlende Werte** bleiben fehlend und werden als „Nicht verfügbar“/NA angezeigt. 0 ist immer ein gemeldeter Nullwert.

### Positionen

Pro Standardposition gibt es eine geordnete Liste gleichbedeutender XBRL-Tags (Tabelle unten). Tags mit abweichender
Definition erhalten eine eigene Position, damit keine Definitionen vermischt werden. Zusätzlich werden **alle** übrigen
Instant-Facts in Währungseinheiten aus Jahresberichten dynamisch erfasst („Weitere XBRL-Positionen“ im Unternehmensprofil).
Diese enthalten neben weiteren Bilanzzeilen auch Stichtagsangaben aus dem Anhang, weil die Company-Facts-API keine
Zuordnung zur Bilanz-Präsentation liefert.

**Berechnete Summen** (gekennzeichnet mit ᴮ und Formel):
* `equity_total = equity_parent + nci`; ohne gemeldete Minderheiten `= equity_parent`
* `total_liabilities = liabilities_and_equity − equity_total − temporary_equity` (Viele Unternehmen melden den Tag
  `Liabilities` nicht.)

Es werden nie Gesamtpositionen und ihre Unterpositionen addiert. Aggregationen summieren immer nur eine Position.

### Aggregation

* **Summe der verfügbaren Werte** – mit Anzahl meldender Unternehmen und Liste der fehlenden.
* **Konsistentes Panel** – nur Unternehmen, die alle gewählten Positionen in allen gewählten Jahren melden; nur so sind
  Vermögen, Verbindlichkeiten und Eigenkapital bzw. verschiedene Jahre direkt vergleichbar.
* **Vorjahresveränderungen** immer like-for-like (nur Unternehmen mit Wert in beiden Jahren).

### Indexzusammensetzung

Standard: die aktuelle Zusammensetzung wird rückwirkend auf alle Jahre angewendet (Survivorship-Bias: inzwischen
entfernte Unternehmen fehlen). Optional („Nur Jahre mit Indexmitgliedschaft“) wird die Mitgliedschaft zum
Geschäftsjahresende aus der Wikipedia-Änderungshistorie rekonstruiert; ehemalige Mitglieder können damit erkannt,
aber nicht geladen werden, da ihre CIK nicht in der aktuellen Liste steht.

### Bekannte Einschränkungen

* Unternehmensspezifische XBRL-Erweiterungen (eigene Namespaces) sind in der Company-Facts-API nicht enthalten.
* Banken und Versicherungen haben keine kurz-/langfristige Gliederung; Current Ratio, Working Capital usw. sind dort NA.
* Die Finanzschulden-Definition schließt Leasingverbindlichkeiten aus. Fehlt der kurzfristige Teil, ist die Kennzahl
  „teilweise“; Nettoverschuldung wird dann nicht berechnet.
* Die Geschäftsjahr-Angabe `fy` der SEC ist in Einzelfällen fehlerhaft; die Plausibilitätsregel fängt grobe Fehler ab.
* Aktiengattungen (z. B. GOOGL/GOOG) werden zu einem Unternehmen (eine CIK) zusammengefasst; daher ~500 statt 503 Zeilen.
* Für das Geschäftsjahr 2025 fehlen Werte von Unternehmen, deren 10-K noch nicht eingereicht ist.

### Machine Learning

**Ziel.** Aus der Bilanzentwicklung eines Unternehmens (und dem Marktregime) schätzen, was in den 12 Monaten nach
Veröffentlichung des Jahresberichts passiert: (1) schlägt die Aktie den S&P 500 (SPY, inkl. Dividenden)? (2) fällt sie
zwischenzeitlich um mehr als 30 %?

**Kursdaten.** Yahoo Finance über `yfinance`, täglich, bereinigt um Splits und Dividenden. TradingView bietet keine freie
Schnittstelle für Massendownloads. Ticker ohne Kurse werden protokolliert und ausgelassen.

**Point-in-time (kein Blick in die Zukunft).**
* Bilanzwerte **wie erstmals gemeldet** – spätere Restatements werden nicht verwendet.
* Prognosezeitpunkt t0 = erster Handelstag **nach dem Einreichungsdatum** des 10-K, nicht das Geschäftsjahresende.
* Kursmerkmale nur bis t0, Ergebnis nur aus Kursen nach t0.

**Merkmale.** ~30 Bilanzmerkmale (Quoten wie Eigenkapitalquote, Verschuldung, Liquidität, Goodwill-Anteil;
Wachstum von Bilanzsumme, Eigenkapital, Schulden, Cash, Forderungen, Vorräten; Veränderungen der Quoten; eigene Aktien
als Rückkauf-Indikator), Branche, Marktregime bei t0 (SPY über/unter 200-Tage-Durchschnitt, 12-Monats-Rendite,
Volatilität). Optional zusätzlich Kursmomentum und Volatilität der Aktie. Fehlende Werte bleiben fehlend; das Modell
(Gradient Boosting) lernt „nicht gemeldet“ als eigene Information.

**Marktregime.** Bull = SPY schliesst über seinem 200-Tage-Durchschnitt, sonst Bear. Das Regime ist ein Merkmal; die
Modelle können daher lernen, dass dieselbe Bilanzentwicklung in Bull- und Bear-Phasen unterschiedlich wirkt.

**Historischer Test (Walk-forward mit Purging).** Für jedes Testjahr ab 2014 wird ein neues Modell trainiert – nur mit
Beobachtungen, deren 12-Monats-Ergebnis vor dem 1. Januar des Testjahres feststand. Gemessen werden AUC (0,5 = Zufall),
Rangkorrelation (IC), Trefferquoten und Überrenditen der Quintile mit höchster/tiefster Prognose samt t-Wert über die
Testjahre, Kalibrierung, Ergebnisse getrennt nach Regime und Permutation Importance.

**Prognosen.** Das Endmodell wird auf allen abgeschlossenen Beobachtungen trainiert und auf die neueste Bilanz jedes
Unternehmens angewendet, deren 12-Monats-Ergebnis noch offen ist. Neben der Modellwahrscheinlichkeit wird gezeigt, was
im Backtest bei ähnlichen Prognosen tatsächlich passierte. Treiber: Veränderung der Prognose, wenn ein Merkmal auf den
Median gesetzt wird. Szenario: Merkmale ändern und die neue Prognose sehen.

**Wirkung von Bilanzentwicklungen.** Modellfrei (Quintile je Jahr, tatsächliche Ergebnisse, getrennt nach Regime) und
aus Modellsicht (Partial Dependence je Regime).

**Grenzen.** Survivorship-Bias (nur heutige Indexmitglieder; Pleiten fehlen), keine Transaktionskosten, rund 11
Testjahre, Zusammenhänge statt Ursachen. Keine Anlageberatung.

### Top-Warnung (Gesamtmarkt)

Ziel: erhöhtes Risiko eines Rückgangs des S&P 500 (SPY) um mindestens 10 % in den nächsten 63 Handelstagen erkennen –
vor dem Rückgang, nicht danach wie die 200-Tage-Regel.

* **Daten:** SPY, Kurse aller Mitglieder (Marktbreite), VIX (`^VIX`), HYG und IEF (Kreditmarkt) von Yahoo Finance.
* **Ampel (Regeln):** Breiten-Divergenz, VIX steigt aus Tief, Kreditmarkt schwächelt, Überdehnung. Gelber Streifen ab 2 Zeichen.
* **ML-Top-Risiko:** logistische Regression (regularisiert) auf 16 Marktmerkmalen; Walk-forward ab 2014, Training nur
  auf Tagen, deren 63-Tage-Ergebnis vor dem Testjahr feststand, wöchentliche Stichprobe gegen Autokorrelation.
  Anzeige als 5-Tage-Durchschnitt; Warnung, wenn das Risiko ein wählbares Vielfaches (Standard 2×) des Durchschnitts übersteigt.
* **Auswertung:** erkannte Einbrüche ≥ 10 % (Warnung bis 3 Monate vor dem Hoch bzw. bevor −10 % erreicht waren),
  Vorlauf, Trefferquote der Warntage, Fehlalarm-Phasen; zusätzlich „Rot = Boden?“: Renditen nach Beginn jeder Bear-Phase.
* **Grenzen:** wenige Ereignisse seit 2010, Marktbreite nur aus heutigen Mitgliedern, kein Timing auf den Tag.

**Märkte:** S&P 500 (SPY, ab 1993, Test ab 2001), JP225 / Nikkei 225 (^N225, ab 1965, Test ab 1990, Kursindex
ohne Dividenden wie der CFD) und Japan-ETF EWJ (USD, ab 1996, Test ab 2005). Japan: Marktbreite aus den 225
Nikkei-Mitgliedern (Liste von Nikkei Inc. bzw. Wikipedia, Fallback `data/nikkei225_constituents.csv`), Nikkei
Volatility Index falls auf Yahoo verfügbar (sonst realisierte Volatilität), USD/JPY als zusätzliches Warnzeichen
(Yen-Stärke: USD/JPY −3 % in einem Monat), US-Kreditmarkt als globaler Risikoappetit.
Einbrüche werden nach dem Zickzack-Prinzip gezählt (−10 % vom Hoch, Ende bei +10 % vom Tief), damit auch Einbrüche
in langen Bärenmärkten wie Japan 1990–2012 einzeln erscheinen.

**Technische Bestätigung und Simulation:** Ausstieg, wenn innerhalb der letzten N Tage die Ampel aktiv war und der
Kurs unter den gewählten Durchschnitt schliesst; Wiedereinstieg bei Schluss darüber. Vergleich mit „nur Technik“,
„nur 200-Tage-Regel“ und Kaufen und Halten (Rendite p. a., maximaler Rückgang, Volatilität, Zeit investiert, Wechsel).

### Markt-Terminal (Gesundheitsindex)

Drei Scores (0–100) für kurz-, mittel- und langfristig plus Gesamt. Jeder Indikator wird mit seinen eigenen Werten
der letzten 5 Jahre verglichen (rollierendes Perzentil, nur Vergangenheit); bei „hoch = schlecht“ (VIX, Divergenzen,
Anteil Aktien unter Hoch) umgedreht. Der Score eines Horizonts ist der Durchschnitt seiner verfügbaren Indikatoren.

* **Kurzfristig:** Index vs. 20-Tage-Schnitt, 10-Tage-Rendite, Anteil Aktien über 20-Tage-Schnitt, McClellan-Oszillator,
  Tages-Divergenzen (Index −0.5 %, Mehrheit der Aktien im Plus), Volatilität vs. 20-Tage-Schnitt, VIX/VIX3M.
* **Mittelfristig:** Index vs. 50-Tage-Schnitt, 3-Monats-Rendite, Breite 50 Tage, Advance-Decline-Linie 3 Monate,
  neue Hochs minus Tiefs, RSP/SPY (gleichgewichtet vs. kapitalgewichtet), IWM/SPY, XLY/XLP, HYG/IEF, (Japan: USD/JPY).
* **Langfristig:** Index vs. 200-Tage-Schnitt, 12-Monats-Rendite, Breite 200 Tage, Anteil Aktien ≥ 20 % unter
  52-Wochen-Hoch, Abstand zum Hoch, HYG/IEF 12 Monate, Zinskurve 10 J. − 3 M., Kupfer/Gold 6 Monate.
* **Test:** Ø Rendite, Anteil positiv und Häufigkeit eines Rückgangs ≥ 10 % in den folgenden 10 / 63 / 252
  Handelstagen je Score-Bereich, plus Rangkorrelation Score ↔ Folgerendite. Überlappende Zeiträume.
* **Neue Ampel-Zeichen:** Schwergewichte tragen den Index (RSP/SPY −3 % in 3 Monaten bei Index nahe Hoch),
  Tages-Divergenzen (≥ 2 in 20 Tagen bei Index nahe Hoch), viele Aktien ≥ 20 % unter Hoch (≥ 35 % bei Index < 5 % unter Hoch).
  Mit 7 Zeichen ist die Standard-Schwelle 3.

### Fundamental-Terminal

**Quellen (gratis, ohne Schlüssel):** FRED (`fredgraph.csv`) für rund 30 Reihen – US-Zinsen (3 M., 2 J., 10 J., Fed
Funds, TIPS-Realrendite, Breakeven), Fed-Bilanz, Treasury-Konto, Reverse Repo, NFCI, Baa- und High-Yield-Spread, M2,
Erstanträge, Arbeitslosenquote, Payrolls, Industrieproduktion, Kerninflation, Konsumentenstimmung, Baugenehmigungen,
Öl (WTI, Brent), Kupfer (IMF, monatlich), Dollar-Index, USD/JPY, Japan (JGB 10 J., Tagesgeld, BoJ-Bilanz, Industrieproduktion, Exporte, Arbeitslosigkeit,
Inflation, OECD-Frühindikator). CFTC Commitments of Traders (Legacy, Futures only): E-mini S&P 500 (13874A),
Nikkei 225 USD (240741). SEC Company Facts (bereits geladen) für Gewinne und Umsätze. Rohstoffe täglich von Yahoo (HG=F Kupfer, GC=F Gold, CL=F Öl, Futures-Fortschreibung ab 2000); davor Öl und Kupfer von FRED. Indikatoren: Öl 3/12 M., Kupfer 3/12 M., Kupfer/Gold 6 M., Gold 12 M.

**Point-in-time:** Jede Reihe gilt erst nach ihrem typischen Veröffentlichungsverzug (Katalog in
`src/fundamental/catalog.py`, z. B. Kerninflation 45 Tage, Erstanträge 5 Tage, COT 3 Tage) und nur so lange, wie sie
aktuell ist. FRED liefert den heutigen (revidierten) Datenstand – Revisionen werden nicht nachgebildet.

**S&P-500-Gewinne:** Quartale aus 10-Q/10-K (zuerst gemeldeter Wert), 4. Quartal = Geschäftsjahr minus 9 Monate, TTM =
4 aufeinanderfolgende Quartale, bekannt ab dem letzten Einreichungsdatum. Wachstum like-for-like (nur Firmen mit beiden
Werten), monatlich zum Monatsende. GAAP-Gewinne (inkl. Sondereffekte). KGV-Näherung: S&P-500-Kursindex / verketteter
Gewinnindex, skaliert auf das heutige Gesamt-KGV (Börsenwert / TTM-Gewinn der Firmen mit einer Aktienklasse).

**Lernender Score:** Perzentil je Indikator (10 Jahre, nur Vergangenheit). Jährlich am 1. Januar: Rangkorrelation
zwischen Perzentil und Rendite der folgenden 126 Handelstage, nur Zeiträume, die vor dem Lerntag abgeschlossen waren
(wöchentliche Stichproben, effektive Stichprobe = n / (126/5)). |t| ≥ 1 → gelernte Richtung, Gewicht = min(|t|, 3);
sonst ökonomische Grundrichtung mit Gewicht 0.5 oder kein Einfluss. Bereichs- und Gesamtscore = gewichteter Durchschnitt.

**ML-Prognose:** HistGradientBoosting (Tiefe 3) auf Perzentilen, Bereichsscores, deren 3-Monats-Veränderung und
Kursmomentum; Ziel: Rendite > 0 nach 63 bzw. 252 Handelstagen. Jährlich neu trainiert mit Purging, Prognose für jeden
Tag des Folgejahres, Auswertung auf Wochenstichproben (AUC, Trefferquote, Kalibrierung). Die heutige Wahrscheinlichkeit
wird per Platt-Skalierung an den Out-of-sample-Prognosen kalibriert (ohne Vorhersagekraft → Basisrate).

**Eingepreist:** Perzentil der 6-Monats-Rendite (5 Jahre) minus Fundamental-Score; > +25 = Kurs vorausgelaufen,
< −25 = Fundamentaldaten besser als Kurs. Historischer Test je Lage. Zusätzlich direkt eingepreiste Grössen: 2 J. minus
Fed Funds (Zinspfad), Breakeven, Risikoprämie, High-Yield-Spread, COT-Index.

**Grenzen:** Fundamentaldaten wirken über Monate, nicht Tage. Für Japan gibt es keine frei verfügbaren Indexgewinne;
der S&P-Gewinnzyklus dient als globaler Zyklus. Wenige unabhängige Zyklen (Rezessionen) → Ergebnisse mit Vorsicht.

### COT-Signale

**Daten:** CFTC Commitments of Traders, Legacy, nur Futures – Gold COMEX (088691), Rohöl WTI NYMEX (067651), Kupfer
COMEX (085692), E-mini S&P 500 (13874A), E-mini Nasdaq 100 (209742). Die in Wirtschaftskalendern (z. B. Tradays)
gezeigten „speculative net positions“ sind die Non-Commercials netto aus genau diesem Bericht. Kurse: Yahoo GC=F,
CL=F, HG=F (Futures-Fortschreibung), ^GSPC, ^NDX; Öl vor 2000 von FRED (DCOILWTICO).

**Zeitpunkte:** Stichtag Dienstag, Veröffentlichung Freitag; gehandelt wird zum Schlusskurs des folgenden Montags
(Stichtag + 6 Tage).

**Merkmale:** Netto in % des Open Interest (Spekulanten, Commercials, kleine Spekulanten), COT-Indizes 26/52/156 Wochen
(Min-Max-Lage, 0–100), 4-Wochen-Veränderungen, Open Interest 13 Wochen. Alle als Perzentil (156 Wochen, mind. 104).

**Lernen:** Jährlich am 1. Januar, Rangkorrelation mit der Rendite der folgenden 13 Wochen, nur Wochen, deren Ergebnis
vor dem Lerntag feststand; t = IC · √(n/13). |t| ≥ 1 → gelernte Richtung, Gewicht min(|t|, 3); sonst Grundrichtung
(Commercials hoch = bullisch, Spekulanten hoch = bärisch) mit Gewicht 0.5. COT-Score = gewichteter Durchschnitt.

**Signal & Test:** Long ≥ obere Schwelle, Short ≤ untere Schwelle (Standard 60/40; klassisch Commercials-Index 3 J.
80/20). Wöchentliche Position, gehalten bis zum Signalwechsel, 0.05 % Kosten je Wechsel, keine Roll- oder
Finanzierungskosten. Test erst ab dem Jahr nach den ersten gelernten Gewichten.

### Gamma Exposure (GEX)

**Quelle:** AnalyticsFX-API `GET /api/v1/gex/{symbol}` (2 Credits; Strikes, Call-/Put-/Netto-GEX je Strike,
Gamma-Flip, Max Gamma, Verfälle) und `GET /api/v1/zeta/{asset}` (1 Credit; Regime, Drift, Risiko für SPY/QQQ).
Schlüssel nur in `.env` (`ANALYTICSFX_API_KEY`). Die API liefert keine Historie – jede Antwort wird roh gespeichert
(`data/gex/raw`) und als Merkmalszeile in `data/gex/snapshots.parquet`.

**Merkmale:** Regime (Spot über/unter Flip; ohne Flip-Angabe: Nullstelle der kumulierten Netto-GEX), Abstände zu Flip,
Call-Wall (max. Call-GEX), Put-Wall (min. Put-GEX) und Max Gamma, Netto-GEX-Anteil, Konzentration ±2 %, Lage zwischen
den Walls, Zeta, normale Tagesschwankung (20 Tage), 5-Tage-Trend.

**Ziele:** nächster Börsenschluss nach dem Abruf (Yahoo; Schluss gilt ab 21:00 UTC): |Rendite| / normale Schwankung,
Richtung, Schluss zwischen den Walls. Je Symbol und US-Handelstag zählt der letzte Abruf.

**Lernen:** Regelmodell (positive Gamma 0.8×, negative 1.25× normale Bewegung) und ab 30 abgeschlossenen Tagen
(alle Symbole zusammen) Ridge- bzw. logistische Regression, für jeden Tag neu trainiert nur mit Tagen, deren Ergebnis
vor dem Abruf feststand. Bewertung nur aus diesen echten Vorhersagen (Rangkorrelation, AUC, Trefferquote) und immer im
Vergleich zum Regelmodell.

### Modell-Konsens S&P 500

Tägliche Tabelle aus den gespeicherten Ergebnissen: Markt-Terminal (H_kurz, H_mittel, H_lang), Fundamental gesamt,
Fundamental-ML (Walk-forward-Wahrscheinlichkeit 3 Monate × 100), COT-Score E-mini S&P 500 (ab Montag nach
Veröffentlichung, max. 10 Tage gültig), optional Top-Warnung (100 − 25 × Ampel-Warnzeichen). Bullisch = Score ≥
Schwelle. Trefferquote = Anteil positiver SPY-Renditen über den Horizont; Trades nicht überlappend mit 0.05 % Kosten je
Seite. Gelernte Kombination: logistische Regression auf allen Scores, jährlich neu, Training nur mit Zeiträumen, die vor
dem 1. Januar abgeschlossen waren (Wochenstichproben).
