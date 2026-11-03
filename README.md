# S&P 500 Balance Sheet Intelligence Dashboard

Interaktives Dashboard mit historischen Bilanzdaten (Geschäftsjahre 2021–2025) aller S&P-500-Unternehmen,
direkt aus den XBRL-Daten der SEC (EDGAR). Keine erfundenen oder geschätzten Werte: jeder Wert trägt XBRL-Tag,
Formular, Einreichungsdatum und einen Link zum Originalbericht; fehlende Werte bleiben fehlend.

## Schnellstart (Windows)

Voraussetzung: **Python 3.10–3.13** (empfohlen 3.12) von https://www.python.org – bei der Installation
„Add python.exe to PATH“ anhaken.

1. Projektordner entpacken.
2. `start.bat` doppelklicken.
   * Beim ersten Start wird `.env` angelegt und in Notepad geöffnet. Trage bei `SEC_USER_AGENT` deinen Namen und
     deine E-Mail-Adresse ein (Pflicht laut SEC), speichern, Notepad schließen.
   * Das Skript erstellt eine virtuelle Umgebung, installiert die Pakete, startet den Datenabruf in einem
     minimierten Fenster und öffnet das Dashboard unter http://localhost:8501.
3. Der erste vollständige Abruf dauert je nach Verbindung ca. 15–40 Minuten (≈ 500 Dateien). Den Fortschritt
   zeigt die Seite **Daten laden**; sobald er fertig ist, Seite neu laden (F5).

Weitere Starts sind schnell: Unternehmen, deren Daten jünger als `MAX_CACHE_AGE_DAYS` sind, werden übersprungen.
`start.bat nodata` startet nur das Dashboard ohne Abruf.

### Manuell / andere Systeme

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # SEC_USER_AGENT eintragen
python -m src.pipeline --limit 20      # Testlauf mit 20 Unternehmen
python -m src.pipeline                 # alle Unternehmen (inkrementell, wiederaufnehmbar)
streamlit run app.py
```

Docker: `docker compose up --build` (liest `.env`, speichert Daten in `./data`).

## Auf iPhone/iPad/Mac: Cloud-Ansicht (Supabase + Streamlit Community Cloud)

Gerechnet wird weiter auf dem PC. Danach lädt `python -m src.cloud.sync` (`cloud_sync.bat`, läuft auch automatisch
nach `start.bat` und dem GEX-Zeitplan) die fertigen Ergebnisse in einen **privaten** Supabase-Bucket
(`dashboard-data`, Projekt macro-signal-tool). Hochgeladen werden nur Ergebnisse (Datenbank komprimiert, ML,
Fundamental, COT, GEX) – keine Rohdaten, keine Schlüssel, nur geänderte Dateien. Die Streamlit-Cloud-App lädt sie
beim Öffnen (alle 10 Minuten neu) und zeigt dasselbe Dashboard, geschützt durch ein Passwort.

1. `cloud.env` enthält die Supabase-Adresse und den Zugangscode für die Edge Function `dashboard-storage`
   (gibt nur kurzlebige, signierte Upload-/Download-Adressen für den Bucket aus; kein Supabase-Schlüssel nötig).
   `cloud_sync.bat` ausführen.
2. Code in ein **privates** GitHub-Repository (ohne `data/`, `.env`, `gex.env`, `cloud.env` – per `.gitignore`).
3. https://share.streamlit.io → New app → Repository, Branch `main`, Datei `app.py`; Advanced settings: Python 3.12,
   Secrets wie in `.streamlit/secrets.toml.example` (fertig ausgefüllt in `STREAMLIT_SECRETS.txt`). Deploy.
4. App-Einstellungen → Sharing: nur eingeladene E-Mail-Adressen. Auf dem iPhone in Safari öffnen →
   Teilen → „Zum Home-Bildschirm“.

## Befehle

| Befehl | Zweck |
|---|---|
| `python -m src.pipeline` | Alle Unternehmen laden/aktualisieren (bricht nie für alle ab, wenn eines fehlschlägt) |
| `python -m src.pipeline --tickers AAPL,MSFT` | Nur bestimmte Unternehmen |
| `python -m src.pipeline --force` | Cache ignorieren und neu laden |
| `python -m src.pipeline --build-only` | Datenbank aus vorhandenen Daten neu aufbauen |
| `python -m src.ml.run` | Kurse laden und ML-Modelle trainieren/testen (`ml.bat`); `--skip-prices` nur neu trainieren |
| `python -m src.fundamental.run` | Makro (FRED), COT (CFTC), SEC-Gewinne laden und Fundamental-Terminal berechnen (`fundamental.bat`); `--offline` ohne Downloads |
| `python -m src.cot.run` | COT-Daten (CFTC) und Kurse laden, COT-Modell lernen und testen (`cot.bat`) |
| `python -m src.gex.collect` | GEX von AnalyticsFX abrufen und GEX-KI aktualisieren (`gex.bat`; max. alle 6 h, `--force`) |
| `python -m src.cloud.sync` | Ergebnisse in den privaten Supabase-Bucket hochladen (`cloud_sync.bat`) |
| `python -m src.verify --sample 15` | Stichprobe gegen die SEC Company Concept API prüfen (`verify.bat`) |
| `python -m pytest` | Tests (`run_tests.bat`) |

Ein unterbrochener Abruf (Fenster geschlossen, Netzwerkfehler) wird beim nächsten Start dort fortgesetzt.

## Dashboard

* **Übersicht** – Anzahl Unternehmen, Abdeckung, Summen von Vermögen/Verbindlichkeiten/Eigenkapital mit Anzahl
  meldender Unternehmen und like-for-like-Veränderung, historische Entwicklung, gestapelte Bilanzstruktur,
  Schulden/Cash, Goodwill/Immaterielle, Branchenvergleich, Rankings, Veränderung über fünf Jahre.
* **Unternehmensvergleich** – mehrere Unternehmen, jede Position oder Kennzahl, Verlauf, Jahresvergleich, Heatmap.
* **Branchen** – Summen und Mediane je GICS-Sektor, Vergleich innerhalb einer Branche.
* **Unternehmensprofil** – vollständige Bilanz aller Jahre, Quartalsmodus, Kennzahlen mit Status,
  Datenqualität, Filing-Links, alle weiteren XBRL-Positionen, Excel-Export.
* **Datentabelle** – filterbar nach Unternehmen/Ticker/CIK, Branche, Jahr, Position, Abschnitt, Verfügbarkeit,
  XBRL-Tag (Datenquelle), Restatements; CSV- und Excel-Export.
* **Datenqualität** – Abrufstatus, Abdeckung je Position/Jahr und je Unternehmen, Bilanzidentitätsprüfung,
  Restatements, Hinweise, Verifikationsbericht.
* **Methodik & Quellen** – alle Regeln, Tag-Zuordnungen, Überlappungen und Kennzahlformeln.

### Markt & Machine Learning

* **Markt-Terminal** – Gesundheitsindex kurz-, mittel- und langfristig (0–100) aus über 20 Indikatoren: Trend,
  Marktbreite (20/50/200 Tage, A/D-Linie, McClellan, Hochs/Tiefs, Aktien ≥ 20 % unter Hoch, Tages-Divergenzen),
  Konzentration (RSP/SPY), Risikoappetit (IWM/SPY, XLY/XLP, HYG/IEF), Volatilität (VIX, VIX/VIX3M), Makro
  (Zinskurve, Kupfer/Gold), für Japan USD/JPY – mit historischem Test je Score-Bereich.

* **Fundamental-Terminal** – S&P 500 und Nikkei: Zinsen & Notenbanken (Fed, BoJ, Zinsdifferenz USA–Japan,
  Realzinsen, eingepreister Zinspfad), Liquidität & Kredit (Fed-Netto-Liquidität, Spreads, Finanzierungsbedingungen),
  Konjunktur (Arbeitsmarkt, Sahm-Regel, Industrie, Inflation, OECD-Frühindikatoren), Rohstoffe (Öl, Kupfer, Gold,
  Kupfer/Gold), Gewinne & Bewertung (Gewinn-/
  Umsatzwachstum und Margen aller S&P-500-Firmen aus den SEC-Daten, KGV, Risikoprämie), Währung (USD/JPY, Dollar) und
  COT-Positionierung. Richtung und Gewicht jedes Indikators werden jährlich aus der Vergangenheit gelernt; dazu eine
  ML-Prognose für 3 und 12 Monate (kalibriert, Jahr für Jahr getestet), „Was ist eingepreist?“ und die Kombination
  Fundamental + Technik. `python -m src.fundamental.run` (`fundamental.bat`).

* **COT-Signale** – Gold, Rohöl WTI, Kupfer, S&P 500 und Nasdaq 100 aus den CFTC-Commitments-of-Traders
  (Spekulanten = „speculative net positions“, Commercials, kleine Spekulanten). Gelernter COT-Score 0–100 bzw.
  klassischer Commercials-Index → Long / Neutral / Short mit einstellbaren Schwellen, historischem Test (Rendite nach
  jedem Signal, Kapitalkurve, Trades). `python -m src.cot.run` (`cot.bat`).

* **Gamma Exposure (GEX)** – über die AnalyticsFX-API (Schlüssel in `.env`: `ANALYTICSFX_API_KEY`): GEX-Profil je
  Strike, Gamma-Flip, Call-/Put-Wall, Max Gamma, Regime (positive/negative Gamma), Zeta-Regime. Da die API nur den
  aktuellen Stand liefert, speichert das Dashboard jeden Abruf; die GEX-KI lernt daraus walk-forward die Grösse der
  nächsten Tagesbewegung, die Richtung und ob der Kurs zwischen den Walls schliesst. `gex.bat`, täglicher Abruf
  per `gex_zeitplan.bat` (Mo–Fr 22:15).

* **Modell-Konsens S&P 500** – wertet alle Modelle zusammen aus (Markt-Terminal kurz/mittel/lang, Fundamental,
  Fundamental-ML, COT, Top-Warnung): Trefferquote, wenn alle bzw. k von n Modellen bullisch sind, jedes Modell
  einzeln, eine gelernte Kombination (walk-forward) und die Short-Seite.

* **ML-Prognosen** – je Unternehmen die Wahrscheinlichkeit, in den 12 Monaten nach der neuesten Jahresbilanz den
  S&P 500 zu schlagen bzw. mehr als 30 % zu fallen; dazu, was historisch bei ähnlichen Prognosen passierte, die
  wichtigsten Treiber und ein Szenario-Rechner („was wäre, wenn die Schulden 20 % stärker gewachsen wären?“).
* **ML-Backtest** – Walk-forward-Test ab 2014 ohne Blick in die Zukunft: Trennschärfe je Jahr, Quintile,
  Kalibrierung, Ergebnisse in Bull- vs. Bear-Phasen, wichtigste Merkmale, Vergleich Bilanz allein vs. Bilanz + Kurse.
* **Wirkung von Bilanzentwicklungen** – wie sich z. B. Schuldenwachstum oder sinkende Eigenkapitalquote historisch
  ausgewirkt haben, getrennt nach Marktregime (modellfrei und aus Modellsicht).
* **Marktregime** – S&P 500 mit Bull-/Bear-Phasen (200-Tage-Durchschnitt), aktuelles Regime.
* **Top-Warnung** – Frühwarnung vor Rückgängen ≥ 10 %: Marktbreite, VIX, Kreditmarkt, Überdehnung (Ampel) und ein
  ML-Top-Risiko, historisch getestet; dazu die Auswertung „Rot = Boden?“. `python -m src.ml.run --top-only`.

Daten: Bilanzen 2010–2025 (`ML_YEARS`), Kurse von Yahoo Finance. `start.bat` startet nach dem SEC-Abruf automatisch
Kursabruf und Training; manuell mit `ml.bat` bzw. `python -m src.ml.run`. Details: `docs/METHODIK.md`.

Dunkler Modus ist Standard; hell über Menü ⋮ → Settings → Theme. Die Diagramme passen sich automatisch an.

## Architektur

```
app.py                         Streamlit-Einstieg und Navigation
src/config.py                  Einstellungen aus .env, verständliche Konfigurationsfehler
src/data_sources/sec_client.py SEC-Client: User-Agent, Rate Limit, Retries/Backoff, gzip-Cache, If-Modified-Since
src/data_sources/sp500.py      Mitgliederliste, Änderungshistorie, Rekonstruktion der Mitgliedschaft
src/normalization/taxonomy.py  Standardpositionen ↔ XBRL-Tags, dokumentierte Überlappungen
src/normalization/facts.py     Instant-Facts, Stichtage, Geschäftsjahre, Restatements, Quartalsmodus
src/normalization/derive.py    Dokumentierte Ableitungen (Total Liabilities, Equity inkl. Minderheiten)
src/normalization/schemas.py   Pydantic-Validierung der SEC-Antworten
src/analytics/metrics.py       Kennzahlen mit Status ok/teilweise/NA und Begründung
src/analytics/aggregates.py    Aggregation mit Abdeckung, Panel, like-for-like
src/database/store.py          Parquet je Unternehmen (wiederaufnehmbar) → DuckDB (atomarer Neuaufbau)
src/pipeline.py                Ablaufsteuerung (CLI)
src/verify.py                  Stichproben-Verifikation
src/ml/prices.py               Kursabruf (Yahoo Finance), Cache
src/ml/features.py             Point-in-time-Merkmale, Labels, Marktregime
src/ml/model.py                Gradient Boosting, Walk-forward, Auswertungen, Treiber, Partial Dependence
src/ml/run.py                  ML-Ablauf (CLI)
src/cot/                       COT-Modell (Merkmale, Lernen, Signale, Backtest) und CLI
src/fundamental/               Fundamental-Terminal: Quellen (FRED, CFTC), SEC-Gewinne, Indikatoren, Lernen, CLI
src/visualization/charts.py    Plotly-Diagramme, farbfehlsichtigkeits-geprüfte Palette hell/dunkel
src/components/                Datenzugriff und UI-Bausteine
src/pages/                     Dashboard-Seiten
tests/                         pytest (Normalisierung, Kennzahlen, Pipeline, Rendern aller Seiten)
docs/METHODIK.md               Methodik, Quellen, Einschränkungen
```

## Tests

`python -m pytest` – über 50 Tests. Sie laufen vollständig offline gegen synthetische Antworten im exakten
SEC-Company-Facts-Format (`tests/fixtures/builder.py`; diese Testwerte sind keine echten Finanzdaten und gelangen
nie in die Anwendung). Abgedeckt: Ausschluss von Duration-Facts, Stichtag- und Geschäftsjahr-Zuordnung
(inkl. Februar-Jahresende), Vergleichsspalten, Restatements, keine Quartalswerte als Jahreswerte, 0 vs. fehlend,
Ableitungen, Kennzahlen inkl. Division durch null und negativem Eigenkapital, Panel-Aggregation, Rekonstruktion der
Indexmitgliedschaft, vollständiger Pipeline-Lauf inkl. 404 und inkrementellem Überspringen, Rendern jeder Seite.

## Bekannte Einschränkungen

Siehe `docs/METHODIK.md` (auch im Dashboard unter „Methodik & Quellen“). Kurz: aktuelle Indexzusammensetzung
rückwirkend (optional rekonstruiert), keine firmenspezifischen XBRL-Erweiterungen, Finanzschulden ohne Leasing,
keine Währungsumrechnung, GJ 2025 unvollständig solange 10-Ks ausstehen. Wie viele Unternehmen tatsächlich geladen
wurden, zeigt das Dashboard nach jedem Lauf – es wird nichts behauptet, was nicht geprüft ist.
