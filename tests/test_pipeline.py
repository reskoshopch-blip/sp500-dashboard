import duckdb

from src.database.store import read_company_status
from src.pipeline import run


def test_full_pipeline_builds_database(pipeline_db):
    settings, meta = pipeline_db
    assert meta["n_companies"] == 3          # BETA und BETA.B teilen eine CIK
    assert meta["n_loaded"] == 2             # NOFC liefert 404 -> als Lücke sichtbar
    con = duckdb.connect(str(settings.db_path), read_only=True)
    comp = con.execute("SELECT cik, ticker, fetch_status FROM companies ORDER BY cik").fetchall()
    assert comp[1][1] == "BETA/BETA-B"
    assert comp[2][2] == "keine Daten"
    n = con.execute("SELECT count(*) FROM balance_items WHERE item_key='total_assets'").fetchone()[0]
    assert n == 5                            # A: 2021-2023, B: 2022-2023
    mem = dict(con.execute("SELECT fiscal_year, was_member FROM membership WHERE cik=2222").fetchall())
    assert mem[2022] is False and mem[2023] is True   # BETA erst ab Juni 2023 im Index
    m = con.execute("SELECT status FROM metrics WHERE cik=2222 AND fiscal_year=2023 AND metric_key='debt_to_equity'").fetchone()
    assert m[0] == "NA"                      # negatives Eigenkapital
    con.close()


def test_incremental_run_skips_fresh_companies(pipeline_db):
    settings, _ = pipeline_db
    before = read_company_status(settings, 1111)["fetched_ts"]
    run()  # zweiter Lauf: Cache frisch -> übersprungen
    assert read_company_status(settings, 1111)["fetched_ts"] == before
