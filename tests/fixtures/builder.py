"""Erzeugt synthetische Company-Facts-JSON im exakten SEC-Format – ausschließlich für Tests.
Diese Werte sind KEINE Finanzdaten realer Unternehmen und werden nie in die Anwendung geladen."""
from __future__ import annotations


class FactsBuilder:
    def __init__(self, cik: int, name: str):
        self.data = {"cik": cik, "entityName": name, "facts": {"dei": {}, "us-gaap": {}}}

    def instant(self, tag: str, end: str, val: float, accn: str, fy: int, fp: str, form: str, filed: str,
                unit: str = "USD", taxonomy: str = "us-gaap"):
        t = self.data["facts"].setdefault(taxonomy, {}).setdefault(tag, {"label": tag, "description": "", "units": {}})
        t["units"].setdefault(unit, []).append(
            {"end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed})
        return self

    def duration(self, tag: str, start: str, end: str, val: float, accn: str, fy: int, fp: str, form: str, filed: str):
        t = self.data["facts"]["us-gaap"].setdefault(tag, {"label": tag, "description": "", "units": {}})
        t["units"].setdefault("USD", []).append(
            {"start": start, "end": end, "val": val, "accn": accn, "fy": fy, "fp": fp, "form": form, "filed": filed})
        return self

    def balance(self, accn: str, fy: int, form: str, filed: str, end: str, values: dict, fp: str = "FY"):
        for tag, val in values.items():
            self.instant(tag, end, val, accn, fy, fp, form, filed)
        return self


def company_a() -> dict:
    """Dezember-Geschäftsjahr, Restatement von FY2022, Liabilities-Tag fehlt in FY2021, Cash = 0 in FY2023."""
    b = FactsBuilder(1111, "Alpha Test Corp")
    # 10-K FY2022 (enthält Vorjahr 2021)
    b.balance("0000001111-23-000001", 2022, "10-K", "2023-02-15", "2022-12-31", {
        "Assets": 880, "LiabilitiesAndStockholdersEquity": 880, "Liabilities": 480, "StockholdersEquity": 400,
        "AssetsCurrent": 300, "LiabilitiesCurrent": 200, "CashAndCashEquivalentsAtCarryingValue": 50,
        "LongTermDebtNoncurrent": 150, "DebtCurrent": 30})
    b.balance("0000001111-23-000001", 2022, "10-K", "2023-02-15", "2021-12-31", {
        "Assets": 800, "LiabilitiesAndStockholdersEquity": 800, "StockholdersEquity": 350,
        "AssetsCurrent": 250, "LiabilitiesCurrent": 180, "CashAndCashEquivalentsAtCarryingValue": 40})
    # 10-K FY2023 – restatet Assets 2022 auf 900
    b.balance("0000001111-24-000001", 2023, "10-K", "2024-02-14", "2023-12-31", {
        "Assets": 1000, "LiabilitiesAndStockholdersEquity": 1000, "Liabilities": 550, "StockholdersEquity": 450,
        "AssetsCurrent": 400, "LiabilitiesCurrent": 0, "CashAndCashEquivalentsAtCarryingValue": 0,
        "LongTermDebtNoncurrent": 200, "DebtCurrent": 50, "AccountsReceivableNetCurrent": 100})
    b.balance("0000001111-24-000001", 2023, "10-K", "2024-02-14", "2022-12-31", {
        "Assets": 900, "LiabilitiesAndStockholdersEquity": 900, "Liabilities": 500, "StockholdersEquity": 400})
    # Duration-Fact (GuV) darf nie in der Bilanz landen
    b.duration("Revenues", "2023-01-01", "2023-12-31", 5000, "0000001111-24-000001", 2023, "FY", "10-K", "2024-02-14")
    b.duration("Assets", "2023-01-01", "2023-12-31", 123456, "0000001111-24-000001", 2023, "FY", "10-K", "2024-02-14")
    # 10-Q Q1 2024 inkl. Vergleichswert 2023-12-31 (mit abweichendem Wert, darf Jahreswert nicht überschreiben)
    b.balance("0000001111-24-000010", 2024, "10-Q", "2024-05-01", "2024-03-31", {
        "Assets": 1050, "LiabilitiesAndStockholdersEquity": 1050, "StockholdersEquity": 470}, fp="Q1")
    b.balance("0000001111-24-000010", 2024, "10-Q", "2024-05-01", "2023-12-31", {"Assets": 999}, fp="Q1")
    # Fremdwährungs- und Nicht-Währungs-Facts
    b.instant("CommonStockSharesOutstanding", "2023-12-31", 1_000_000, "0000001111-24-000001", 2023, "FY", "10-K",
              "2024-02-14", unit="shares")
    return b.data


def company_b() -> dict:
    """Einzelhändler mit Geschäftsjahresende Anfang Februar, negatives Eigenkapital, Minderheiten."""
    b = FactsBuilder(2222, "Beta Retail Test Inc")
    b.balance("0000002222-24-000001", 2023, "10-K", "2024-03-20", "2024-02-03", {
        "Assets": 500, "LiabilitiesAndStockholdersEquity": 500, "StockholdersEquity": -20, "MinorityInterest": 5,
        "LongTermDebt": 300, "ShortTermBorrowings": 10, "Goodwill": 50,
        "AssetsCurrent": 200, "LiabilitiesCurrent": 150, "CashAndCashEquivalentsAtCarryingValue": 30})
    b.balance("0000002222-24-000001", 2023, "10-K", "2024-03-20", "2023-01-28", {
        "Assets": 450, "LiabilitiesAndStockholdersEquity": 450, "StockholdersEquity": 10, "MinorityInterest": 5})
    b.balance("0000002222-23-000001", 2022, "10-K", "2023-03-22", "2023-01-28", {
        "Assets": 450, "LiabilitiesAndStockholdersEquity": 450, "StockholdersEquity": 10, "MinorityInterest": 5})
    return b.data
