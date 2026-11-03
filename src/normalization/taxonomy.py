"""Kanonische Bilanzpositionen und ihre Zuordnung zu US-GAAP-XBRL-Tags.

Regeln:
  * Pro Position eine geordnete Liste von Kandidaten-Tags. Pro Unternehmen und Geschäftsjahr
    wird der erste Tag verwendet, für den ein Wert existiert. Der verwendete Tag wird gespeichert.
  * Kandidaten-Tags einer Position haben dieselbe ökonomische Definition. Tags mit abweichender
    Definition (z. B. Cash inkl. Restricted Cash, Debt inkl. Leasing) bekommen eine eigene Position,
    damit keine Definitionen vermischt werden.
  * Positionen werden NIE aufaddiert, um eine übergeordnete Summe zu bilden – Ausnahme sind die
    dokumentierten, als "berechnet" markierten Ableitungen in derive.py.
  * Die Liste ist eine Startliste. Alle weiteren Instant-Facts in USD aus Jahresberichten werden
    zusätzlich dynamisch als "weitere XBRL-Positionen" erfasst (siehe facts.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    key: str
    label_de: str
    label_en: str
    section: str          # Aktiva | Passiva | Eigenkapital | Banken | Versicherungen | Immobilien | Abstimmung
    level: str            # total | subtotal | detail
    tags: tuple[str, ...]
    note: str = ""
    overlaps: tuple[str, ...] = field(default_factory=tuple)  # Positionen, die diese Position (teilweise) enthalten


ITEMS: tuple[Item, ...] = (
    # ------------------------------------------------------------------ Aktiva
    Item("total_assets", "Bilanzsumme (Total Assets)", "Total Assets", "Aktiva", "total", ("Assets",)),
    Item("current_assets", "Umlaufvermögen", "Current Assets", "Aktiva", "subtotal", ("AssetsCurrent",)),
    Item("noncurrent_assets", "Anlagevermögen (gesamt)", "Non-Current Assets", "Aktiva", "subtotal", ("AssetsNoncurrent",),
         "Wird nur von wenigen Unternehmen ausgewiesen; nicht aus Teilpositionen abgeleitet."),
    Item("cash", "Zahlungsmittel und -äquivalente", "Cash and Cash Equivalents", "Aktiva", "detail",
         ("CashAndCashEquivalentsAtCarryingValue", "Cash"), overlaps=("current_assets",)),
    Item("cash_incl_restricted", "Zahlungsmittel inkl. Restricted Cash", "Cash incl. Restricted Cash", "Aktiva", "detail",
         ("CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",),
         "Andere Definition als 'cash' (enthält verfügungsbeschränkte Mittel); wird nicht als Ersatz verwendet."),
    Item("st_investments", "Kurzfristige Finanzanlagen", "Short-Term Investments", "Aktiva", "detail",
         ("ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"),
         overlaps=("current_assets",)),
    Item("receivables", "Forderungen aus L&L", "Accounts Receivable", "Aktiva", "detail",
         ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent", "AccountsNotesAndLoansReceivableNetCurrent"),
         "ReceivablesNetCurrent kann sonstige Forderungen enthalten (Tag wird gespeichert).", ("current_assets",)),
    Item("inventory", "Vorräte", "Inventory", "Aktiva", "detail", ("InventoryNet",), overlaps=("current_assets",)),
    Item("prepaid", "Rechnungsabgrenzung (Prepaid Expenses)", "Prepaid Expenses", "Aktiva", "detail",
         ("PrepaidExpenseCurrent",), overlaps=("current_assets",)),
    Item("prepaid_and_other_current", "Prepaid und sonstiges Umlaufvermögen", "Prepaid Expense and Other Current Assets",
         "Aktiva", "detail", ("PrepaidExpenseAndOtherAssetsCurrent",),
         "Kombiposition – überschneidet sich mit 'prepaid' und 'other_current_assets'.", ("current_assets",)),
    Item("other_current_assets", "Sonstiges Umlaufvermögen", "Other Current Assets", "Aktiva", "detail",
         ("OtherAssetsCurrent",), overlaps=("current_assets",)),
    Item("ppe", "Sachanlagen (netto)", "Property, Plant and Equipment, net", "Aktiva", "detail",
         ("PropertyPlantAndEquipmentNet",)),
    Item("ppe_incl_fin_lease", "Sachanlagen inkl. Finance-Lease-ROU", "PP&E incl. Finance Lease ROU", "Aktiva", "detail",
         ("PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",),
         "Andere Definition als 'ppe' (enthält Finance-Lease-Nutzungsrechte)."),
    Item("goodwill", "Goodwill", "Goodwill", "Aktiva", "detail", ("Goodwill",)),
    Item("intangibles", "Immaterielle Vermögenswerte (ohne Goodwill)", "Intangible Assets excl. Goodwill", "Aktiva", "detail",
         ("IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"),
         "FiniteLivedIntangibleAssetsNet enthält keine unbegrenzt nutzbaren Immateriellen (Tag wird gespeichert)."),
    Item("lt_investments", "Langfristige Finanzanlagen", "Long-Term Investments", "Aktiva", "detail",
         ("LongTermInvestments", "MarketableSecuritiesNoncurrent", "AvailableForSaleSecuritiesDebtSecuritiesNoncurrent")),
    Item("equity_method_investments", "At-Equity-Beteiligungen", "Equity Method Investments", "Aktiva", "detail",
         ("EquityMethodInvestments",)),
    Item("dta", "Latente Steueransprüche", "Deferred Tax Assets", "Aktiva", "detail",
         ("DeferredIncomeTaxAssetsNet", "DeferredTaxAssetsNetNoncurrent"),
         "DeferredTaxAssetsNet (Anhangangabe vor Saldierung) wird bewusst nicht verwendet."),
    Item("rou_assets", "Operating-Lease-Nutzungsrechte", "Operating Lease ROU Assets", "Aktiva", "detail",
         ("OperatingLeaseRightOfUseAsset",)),
    Item("other_noncurrent_assets", "Sonstiges Anlagevermögen", "Other Non-Current Assets", "Aktiva", "detail",
         ("OtherAssetsNoncurrent",)),
    # ----------------------------------------------------------------- Passiva
    Item("total_liabilities", "Verbindlichkeiten gesamt", "Total Liabilities", "Passiva", "total", ("Liabilities",),
         "Fehlt der Tag, wird der Wert aus Bilanzsumme − Eigenkapital (inkl. Minderheiten) − Mezzanine-Kapital berechnet."),
    Item("current_liabilities", "Kurzfristige Verbindlichkeiten", "Current Liabilities", "Passiva", "subtotal",
         ("LiabilitiesCurrent",)),
    Item("noncurrent_liabilities", "Langfristige Verbindlichkeiten", "Non-Current Liabilities", "Passiva", "subtotal",
         ("LiabilitiesNoncurrent",)),
    Item("accounts_payable", "Verbindlichkeiten aus L&L", "Accounts Payable", "Passiva", "detail",
         ("AccountsPayableCurrent", "AccountsPayableTradeCurrent"), overlaps=("current_liabilities",)),
    Item("ap_and_accrued", "Verbindl. aus L&L und Abgrenzungen", "Accounts Payable and Accrued Liabilities",
         "Passiva", "detail", ("AccountsPayableAndAccruedLiabilitiesCurrent",),
         "Kombiposition – überschneidet sich mit 'accounts_payable' und 'accrued'.", ("current_liabilities",)),
    Item("accrued", "Passive Abgrenzungen (Accrued Expenses)", "Accrued Liabilities", "Passiva", "detail",
         ("AccruedLiabilitiesCurrent",), overlaps=("current_liabilities",)),
    Item("st_debt", "Kurzfristige Finanzschulden", "Short-Term Debt", "Passiva", "detail",
         ("ShortTermBorrowings", "CommercialPaper", "OtherShortTermBorrowings"),
         "Ohne kurzfristigen Anteil langfristiger Schulden.", ("current_liabilities", "debt_current_total")),
    Item("ltd_current", "Kurzfristiger Anteil langfr. Schulden", "Current Portion of Long-Term Debt", "Passiva", "detail",
         ("LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
         "Zweiter Tag enthält Finance-Lease-Anteile (Tag wird gespeichert).", ("current_liabilities", "debt_current_total", "ltd_total")),
    Item("debt_current_total", "Kurzfristige Schulden gesamt", "Debt, Current (total)", "Passiva", "detail",
         ("DebtCurrent",), "Enthält st_debt + ltd_current – nie mit diesen addieren.", ("current_liabilities",)),
    Item("ltd_noncurrent", "Langfristige Finanzschulden", "Long-Term Debt, Non-Current", "Passiva", "detail",
         ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"),
         "Zweiter Tag enthält Finance-Lease-Verbindlichkeiten (Tag wird gespeichert).", ("ltd_total",)),
    Item("ltd_total", "Langfristige Schulden inkl. kurzfr. Anteil", "Long-Term Debt (incl. current portion)", "Passiva",
         "detail", ("LongTermDebt",), "Enthält ltd_current + ltd_noncurrent."),
    Item("op_lease_liab", "Operating-Lease-Verbindlichkeiten gesamt", "Operating Lease Liabilities", "Passiva", "detail",
         ("OperatingLeaseLiability",)),
    Item("op_lease_liab_current", "Operating-Lease-Verbindl. (kurzfristig)", "Operating Lease Liabilities, Current",
         "Passiva", "detail", ("OperatingLeaseLiabilityCurrent",), overlaps=("op_lease_liab", "current_liabilities")),
    Item("op_lease_liab_noncurrent", "Operating-Lease-Verbindl. (langfristig)", "Operating Lease Liabilities, Non-Current",
         "Passiva", "detail", ("OperatingLeaseLiabilityNoncurrent",), overlaps=("op_lease_liab",)),
    Item("fin_lease_liab", "Finance-Lease-Verbindlichkeiten", "Finance Lease Liabilities", "Passiva", "detail",
         ("FinanceLeaseLiability",)),
    Item("pension", "Pensionsverpflichtungen (langfristig)", "Pension Liabilities", "Passiva", "detail",
         ("DefinedBenefitPensionPlanLiabilitiesNoncurrent",
          "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
          "PensionAndOtherPostretirementAndPostemploymentBenefitPlansLiabilitiesNoncurrent"),
         "Teilweise inkl. sonstiger Leistungen nach Beendigung (Tag wird gespeichert)."),
    Item("dtl", "Latente Steuerschulden", "Deferred Tax Liabilities", "Passiva", "detail",
         ("DeferredIncomeTaxLiabilitiesNet", "DeferredTaxLiabilitiesNoncurrent"),
         "DeferredTaxLiabilities (Anhangangabe brutto) wird bewusst nicht verwendet."),
    Item("contract_liab_current", "Vertragsverbindlichkeiten (kurzfristig)", "Contract Liabilities, Current", "Passiva",
         "detail", ("ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent"), overlaps=("current_liabilities",)),
    Item("contract_liab_noncurrent", "Vertragsverbindlichkeiten (langfristig)", "Contract Liabilities, Non-Current",
         "Passiva", "detail", ("ContractWithCustomerLiabilityNoncurrent", "DeferredRevenueNoncurrent")),
    Item("provisions_contingencies", "Rückstellungen für Eventualverbindlichkeiten", "Loss Contingency Accruals", "Passiva",
         "detail", ("LossContingencyAccrualAtCarryingValue",),
         "US-GAAP kennt keine allgemeine Position 'Provisions'; gemeldete Rückstellungs-Tags werden einzeln geführt."),
    Item("aro", "Rückbauverpflichtungen", "Asset Retirement Obligations", "Passiva", "detail",
         ("AssetRetirementObligationsNoncurrent", "AssetRetirementObligation")),
    Item("other_current_liab", "Sonstige kurzfristige Verbindlichkeiten", "Other Current Liabilities", "Passiva", "detail",
         ("OtherLiabilitiesCurrent",), overlaps=("current_liabilities",)),
    Item("other_noncurrent_liab", "Sonstige langfristige Verbindlichkeiten", "Other Non-Current Liabilities", "Passiva",
         "detail", ("OtherLiabilitiesNoncurrent",)),
    # ------------------------------------------------------------ Eigenkapital
    Item("equity_parent", "Eigenkapital (Aktionäre der Muttergesellschaft)", "Total Stockholders' Equity", "Eigenkapital",
         "total", ("StockholdersEquity",)),
    Item("equity_total", "Eigenkapital inkl. Minderheiten", "Total Equity incl. Noncontrolling Interests", "Eigenkapital",
         "total", ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",),
         "Fehlt der Tag, wird der Wert aus equity_parent + nci berechnet (bzw. = equity_parent, wenn keine Minderheiten gemeldet)."),
    Item("preferred_stock", "Vorzugsaktien", "Preferred Stock", "Eigenkapital", "detail", ("PreferredStockValue",)),
    Item("common_stock", "Gezeichnetes Kapital (Common Stock)", "Common Stock", "Eigenkapital", "detail",
         ("CommonStockValue",)),
    Item("common_stock_incl_apic", "Common Stock inkl. Kapitalrücklage", "Common Stock incl. APIC", "Eigenkapital", "detail",
         ("CommonStocksIncludingAdditionalPaidInCapital",), "Kombiposition – überschneidet sich mit common_stock und apic."),
    Item("apic", "Kapitalrücklage (APIC)", "Additional Paid-in Capital", "Eigenkapital", "detail",
         ("AdditionalPaidInCapital", "AdditionalPaidInCapitalCommonStock")),
    Item("retained_earnings", "Gewinnrücklagen / Bilanzverlust", "Retained Earnings (Accumulated Deficit)", "Eigenkapital",
         "detail", ("RetainedEarningsAccumulatedDeficit",)),
    Item("treasury_stock", "Eigene Aktien", "Treasury Stock", "Eigenkapital", "detail",
         ("TreasuryStockValue", "TreasuryStockCommonValue"),
         "SEC meldet eigene Aktien als positiven Betrag, der das Eigenkapital mindert."),
    Item("aoci", "Kumuliertes sonstiges Ergebnis (AOCI)", "Accumulated Other Comprehensive Income", "Eigenkapital", "detail",
         ("AccumulatedOtherComprehensiveIncomeLossNetOfTax",)),
    Item("nci", "Nicht beherrschende Anteile", "Noncontrolling Interests", "Eigenkapital", "detail", ("MinorityInterest",)),
    Item("temporary_equity", "Mezzanine-Kapital (Temporary Equity)", "Temporary / Redeemable Equity", "Abstimmung", "detail",
         ("TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests",
          "TemporaryEquityCarryingAmountAttributableToParent",
          "RedeemableNoncontrollingInterestEquityCarryingAmount"),
         "Steht zwischen Fremd- und Eigenkapital; wird bei der Ableitung von Total Liabilities abgezogen."),
    Item("liabilities_and_equity", "Summe Passiva (Liabilities and Equity)", "Liabilities and Stockholders' Equity",
         "Abstimmung", "total", ("LiabilitiesAndStockholdersEquity",)),
    # -------------------------------------------------------- Branchenspezifisch
    Item("bank_deposits", "Kundeneinlagen", "Deposits", "Banken", "detail", ("Deposits",)),
    Item("bank_loans_net", "Kredite (netto)", "Loans and Leases Receivable, net", "Banken", "detail",
         ("LoansAndLeasesReceivableNetReportedAmount",
          "FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss")),
    Item("bank_afs_securities", "Wertpapiere Available-for-Sale", "AFS Debt Securities", "Banken", "detail",
         ("AvailableForSaleSecuritiesDebtSecurities",)),
    Item("bank_htm_securities", "Wertpapiere Held-to-Maturity", "HTM Securities", "Banken", "detail",
         ("HeldToMaturitySecuritiesAmortizedCostAfterAllowanceForCreditLoss", "HeldToMaturitySecurities")),
    Item("bank_reverse_repo", "Reverse Repos / Fed Funds Sold", "Fed Funds Sold and Reverse Repos", "Banken", "detail",
         ("FederalFundsSoldAndSecuritiesPurchasedUnderAgreementsToResell",)),
    Item("bank_repo", "Repos / Fed Funds Purchased", "Fed Funds Purchased and Repos", "Banken", "detail",
         ("FederalFundsPurchasedAndSecuritiesSoldUnderAgreementsToRepurchase", "SecuritiesSoldUnderAgreementsToRepurchase")),
    Item("ins_future_policy_benefits", "Deckungsrückstellung", "Liability for Future Policy Benefits", "Versicherungen",
         "detail", ("LiabilityForFuturePolicyBenefits",)),
    Item("ins_unpaid_claims", "Schadenrückstellung", "Liability for Unpaid Claims", "Versicherungen", "detail",
         ("LiabilityForClaimsAndClaimsAdjustmentExpense",)),
    Item("ins_separate_account_assets", "Separate-Account-Vermögen", "Separate Account Assets", "Versicherungen", "detail",
         ("SeparateAccountAssets",)),
    Item("ins_separate_account_liab", "Separate-Account-Verbindlichkeiten", "Separate Account Liabilities",
         "Versicherungen", "detail", ("SeparateAccountsLiability",)),
    Item("reit_real_estate_net", "Renditeimmobilien (netto)", "Real Estate Investment Property, net", "Immobilien",
         "detail", ("RealEstateInvestmentPropertyNet",)),
)

ITEM_BY_KEY: dict[str, Item] = {i.key: i for i in ITEMS}
SECTION_ORDER = ["Aktiva", "Passiva", "Eigenkapital", "Abstimmung", "Banken", "Versicherungen", "Immobilien"]

# Tag -> Position (für Duplikatprüfung)
TAG_TO_ITEM: dict[str, str] = {}
for _it in ITEMS:
    for _t in _it.tags:
        if _t in TAG_TO_ITEM:
            raise ValueError(f"Tag {_t} ist mehreren Positionen zugeordnet")
        TAG_TO_ITEM[_t] = _it.key


def overlap_documentation() -> list[dict]:
    """Tabellarische Dokumentation überlappender Positionen/Tags für die Methodik-Seite."""
    rows = []
    for it in ITEMS:
        if it.overlaps or len(it.tags) > 1 or it.note:
            rows.append({
                "Position": it.label_de, "Schlüssel": it.key, "Abschnitt": it.section,
                "XBRL-Tags (Priorität)": " → ".join(it.tags),
                "Enthalten in": ", ".join(ITEM_BY_KEY[k].label_de for k in it.overlaps if k in ITEM_BY_KEY),
                "Hinweis": it.note,
            })
    return rows
