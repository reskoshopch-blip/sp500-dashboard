"""Pydantic-Schemas zur Validierung der SEC-Antworten und der gespeicherten Bilanzwerte."""
from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FactEntry(BaseModel):
    model_config = ConfigDict(extra="allow")
    end: date
    val: float
    accn: str
    fy: int | None = None
    fp: str | None = None
    form: str
    filed: date
    start: date | None = None


class Concept(BaseModel):
    model_config = ConfigDict(extra="allow")
    label: str | None = None
    units: dict[str, list[dict]]


class CompanyFacts(BaseModel):
    model_config = ConfigDict(extra="allow")
    cik: int
    entityName: str = ""
    facts: dict[str, dict[str, Concept]] = Field(default_factory=dict)


class BalanceItemRecord(BaseModel):
    cik: int
    fiscal_year: int
    item_key: str
    value: float
    unit: str
    tag: str
    is_derived: bool

    @field_validator("unit")
    @classmethod
    def currency(cls, v: str) -> str:
        if len(v) != 3 or not v.isalpha():
            raise ValueError(f"Keine Währungseinheit: {v}")
        return v


def validate_companyfacts(data: dict, sample_per_concept: int = 3) -> CompanyFacts:
    """Prüft die Struktur vollständig und je Konzept/Einheit stichprobenartig einzelne Facts
    (eine Vollprüfung aller Millionen Facts wäre unnötig langsam)."""
    cf = CompanyFacts.model_validate(data)
    for concepts in cf.facts.values():
        for concept in concepts.values():
            for entries in concept.units.values():
                for e in entries[:sample_per_concept]:
                    FactEntry.model_validate(e)
    return cf
