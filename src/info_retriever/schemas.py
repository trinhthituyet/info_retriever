"""Pydantic models that define the structured-output schemas Claude extracts into.

Every model sets ``extra="forbid"`` because the structured-outputs API requires
``additionalProperties: false`` on all objects. Dates are typed as strings (not
``datetime.date``) on purpose: scanned contracts often state partial or oddly
formatted dates, and a hard date type turns those into schema-validation retry
loops. Normalisation to a real date happens in :mod:`info_retriever.db`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocType = Literal["rental", "employment", "insurance", "other"]

DOC_TYPES: tuple[str, ...] = ("rental", "employment", "insurance", "other")

_ISO = "ISO 8601 date as YYYY-MM-DD, or null if the document does not state it."


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Classification(_Strict):
    """First pass: what kind of document is this?"""

    doc_type: DocType = Field(description="Which contract category this document belongs to.")
    title: str = Field(description="Short human-readable title, e.g. 'Lease — 12 Rose St, 2024–2026'.")
    language: str = Field(description="Primary language of the document as an ISO 639-1 code, e.g. 'en', 'vi'.")
    summary: str = Field(description="Two or three sentences covering what this document is and its key terms.")


class Party(_Strict):
    name: str = Field(description="Legal name of the person or organisation.")
    role: str = Field(description="Their role, e.g. 'landlord', 'tenant', 'employer', 'insurer', 'policyholder'.")


class Money(_Strict):
    amount: float | None = Field(description="Numeric amount, or null if not stated.")
    currency: str | None = Field(description="ISO 4217 currency code, e.g. 'USD', 'VND', or null if not stated.")


class Clause(_Strict):
    heading: str = Field(description="Clause number and/or heading as written in the document.")
    page: int | None = Field(description="1-indexed page the clause starts on, or null if unknown.")
    summary: str = Field(description="One sentence describing the obligation or right this clause creates.")


class BaseContract(_Strict):
    """Fields worth extracting from any contract."""

    title: str
    summary: str = Field(description="Two or three sentences covering the substance of the agreement.")
    parties: list[Party] = Field(description="Every named party to the agreement.")
    effective_date: str | None = Field(description=f"Date the agreement starts. {_ISO}")
    end_date: str | None = Field(description=f"Date the agreement ends or expires. {_ISO}")
    notice_period_days: int | None = Field(
        description="Days of advance notice required to terminate, or null if not stated."
    )
    auto_renews: bool | None = Field(description="Whether the agreement renews automatically, or null if unclear.")
    governing_law: str | None = Field(description="Jurisdiction whose law governs, or null if not stated.")
    notable_clauses: list[Clause] = Field(
        description=(
            "The clauses that would actually affect the person day to day — penalties, "
            "renewal, termination, liability caps, restrictions. At most 12; skip boilerplate."
        )
    )
    obligations: list[str] = Field(
        description="Concrete things the reader personally must do or must not do, in plain language."
    )


class RentalContract(BaseContract):
    property_address: str | None = Field(description="Full address of the rented property.")
    monthly_rent: Money
    security_deposit: Money
    rent_due_day: int | None = Field(description="Day of the month rent is due (1-31), or null if not stated.")
    utilities_included: list[str] = Field(description="Utilities the rent covers, e.g. ['water', 'internet'].")
    late_fee: Money


class EmploymentContract(BaseContract):
    job_title: str | None
    employer_name: str | None
    base_salary: Money
    salary_period: str | None = Field(description="Pay period for base_salary, e.g. 'year', 'month', 'hour'.")
    bonus_terms: str | None = Field(description="How bonuses or commission are determined, if stated.")
    probation_period_months: int | None
    annual_leave_days: int | None
    working_hours: str | None = Field(description="Stated working hours or FTE, e.g. '40 hours/week'.")
    non_compete: str | None = Field(description="Non-compete or non-solicit restriction, if any.")


class InsuranceContract(BaseContract):
    policy_number: str | None
    insurer_name: str | None
    coverage_type: str | None = Field(description="What is insured, e.g. 'health', 'auto', 'renters', 'life'.")
    premium: Money
    premium_period: str | None = Field(description="Billing period for premium, e.g. 'month', 'year'.")
    deductible: Money
    coverage_limit: Money
    exclusions: list[str] = Field(description="What the policy explicitly does not cover.")


class OtherContract(BaseContract):
    """Fallback for documents that don't fit the three known categories."""


EXTRACTION_MODELS: dict[str, type[BaseContract]] = {
    "rental": RentalContract,
    "employment": EmploymentContract,
    "insurance": InsuranceContract,
    "other": OtherContract,
}


def extraction_model_for(doc_type: str) -> type[BaseContract]:
    return EXTRACTION_MODELS.get(doc_type, OtherContract)
