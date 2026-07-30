from pydantic import BaseModel, Field, field_validator

def clean_number(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.replace(",", "").replace("$", "").strip()
        return float(value) if value else 0.0
    return value

def clean_string(value):
    if value is None:
        return ""
    return str(value).strip()

def clean_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"yes","y", "true", "t","1",}
    return False

class Claim(BaseModel):
    claim_number: str = ""
    loss_date: str = ""
    status: str = ""
    cause_of_loss: str = ""
    state: str | None = None
    paid_loss: float = 0.0
    paid_expense: float = 0.0
    reserve: float = 0.0
    total_incurred: float = 0.0
    subrogation: float = 0.0
    cat_indicator: bool = False

    @field_validator(
        "claim_number",
        "loss_date",
        "status",
        "cause_of_loss",
        mode="before",
    )
    @classmethod
    def validate_strings(cls, value):
        return clean_string(value)

    @field_validator(
        "paid_loss",
        "paid_expense",
        "reserve",
        "total_incurred",
        "subrogation",
        mode="before",
    )
    @classmethod
    def validate_numbers(cls, value):
        return clean_number(value)

    @field_validator("cat_indicator", mode="before")
    @classmethod
    def validate_bool(cls, value):
        return clean_bool(value)

class LossRun(BaseModel):
    source_file: str = ""
    insured_name: str = ""
    carrier_name: str | None = None
    policy_number: str | None = None
    policy_start: str | None = None
    policy_end: str | None = None
    annual_premium: float | None = None
    per_occurrence_limit: float | None = None
    aggregate_limit: float | None = None
    deductible: float | None = None
    claims: list[Claim] = Field(default_factory=list)

    @field_validator(
        "source_file",
        "insured_name",
        mode="before",
    )
    @classmethod
    def validate_strings(cls, value):
        return clean_string(value)

    @field_validator(
        "annual_premium",
        "per_occurrence_limit",
        "aggregate_limit",
        "deductible",
        mode="before",
    )
    @classmethod
    def validate_numbers(cls, value):
        return None if value is None else clean_number(value)