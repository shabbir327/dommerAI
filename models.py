"""Pydantic request/response schemas for the DommerAI API."""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

ExamLevel = Literal["PD2", "PD3"]
RubricLevel = Literal["Top", "Midt", "Bund", "Under niveau"]
Grade = Literal[12, 10, 7, 4, 2, 0, -3]
PassFail = Literal["PASSED", "NOT PASSED"]
SubmissionStatus = Literal["pending", "processing", "scored", "failed"]
WebhookEvent = Literal["evaluation.completed", "evaluation.failed"]
ErrorType = Literal[
    "spelling", "morphology", "inversion", "syntax", "agreement",
    "punctuation", "word_choice", "missing_word", "other",
]


class EvaluationRequest(BaseModel):
    eval_id: str = Field(..., min_length=1, description="Unique evaluation ID supplied by the backend.")
    candidate_id: str | None = None
    exam_type: ExamLevel
    question: str
    question_description: str | None = None
    answer: str
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)
    webhook_url: HttpUrl | None = None

    @field_validator("eval_id", "question", "answer")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("field cannot be empty")
        return value


class AckResponse(BaseModel):
    eval_id: str
    status: SubmissionStatus = "pending"
    submitted_at: datetime


class RubricScores(BaseModel):
    pragmatisk: RubricLevel
    diskursiv: RubricLevel
    lingvistisk: RubricLevel


class InlineError(BaseModel):
    original: str
    correction: str
    type: ErrorType
    explanation_da: str
    explanation_en: str


class WebhookPayload(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    event: WebhookEvent
    eval_id: str
    status: SubmissionStatus
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    candidate_id: str | None = None
    submitted_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    webhook_url: HttpUrl | None = None

    rubrik: RubricScores | None = None
    overall: Grade | None = None
    pass_fail: PassFail | None = None
    feedback_da: str | None = None
    errors: list[InlineError] = Field(default_factory=list)
    word_count: int | None = None
    model_name: str | None = None
    prompt_version: str = "v1"
    error: str | None = None


class EvaluationStatusResponse(BaseModel):
    """Stable polling response retained for backward compatibility."""

    eval_id: str
    status: SubmissionStatus
    candidate_id: str | None = None
    exam_type: ExamLevel
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    webhook_url: HttpUrl | None = None
    result: WebhookPayload | None = None
    error: str | None = None


# Newer code may use this name; older integrations keep EvaluationStatusResponse.
EvaluationRecord = EvaluationStatusResponse


class HealthResponse(BaseModel):
    status: str
    scorer_ready: bool
    database_ready: bool = False
