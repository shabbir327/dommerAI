"""
models.py — Pydantic request and response schemas for DommerAI.

Async flow:
  1. Backend POSTs an EvaluationRequest to /evaluate.
  2. DommerAI immediately returns AckResponse with status="pending".
  3. DommerAI scores the essay asynchronously.
  4. DommerAI sends WebhookPayload to the request's webhook_url.
"""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


# ---------------------------------------------------------------------------
# Shared literals
# ---------------------------------------------------------------------------

ExamLevel = Literal["PD2", "PD3"]
RubricLevel = Literal["Top", "Midt", "Bund", "Under niveau"]
Grade = Literal[12, 10, 7, 4, 2, 0, -3]
PassFail = Literal["PASSED", "NOT PASSED"]
SubmissionStatus = Literal["pending", "scored", "failed"]
WebhookEvent = Literal["evaluation.completed", "evaluation.failed"]

ErrorType = Literal[
    "spelling",
    "morphology",
    "inversion",
    "syntax",
    "agreement",
    "punctuation",
    "word_choice",
    "missing_word",
    "other",
]


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Inbound request: DanskProeve backend -> DommerAI
# ---------------------------------------------------------------------------

class EvaluationRequest(BaseModel):
    eval_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Unique evaluation ID supplied by the calling backend.",
    )
    candidate_id: str | None = Field(
        default=None,
        max_length=200,
        description="Optional candidate identifier from the calling backend.",
    )
    exam_type: ExamLevel = Field(..., description="Danish exam level: PD2 or PD3.")
    question: str = Field(..., min_length=1, description="The exam task shown to the candidate.")
    question_description: str | None = Field(
        default=None,
        description="Optional sub-tasks, instructions, or task constraints.",
    )
    answer: str = Field(..., min_length=1, description="The candidate's written response.")
    submitted_at: datetime = Field(
        default_factory=utc_now,
        description="When the calling system received the submission.",
    )
    webhook_url: HttpUrl = Field(
        ...,
        description="HTTPS endpoint that receives the completed or failed result.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional caller-defined metadata echoed in the webhook payload.",
    )

    @field_validator("eval_id", "candidate_id", "question", "question_description", "answer")
    @classmethod
    def strip_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        return value

    @field_validator("eval_id", "question", "answer")
    @classmethod
    def required_text_not_empty(cls, value: str | None) -> str:
        if value is None or not value.strip():
            raise ValueError("field cannot be empty")
        return value.strip()

    @field_validator("submitted_at")
    @classmethod
    def submitted_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("submitted_at must include a timezone")
        return value


# ---------------------------------------------------------------------------
# Immediate acknowledgement: DommerAI -> calling backend
# ---------------------------------------------------------------------------

class AckResponse(BaseModel):
    eval_id: str
    status: Literal["pending"] = "pending"


# ---------------------------------------------------------------------------
# Scoring result models
# ---------------------------------------------------------------------------

class RubricScores(BaseModel):
    pragmatisk: RubricLevel = Field(..., description="Task fulfilment, genre, and register.")
    diskursiv: RubricLevel = Field(..., description="Structure, cohesion, and coherence.")
    lingvistisk: RubricLevel = Field(..., description="Vocabulary, grammar, and spelling.")


class InlineError(BaseModel):
    original: str = Field(..., min_length=1, description="Exact text span from the candidate essay.")
    correction: str = Field(..., description="Corrected Danish version of the text span.")
    type: ErrorType = Field(..., description="Error category.")
    explanation_da: str = Field(..., min_length=1, description="Short explanation in Danish.")
    explanation_en: str = Field(..., min_length=1, description="Short explanation in English.")


# ---------------------------------------------------------------------------
# Webhook payload: DommerAI -> calling backend
# ---------------------------------------------------------------------------

class WebhookPayload(BaseModel):
    event: WebhookEvent
    eval_id: str
    candidate_id: str | None = None
    status: Literal["scored", "failed"]
    completed_at: datetime = Field(default_factory=utc_now)

    # Successful scoring result
    rubrik: RubricScores | None = None
    overall: Grade | None = None
    pass_fail: PassFail | None = None
    feedback_da: str | None = None
    errors: list[InlineError] = Field(default_factory=list)
    word_count: int | None = Field(default=None, ge=0)

    # Failure details
    error: str | None = None

    # Caller-defined context echoed from EvaluationRequest
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_event_and_status(self) -> "WebhookPayload":
        if self.status == "scored":
            if self.event != "evaluation.completed":
                raise ValueError("status='scored' requires event='evaluation.completed'")
            if self.rubrik is None or self.overall is None or self.pass_fail is None:
                raise ValueError("a scored payload requires rubrik, overall, and pass_fail")
            if self.error is not None:
                raise ValueError("a scored payload must not contain an error")

        if self.status == "failed":
            if self.event != "evaluation.failed":
                raise ValueError("status='failed' requires event='evaluation.failed'")
            if not self.error:
                raise ValueError("a failed payload requires an error message")

        return self


# ---------------------------------------------------------------------------
# Stored evaluation response
# ---------------------------------------------------------------------------

class EvaluationStatusResponse(BaseModel):
    """Stable polling response retained for backward compatibility."""

    eval_id: str
    status: SubmissionStatus
    candidate_id: Optional[str] = None
    exam_type: Optional[ExamLevel] = None

    submitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    webhook_url: Optional[HttpUrl] = None
    result: Optional[WebhookPayload] = None
    error: Optional[str] = None

# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    scorer_ready: bool
