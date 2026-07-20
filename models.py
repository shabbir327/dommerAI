from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator

ExamLevel = Literal["PD2", "PD3"]
RubricLevel = Literal["Top", "Midt", "Bund", "Under niveau"]
Grade = Literal[12, 10, 7, 4, 2, 0, -3]
PassFail = Literal["PASSED", "NOT PASSED"]
SubmissionStatus = Literal["pending", "scored", "failed"]
ErrorType = Literal[
    "spelling", "morphology", "inversion", "syntax", "agreement",
    "punctuation", "word_choice", "missing_word", "other",
]


class EvaluationRequest(BaseModel):
    eval_id: str = Field(..., min_length=1)
    candidate_id: Optional[str] = None
    exam_type: ExamLevel
    question: str = Field(..., min_length=1)
    question_description: Optional[str] = None
    answer: str = Field(..., min_length=1)
    submitted_at: Optional[datetime] = None
    webhook_url: HttpUrl
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question", "answer")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty.")
        return value


class AckResponse(BaseModel):
    eval_id: str
    status: Literal["pending"] = "pending"


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
    event: Literal["evaluation.completed", "evaluation.failed"]
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    eval_id: str
    status: SubmissionStatus
    completed_at: datetime
    rubrik: Optional[RubricScores] = None
    overall: Optional[Grade] = None
    pass_fail: Optional[PassFail] = None
    feedback_da: Optional[str] = None
    errors: list[InlineError] = Field(default_factory=list)
    word_count: Optional[int] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationStatusResponse(BaseModel):
    eval_id: str
    status: SubmissionStatus
    result: Optional[WebhookPayload] = None


class HealthResponse(BaseModel):
    status: str
    scorer_ready: bool
