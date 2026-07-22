"""Pydantic request/response schemas for DommerAI."""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

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
    exam_type: ExamLevel
    question: str = Field(..., min_length=1)
    question_description: Optional[str] = None
    answer: str = Field(..., min_length=1)

    @field_validator("answer", "question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value


class AckResponse(BaseModel):
    eval_id: str
    status: SubmissionStatus = "pending"


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


class KnowledgeCitation(BaseModel):
    knowledge_id: str
    knowledge_type: str
    reason_used: str


class WebhookPayload(BaseModel):
    eval_id: str
    status: SubmissionStatus
    rubrik: Optional[RubricScores] = None
    overall: Optional[Grade] = None
    pass_fail: Optional[PassFail] = None
    feedback_da: Optional[str] = None
    errors: Optional[list[InlineError]] = None
    word_count: Optional[int] = None
    knowledge_used: Optional[list[KnowledgeCitation]] = None
    retrieval_metadata: Optional[dict] = None
    model_metadata: Optional[dict] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    scorer_ready: bool
    knowledge_ready: bool = False
    knowledge_counts: dict[str, int] = Field(default_factory=dict)
