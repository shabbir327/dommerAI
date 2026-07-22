"""Pydantic request and response schemas for DommerAI."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

ExamLevel = Literal["PD2", "PD3"]
RubricLevel = Literal["Top", "Midt", "Bund", "Under niveau"]
Grade = Literal[12, 10, 7, 4, 2, 0, -3]
PassFail = Literal["PASSED", "NOT PASSED"]
SubmissionStatus = Literal["pending", "scored", "failed"]
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
ErrorSeverity = Literal["low", "medium", "high"]


class EvaluationRequest(BaseModel):
    # Kept unchanged so Adnan can continue using the same Swagger/API payload.
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
    # Existing fields remain unchanged.
    original: str
    correction: str
    type: ErrorType
    explanation_da: str
    explanation_en: str

    # Additive location/highlighting fields.
    severity: ErrorSeverity = "medium"
    line: int = Field(..., ge=1, description="1-based line number")
    column_start: int = Field(..., ge=1, description="1-based start column")
    column_end: int = Field(..., ge=1, description="1-based exclusive end column")
    start_char: int = Field(..., ge=0, description="0-based character offset")
    end_char: int = Field(..., ge=0, description="0-based exclusive character offset")
    line_text: str
    grammar_rule_title: Optional[str] = None


class KnowledgeCitation(BaseModel):
    knowledge_id: str
    knowledge_type: str
    reason_used: str


class WritingStatistics(BaseModel):
    sentence_count: int = 0
    average_sentence_length: float = 0.0
    unique_word_count: int = 0
    lexical_diversity: float = 0.0
    detected_verbs: list[str] = Field(default_factory=list)
    detected_adjectives: list[str] = Field(default_factory=list)
    repeated_words: list[str] = Field(default_factory=list)


class WebhookPayload(BaseModel):
    # Existing response fields remain intact.
    eval_id: str
    status: SubmissionStatus
    rubrik: Optional[RubricScores] = None
    overall: Optional[Grade] = None
    pass_fail: Optional[PassFail] = None
    feedback_da: Optional[str] = None
    errors: Optional[list[InlineError]] = None
    word_count: Optional[int] = None

    # Additive fields; existing clients can ignore them safely.
    examiner_summary: Optional[str] = None
    writing_statistics: Optional[WritingStatistics] = None
    knowledge_used: Optional[list[KnowledgeCitation]] = None
    retrieval_metadata: Optional[dict[str, Any]] = None
    model_metadata: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class KnowledgeSourceHealth(BaseModel):
    status: str
    count: int = 0
    required: bool = False
    relation: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    scorer_ready: bool
    knowledge_ready: bool = False
    knowledge_counts: dict[str, int] = Field(default_factory=dict)
    knowledge_sources: dict[str, KnowledgeSourceHealth] = Field(default_factory=dict)
