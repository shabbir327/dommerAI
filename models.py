"""Pydantic request and response schemas for DommerAI."""

from typing import Any, Literal, Optional

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator

ExamLevel = Literal["PD2", "PD3"]
RubricLevel = Literal["Top", "Midt", "Bund", "Under niveau"]
Grade = Literal[12, 10, 7, 4, 2, 0, -3]
PassFail = Literal["PASSED", "NOT PASSED"]
SubmissionStatus = Literal["pending", "scored", "failed"]
WebhookSource = Literal["request", "environment", "none"]
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
RubricDimension = Literal["pragmatisk", "diskursiv", "lingvistisk"]
DifficultyLevel = Literal["A1", "A2", "B1", "B2", "C1", "C2", "unknown"]


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    eval_id: str = Field(..., min_length=1, examples=["pd2-test-001"])
    exam_type: ExamLevel
    question: str = Field(..., min_length=1)
    question_description: Optional[str] = None
    answer: str = Field(..., min_length=1)
    webhook_url: Optional[AnyHttpUrl] = Field(
        default=None,
        description=(
            "Optional callback URL for this evaluation. When omitted, DommerAI "
            "uses the WEBHOOK_URL environment variable."
        ),
        examples=["https://webhook.site/your-test-id"],
    )

    @field_validator("eval_id", "answer", "question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value cannot be empty")
        return value


class AckResponse(BaseModel):
    eval_id: str
    status: SubmissionStatus = "pending"
    webhook_url_used: Optional[str] = None
    webhook_source: WebhookSource = "none"


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
    severity: ErrorSeverity = "medium"
    line: int = Field(..., ge=1, description="1-based line number")
    column_start: int = Field(..., ge=1, description="1-based start column")
    column_end: int = Field(..., ge=1, description="1-based exclusive end column")
    start_char: int = Field(..., ge=0, description="0-based character offset")
    end_char: int = Field(..., ge=0, description="0-based exclusive character offset")
    line_text: str
    grammar_rule_title: Optional[str] = None
    official_reference: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    affects_score: bool = True
    rubric_dimension: RubricDimension = "lingvistisk"
    difficulty: DifficultyLevel = "unknown"


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
    model_config = ConfigDict(protected_namespaces=())

    eval_id: str
    status: SubmissionStatus
    rubrik: Optional[RubricScores] = None
    overall: Optional[Grade] = None
    pass_fail: Optional[PassFail] = None
    feedback_da: Optional[str] = None
    examiner_summary: Optional[str] = None
    dimension_reasons: Optional[dict[str, str]] = None
    task_coverage: Optional[list[dict[str, str]]] = None
    strengths: Optional[list[str]] = None
    improvements: Optional[list[str]] = None
    errors: Optional[list[InlineError]] = None
    word_count: Optional[int] = None
    writing_statistics: Optional[WritingStatistics] = None
    knowledge_used: Optional[list[KnowledgeCitation]] = None
    retrieval_metadata: Optional[dict[str, Any]] = None
    model_metadata: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    updated_at: Optional[str] = None


class KnowledgeSourceHealth(BaseModel):
    status: str
    count: int = 0
    required: bool = False
    relation: str
    detail: Optional[str] = None


class ScorerHealth(BaseModel):
    status: str
    ready: bool
    provider: Optional[str] = None
    model: Optional[str] = None
    intern_model: Optional[str] = None
    prompt_version: Optional[str] = None
    grammar_hub_integrated: bool = False


class GrammarHubHealth(BaseModel):
    status: str
    ready: bool
    configured: bool = False
    database_reachable: bool = False
    sample_row_available: bool = False
    latency_ms: Optional[float] = None
    integrated_into_scorer: bool = False
    detail: Optional[str] = None


class PersistenceHealth(BaseModel):
    status: str
    ready: bool
    database_client_ready: bool = False


class PosTaggerHealth(BaseModel):
    status: str
    ready: bool
    model: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    app_version: str
    scorer_ready: bool
    knowledge_ready: bool = False
    lexical_engine_ready: bool = False
    grammar_hub_connected: bool = False
    persistence_ready: bool = False
    scorer: ScorerHealth
    grammar_hub: GrammarHubHealth
    persistence: PersistenceHealth
    pos_tagger: PosTaggerHealth
    knowledge_counts: dict[str, int] = Field(default_factory=dict)
    knowledge_sources: dict[str, KnowledgeSourceHealth] = Field(default_factory=dict)


class EvaluationListResponse(BaseModel):
    count: int
    items: list[dict[str, Any]] = Field(default_factory=list)
