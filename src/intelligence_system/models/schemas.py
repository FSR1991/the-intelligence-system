from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class TaskStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


class TaskCreateRequest(ApiSchema):
    query: str = Field(min_length=1, max_length=20000)
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    metadata_filter: dict[
        str, str | int | float | bool | list[str]
    ] = Field(default_factory=dict, max_length=20)


class TaskAcceptedResponse(ApiSchema):
    task_id: str = Field(min_length=1)
    status: TaskStatus = TaskStatus.QUEUED


class SourceCitation(ApiSchema):
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    excerpt: str = Field(min_length=1, max_length=3000)
    score: float | None = None


class TaskResult(ApiSchema):
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)
    elapsed_ms: float | None = Field(default=None, ge=0)


class TaskStatusResponse(ApiSchema):
    task_id: str = Field(min_length=1)
    status: TaskStatus
    conversation_id: str | None = None
    created_at: datetime
    updated_at: datetime
    result: TaskResult | None = None
    error: str | None = None


class ApprovalDecisionRequest(ApiSchema):
    decision: Literal["approve", "reject"]
    feedback: str | None = Field(default=None, max_length=2000)


class ApprovalDecisionResponse(ApiSchema):
    task_id: str = Field(min_length=1)
    status: TaskStatus


class HealthResponse(ApiSchema):
    status: Literal["ok", "degraded"]
    app_name: str


class ErrorResponse(ApiSchema):
    code: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)
    request_id: str | None = None


# Contratos de entrada y salida de herramientas.

class RAGSearchInput(ApiSchema):
    query: str = Field(min_length=1, max_length=12000)
    top_k: int = Field(default=5, ge=1, le=20)
    metadata_filter: dict[
        str, str | int | float | bool | list[str]
    ] = Field(default_factory=dict, max_length=20)


class RAGSearchOutput(ApiSchema):
    results: list[SourceCitation] = Field(default_factory=list)


class SupervisorDecision(ApiSchema):
    next_agent: Literal[
        "retrieval_agent",
        "answer_agent",
        "finish",
    ]
    reason: str = Field(min_length=1, max_length=2000)


class AgentMessage(ApiSchema):
    agent: str = Field(min_length=1, max_length=100)
    content: str
    sources: list[SourceCitation] = Field(default_factory=list)