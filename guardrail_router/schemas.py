"""Request and response models shared by the routes, the pipeline and the tests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Decision = Literal["allow", "review", "block"]
Category = Literal["benign", "injection", "exfiltration", "jailbreak"]
Stage = Literal["classifier", "judge", "fallback"]

MAX_ITEMS = 64


class ScreenRequest(BaseModel):
    """One piece of untrusted text to screen before it reaches a model."""

    tenant_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=20000)
    context: str | None = Field(default=None, max_length=2000)

    @field_validator("tenant_id")
    @classmethod
    def _tenant_is_slug(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("tenant_id must not be blank")
        return cleaned


class BatchScreenRequest(BaseModel):
    """A batch of texts from one tenant, used for offline evaluation runs."""

    tenant_id: str = Field(min_length=1, max_length=64)
    items: list[str] = Field(min_length=1, max_length=MAX_ITEMS)
    context: str | None = Field(default=None, max_length=2000)

    @field_validator("items")
    @classmethod
    def _items_not_blank(cls, value: list[str]) -> list[str]:
        for item in value:
            if not item.strip():
                raise ValueError("batch items must not be blank")
        return value


class Verdict(BaseModel):
    """The outcome of screening one text."""

    decision: Decision
    category: Category
    score: float = Field(ge=0.0, le=1.0)
    stage: Stage
    latency_ms: float = Field(ge=0.0)
    excerpt: str


class BatchScreenResponse(BaseModel):
    """Verdicts for a batch, in the same order as the submitted items."""

    tenant_id: str
    verdicts: list[Verdict]
    blocked: int = 0

    @classmethod
    def from_verdicts(cls, tenant_id: str, verdicts: list[Verdict]) -> "BatchScreenResponse":
        blocked = sum(1 for verdict in verdicts if verdict.decision == "block")
        return cls(tenant_id=tenant_id, verdicts=verdicts, blocked=blocked)


class HealthResponse(BaseModel):
    """Readiness payload for the Kubernetes probe."""

    status: Literal["ok", "degraded"]
    classifier_loaded: bool
    judge_configured: bool
    version: str
