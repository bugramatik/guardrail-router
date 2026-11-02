"""Prometheus instruments for screening outcomes."""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .schemas import Verdict

REQUEST_LATENCY = Histogram(
    "guardrail_request_latency_ms",
    "End-to-end screening latency in milliseconds",
    buckets=(5, 10, 20, 40, 60, 80, 120, 200, 400, 800),
)

STAGE_COUNTER = Counter(
    "guardrail_stage_total",
    "Verdicts produced per pipeline stage",
    ["stage"],
)

DECISION_COUNTER = Counter(
    "guardrail_decision_total",
    "Verdicts produced per decision and category",
    ["decision", "category"],
)

RATE_LIMITED = Counter(
    "guardrail_rate_limited_total",
    "Requests rejected by the per-tenant limiter",
    ["tenant_id"],
)


def observe(verdict: Verdict) -> None:
    """Record one verdict in the latency, stage and decision instruments."""
    REQUEST_LATENCY.observe(verdict.latency_ms)
    STAGE_COUNTER.labels(stage=verdict.stage).inc()
    DECISION_COUNTER.labels(decision=verdict.decision, category=verdict.category).inc()


def render() -> bytes:
    """Serialise the current metric values in the Prometheus text format."""
    return generate_latest()


__all__ = [
    "CONTENT_TYPE_LATEST",
    "DECISION_COUNTER",
    "RATE_LIMITED",
    "REQUEST_LATENCY",
    "STAGE_COUNTER",
    "observe",
    "render",
]
