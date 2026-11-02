"""Two-stage screening: cheap classifier first, small model judge only for the uncertain band."""

from __future__ import annotations

import re
import time
from typing import Optional

from .classifier import PromptClassifier
from .config import Settings
from .schemas import Verdict
from .slm import SlmJudge

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
SECRET_RE = re.compile(r"\b(?:sk|pk|ghp|xoxb)-[A-Za-z0-9_-]{8,}\b")

# The judge answers with one of its own labels; the verdict uses the API's categories.
CATEGORY_BY_LABEL = {
    "injection": "injection",
    "exfiltration": "exfiltration",
    "benign": "benign",
}


class ScreeningPipeline:
    """Decides whether a piece of untrusted text may reach a downstream model."""

    def __init__(
        self,
        classifier: PromptClassifier,
        judge: SlmJudge,
        settings: Settings,
    ) -> None:
        self.classifier = classifier
        self.judge = judge
        self.settings = settings

    @classmethod
    def from_settings(cls, settings: Settings) -> "ScreeningPipeline":
        """Build the pipeline with the real classifier and judge for the given settings."""
        return cls(
            PromptClassifier(settings.model_path),
            SlmJudge(settings.slm_endpoint, settings.slm_timeout_ms),
            settings,
        )

    async def screen(self, text: str, context: Optional[str] = None) -> Verdict:
        started = time.perf_counter()
        clipped = text[: self.settings.max_text_chars]

        score = self.classifier.score(clipped)
        judge_score: Optional[float] = None
        category = "benign"
        stage = "classifier"

        if score >= self.settings.threshold_high:
            category = "injection"
        elif score <= self.settings.threshold_low:
            category = "benign"
        else:
            stage = "judge"
            judge_score, label = await self.judge.judge(clipped, context)
            category = CATEGORY_BY_LABEL.get(label, "injection")
            score = judge_score

        decision = self.decide(score, judge_score)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)

        return Verdict(
            decision=decision,
            category=category if decision != "allow" else "benign",
            score=round(float(score), 4),
            stage=stage,
            latency_ms=latency_ms,
            excerpt=self.redact(clipped),
        )

    def decide(self, score: float, judge_score: Optional[float]) -> str:
        """Map the final score to allow, review or block."""
        final = judge_score if judge_score is not None else score
        if final >= self.settings.threshold_high:
            return "block"
        if final >= self.settings.threshold_low:
            return "review"
        return "allow"

    def redact(self, text: str) -> str:
        """Mask emails, phone numbers, card-like digits and API keys in the excerpt."""
        masked = EMAIL_RE.sub("[email]", text)
        masked = CARD_RE.sub("[card]", masked)
        masked = PHONE_RE.sub("[phone]", masked)
        masked = SECRET_RE.sub("[secret]", masked)
        masked = " ".join(masked.split())
        if len(masked) > 160:
            masked = masked[:157].rstrip() + "..."
        return masked
