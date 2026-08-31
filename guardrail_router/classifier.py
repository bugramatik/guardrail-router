"""Fast first-stage classifier for untrusted text.

The exported model is a linear model (TF-IDF features into a logistic
regression) stored as ONNX. When the file is missing or ONNX Runtime is not
installed, a deterministic hashing scorer keeps the service usable so local
development and tests do not depend on a model artifact.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Phrases that a linear model trained on prompt-injection data picks up quickly.
# The fallback uses them so its scores stay in the same range as the real model.
_SUSPICIOUS_PHRASES = (
    "ignore previous",
    "ignore all previous",
    "disregard the above",
    "disregard your instructions",
    "system prompt",
    "you are now",
    "act as",
    "jailbreak",
    "developer mode",
    "reveal your",
    "print your instructions",
    "repeat the text above",
    "send the following",
    "api key",
    "password",
    "credentials",
    "base64",
    "do anything now",
)


class PromptClassifier:
    """Scores a piece of text between 0 and 1 for prompt-injection risk."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._session = None
        self._input_name: str | None = None
        self._load_error: str | None = None
        if model_path:
            self._load(model_path)

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    def _load(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            self._load_error = f"model file not found: {model_path}"
            logger.warning("classifier falling back to hashing scorer: %s", self._load_error)
            return
        try:
            import onnxruntime as ort  # imported lazily so the fallback works without it
        except ImportError:
            self._load_error = "onnxruntime is not installed"
            logger.warning("classifier falling back to hashing scorer: %s", self._load_error)
            return
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
        except Exception as exc:  # a corrupt artifact must not take the service down
            self._load_error = f"could not load model: {exc}"
            self._session = None
            logger.warning("classifier falling back to hashing scorer: %s", self._load_error)

    def score(self, text: str) -> float:
        """Return a risk score in [0, 1]; higher means more likely an injection."""
        if not text or not text.strip():
            return 0.0
        if self._session is not None and self._input_name is not None:
            try:
                return self._score_onnx(text)
            except Exception as exc:
                logger.warning("onnx inference failed, using fallback: %s", exc)
        return self._fallback_score(text)

    def _score_onnx(self, text: str) -> float:
        import numpy as np

        tokens = _TOKEN_RE.findall(text.lower())[:256]
        ids = np.array([[self._token_id(t) for t in tokens] or [0]], dtype=np.int64)
        outputs = self._session.run(None, {self._input_name: ids})
        logit = float(np.asarray(outputs[0]).reshape(-1)[0])
        return _sigmoid(logit)

    @staticmethod
    def _token_id(token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, "big") % 30000 + 1

    def _fallback_score(self, text: str) -> float:
        lowered = text.lower()
        hits = sum(1 for phrase in _SUSPICIOUS_PHRASES if phrase in lowered)
        tokens = _TOKEN_RE.findall(lowered)
        if not tokens:
            return 0.0
        # Long pasted blocks with instruction-like verbs are the common shape of
        # an indirect injection, so length contributes a small amount.
        length_term = min(len(tokens) / 400.0, 1.0) * 0.15
        imperative = 0.1 if any(t in {"ignore", "disregard", "override", "forget"} for t in tokens) else 0.0
        raw = 0.12 * hits + length_term + imperative
        return round(min(max(raw, 0.0), 0.99), 4)



def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
