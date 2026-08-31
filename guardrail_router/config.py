"""Runtime settings, read from the environment once per process."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Everything the service needs to start, with defaults that work locally."""

    model_config = SettingsConfigDict(env_prefix="GUARDRAIL_", env_file=".env", extra="ignore")

    model_path: str | None = Field(
        default=None,
        description="Path to the exported ONNX classifier; hashing fallback when unset.",
    )
    slm_endpoint: str | None = Field(
        default=None,
        description="Base URL of the small language model judge; heuristic judge when unset.",
    )
    threshold_low: float = Field(default=0.35, ge=0.0, le=1.0)
    threshold_high: float = Field(default=0.80, ge=0.0, le=1.0)
    redis_url: str | None = Field(default=None, description="Redis URL for the tenant limiter.")
    rate_limit_per_minute: int = Field(default=600, ge=1)
    slm_timeout_ms: int = Field(default=250, ge=10, le=5000)
    max_text_chars: int = Field(default=8000, ge=64, le=200_000)
    excerpt_chars: int = Field(default=120, ge=20, le=400)

    def model_post_init(self, __context: object) -> None:
        if self.threshold_low >= self.threshold_high:
            raise ValueError("threshold_low must be below threshold_high")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
