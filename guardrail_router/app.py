"""FastAPI application: routes, dependencies and pipeline startup."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from . import __version__
from .classifier import PromptClassifier
from .config import Settings, get_settings
from .metrics import RATE_LIMITED, observe, render
from .pipeline import ScreeningPipeline
from .ratelimit import TenantLimiter
from .schemas import BatchScreenRequest, BatchScreenResponse, ScreenRequest, Verdict
from .slm import SlmJudge

logger = logging.getLogger("guardrail_router")


def build_pipeline(settings: Settings) -> ScreeningPipeline:
    classifier = PromptClassifier(settings.model_path)
    judge = SlmJudge(settings.slm_endpoint, settings.slm_timeout_ms)
    return ScreeningPipeline(classifier, judge, settings)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = resolved
        app.state.pipeline = build_pipeline(resolved)
        app.state.limiter = TenantLimiter(resolved.redis_url, resolved.rate_limit_per_minute)
        logger.info(
            "guardrail router %s ready, classifier loaded=%s",
            __version__,
            app.state.pipeline.classifier.is_loaded,
        )
        try:
            yield
        finally:
            await app.state.limiter.close()

    app = FastAPI(
        title="Guardrail Router",
        version=__version__,
        description="Screens untrusted text before it reaches a downstream model.",
        lifespan=lifespan,
    )

    def pipeline(request: Request) -> ScreeningPipeline:
        return request.app.state.pipeline

    def limiter(request: Request) -> TenantLimiter:
        return request.app.state.limiter

    async def enforce(tenant_id: str, bucket: TenantLimiter) -> None:
        if not await bucket.allow(tenant_id):
            RATE_LIMITED.labels(tenant_id=tenant_id).inc()
            raise HTTPException(
                status_code=429,
                detail="tenant rate limit exceeded, retry after the current window",
                headers={"Retry-After": "60"},
            )

    @app.post("/v1/screen", response_model=Verdict)
    async def screen(
        payload: ScreenRequest,
        pipe: ScreeningPipeline = Depends(pipeline),
        bucket: TenantLimiter = Depends(limiter),
    ) -> Verdict:
        await enforce(payload.tenant_id, bucket)
        verdict = await pipe.screen(payload.text, payload.context)
        observe(verdict)
        return verdict

    @app.post("/v1/screen/batch", response_model=BatchScreenResponse)
    async def screen_batch(
        payload: BatchScreenRequest,
        pipe: ScreeningPipeline = Depends(pipeline),
        bucket: TenantLimiter = Depends(limiter),
    ) -> BatchScreenResponse:
        await enforce(payload.tenant_id, bucket)
        verdicts = [await pipe.screen(item, payload.context) for item in payload.items]
        for verdict in verdicts:
            observe(verdict)
        return BatchScreenResponse.from_verdicts(payload.tenant_id, verdicts)

    @app.get("/healthz")
    async def healthz(request: Request) -> JSONResponse:
        pipe: ScreeningPipeline = request.app.state.pipeline
        return JSONResponse(
            {
                "status": "ok",
                "version": __version__,
                "classifier_loaded": pipe.classifier.is_loaded,
                "judge_mode": "remote" if pipe.judge.is_remote else "heuristic",
            }
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(content=render(), media_type="text/plain; version=0.0.4")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "path": request.url.path},
            headers=exc.headers,
        )

    return app


app = create_app()
