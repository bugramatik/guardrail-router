"""Per-tenant token bucket backed by Redis, with an in-process fallback for local runs."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

try:  # redis is optional so the service can run without it
    from redis.asyncio import Redis
    from redis.exceptions import RedisError

    _REDIS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal installs
    Redis = None  # type: ignore[assignment]
    RedisError = Exception  # type: ignore[assignment,misc]
    _REDIS_AVAILABLE = False


# Refill the bucket every second; capacity is the per-minute budget.
_REFILL_INTERVAL = 1.0
_KEY_TTL_SECONDS = 120


@dataclass
class _Bucket:
    tokens: float
    updated_at: float = field(default_factory=time.monotonic)


class TenantLimiter:
    """Token bucket per tenant.

    Each tenant gets ``per_minute`` tokens, refilled at ``per_minute / 60`` per
    second. Redis holds the shared state so every replica sees the same budget;
    when Redis is not reachable the limiter falls back to a process-local bucket
    and keeps serving instead of failing closed.
    """

    def __init__(self, redis_url: str | None, per_minute: int) -> None:
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.per_minute = per_minute
        self._rate = per_minute / 60.0
        self._redis_url = redis_url
        self._redis: "Redis | None" = None
        self._local: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()
        self._redis_broken = False

    async def _client(self) -> "Redis | None":
        if not _REDIS_AVAILABLE or not self._redis_url or self._redis_broken:
            return None
        if self._redis is None:
            self._redis = Redis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    async def allow(self, tenant_id: str) -> bool:
        """Consume one token for the tenant. False means the caller gets a 429."""
        client = await self._client()
        if client is not None:
            try:
                return await self._allow_redis(client, tenant_id)
            except (RedisError, OSError):
                # One bad connection should not take the service down; degrade
                # to the local bucket for the rest of this process's life.
                self._redis_broken = True
        return await self._allow_local(tenant_id)

    async def remaining(self, tenant_id: str) -> int:
        """Tokens left for the tenant, rounded down, for the rate-limit headers."""
        client = await self._client()
        if client is not None:
            try:
                raw = await client.get(self._key(tenant_id))
                if raw is None:
                    return self.per_minute
                tokens, updated_at = (float(part) for part in raw.split(":"))
                elapsed = max(0.0, time.time() - updated_at)
                return max(0, int(tokens + elapsed * self._rate))
            except (RedisError, OSError, ValueError):
                self._redis_broken = True
        async with self._lock:
            bucket = self._local.get(tenant_id)
            if bucket is None:
                return self.per_minute
            self._refill(bucket)
            return max(0, int(bucket.tokens))

    async def _allow_redis(self, client: "Redis", tenant_id: str) -> bool:
        key = self._key(tenant_id)
        now = time.time()
        raw = await client.get(key)
        if raw is None:
            tokens, updated_at = float(self.per_minute), now
        else:
            tokens, updated_at = (float(part) for part in raw.split(":"))
            tokens = min(float(self.per_minute), tokens + max(0.0, now - updated_at) * self._rate)
        allowed = tokens >= 1.0
        if allowed:
            tokens -= 1.0
        await client.set(key, f"{tokens:.4f}:{now:.4f}", ex=_KEY_TTL_SECONDS)
        return allowed

    async def _allow_local(self, tenant_id: str) -> bool:
        async with self._lock:
            bucket = self._local.get(tenant_id)
            if bucket is None:
                bucket = _Bucket(tokens=float(self.per_minute))
                self._local[tenant_id] = bucket
            self._refill(bucket)
            if bucket.tokens < 1.0:
                return False
            bucket.tokens -= 1.0
            return True

    def _refill(self, bucket: _Bucket) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - bucket.updated_at)
        bucket.tokens = min(float(self.per_minute), bucket.tokens + elapsed * self._rate)
        bucket.updated_at = now

    @staticmethod
    def _key(tenant_id: str) -> str:
        return f"guardrail:rl:{tenant_id}"

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
