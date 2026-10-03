import logging
import hashlib
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from starlette.middleware.base import BaseHTTPMiddleware

from backend.app.config import get_settings

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce a shared fixed-window request limit using Redis."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        settings = get_settings()
        if request.url.path not in {"/health", "/metrics"} and settings.RATE_LIMIT_REQUESTS > 0:
            identity = request.headers.get("authorization") or (
                request.client.host if request.client else "unknown"
            )
            identity_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            window = settings.RATE_LIMIT_WINDOW_SECONDS
            bucket = int(time.time()) // window
            key = f"rate-limit:{identity_hash}:{bucket}"
            redis = Redis.from_url(settings.REDIS_URL, decode_responses=True)
            try:
                count = await redis.incr(key)
                if count == 1:
                    await redis.expire(key, window + 1)
                if count > settings.RATE_LIMIT_REQUESTS:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Rate limit exceeded"},
                        headers={"Retry-After": str(window - (int(time.time()) % window))},
                    )
            except Exception:
                logger.exception("Redis rate-limit check failed; allowing request")
            finally:
                await redis.aclose()

        response = await call_next(request)
        elapsed = time.time() - start

        response.headers["X-Response-Time"] = f"{elapsed:.3f}s"
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start

        if not request.url.path.startswith("/health"):
            logger.info(
                f"{request.method} {request.url.path} -> {response.status_code} ({elapsed:.3f}s)"
            )
        return response
