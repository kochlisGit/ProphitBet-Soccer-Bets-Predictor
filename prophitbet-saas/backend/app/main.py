import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import get_settings, setup_ml_path

settings = get_settings()
setup_ml_path()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("prophitbet")

app = FastAPI(
    title="ProphitBet API",
    description="AI-powered soccer predictions SaaS platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

from backend.app.middleware import RequestLoggingMiddleware, RateLimitMiddleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator(excluded_handlers=["/metrics", "/health"]).instrument(app).expose(app, include_in_schema=False)
except Exception as e:
    logger.warning("Prometheus instrumentator initialization skipped: %s", e)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3005",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3005",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|percpc|.*\.ts\.net|100\.\d+\.\d+\.\d+)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.on_event("startup")
async def startup():
    logger.info("ProphitBet API starting up...")
    try:
        from backend.app.services.league_service import ensure_s3_bucket_sync

        ensure_s3_bucket_sync()
    except Exception as e:
        logger.warning(f"Could not ensure S3 bucket exists: {e}")

    from backend.app.services.vector_rag_service import QdrantVectorRAGService

    for attempt in range(10):
        if await asyncio.to_thread(QdrantVectorRAGService.ensure_collection):
            seeded = await asyncio.to_thread(QdrantVectorRAGService.seed_tactical_intel_dossiers)
            logger.info("Qdrant ready; seeded %s tactical profiles", seeded)
            break
        if attempt < 9:
            await asyncio.sleep(1)
    else:
        logger.warning("Qdrant was not ready after 10 attempts; vector RAG will use fallbacks")

@app.on_event("shutdown")
async def shutdown():
    from backend.app.db.session import engine
    await engine.dispose()
    logger.info("ProphitBet API shut down")


from backend.app.api import api_router  # noqa: E402
app.include_router(api_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "prophitbet-api"}
