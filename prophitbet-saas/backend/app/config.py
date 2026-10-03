import sys
from pathlib import Path
from functools import lru_cache
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENVIRONMENT: str = "development"
    RATE_LIMIT_REQUESTS: int = Field(default=1000, ge=0)
    RATE_LIMIT_WINDOW_SECONDS: int = Field(default=86400, gt=0)

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://prophitbet:prophitbet@localhost:5432/prophitbet"
    DATABASE_URL_SYNC: str = "postgresql://prophitbet:prophitbet@localhost:5432/prophitbet"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Auth
    SECRET_KEY: str = "change-me-to-a-random-64-char-string"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"

    # OAuth
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GITHUB_CLIENT_ID: str = ""
    GITHUB_CLIENT_SECRET: str = ""

    # Stripe
    STRIPE_SECRET_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    STRIPE_PRICE_PRO: str = ""
    STRIPE_PRICE_ELITE: str = ""

    # S3 / MinIO
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "minioadmin"
    S3_SECRET_KEY: str = "minioadmin"
    S3_BUCKET: str = "prophitbet"

    # Frontend
    FRONTEND_URL: str = "http://localhost:3000"

    # ML core path (relative to project root or absolute)
    ML_CORE_PATH: str = "../../src"

    # MiroFish Swarm Intelligence Integration
    MIROFISH_API_URL: str = "http://mirofish:5001"
    MIROFISH_ENABLED: bool = True
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "https://api.openai.com/v1"
    LLM_MODEL: str = "gpt-4o-mini"
    ENSEMBLE_ML_WEIGHT: float = 0.50
    ENSEMBLE_SWARM_WEIGHT: float = 0.50

    # Qdrant Vector Database
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str = "http://qdrant:6333"

    # Optional factual football feed. Get a token from football-data.org.
    FOOTBALL_DATA_API_KEY: str = ""
    API_FOOTBALL_KEY: str = ""

    # Celery: skip re-syncing leagues already synced within N hours (0 = always full sync)
    SYNC_SKIP_IF_NEWER_THAN_HOURS: int = 24
    # Abort sync if less than this many MB free on DISK_CHECK_PATH (0 = disable)
    SYNC_MIN_FREE_DISK_MB: int = 256
    DISK_CHECK_PATH: str = "/"

    @model_validator(mode="after")
    def validate_deployment_secrets(self):
        """Reject public deployments that still use repository development secrets."""
        if self.ENVIRONMENT.lower() not in {"development", "dev", "test"}:
            insecure = []
            if self.SECRET_KEY == "change-me-to-a-random-64-char-string" or len(self.SECRET_KEY) < 32:
                insecure.append("SECRET_KEY")
            if self.S3_ACCESS_KEY == "minioadmin":
                insecure.append("S3_ACCESS_KEY")
            if self.S3_SECRET_KEY == "minioadmin":
                insecure.append("S3_SECRET_KEY")
            if insecure:
                raise ValueError(
                    "Secure values are required outside development for: " + ", ".join(insecure)
                )
        return self


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def setup_ml_path():
    """Add the original desktop app root to sys.path so we can import from src.*."""
    settings = get_settings()
    ml_path = Path(settings.ML_CORE_PATH)
    if not ml_path.is_absolute():
        ml_path = Path(__file__).resolve().parent.parent / ml_path
    ml_path = ml_path.resolve()
    # The parent of src/ is the project root â€” add that so `import src.*` works
    root = ml_path.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def get_storage_path() -> Path:
    """Resolve the path to the storage/ directory (contains leagues.json, etc.)."""
    settings = get_settings()
    ml_path = Path(settings.ML_CORE_PATH)
    if not ml_path.is_absolute():
        ml_path = Path(__file__).resolve().parent.parent / ml_path
    # storage/ is a sibling of src/
    storage = ml_path.resolve().parent / "storage"
    return storage
