import io
import logging
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import League, LeagueDataset

logger = logging.getLogger(__name__)


def ensure_s3_bucket_sync() -> None:
    """Create the configured S3/MinIO bucket if missing (for API startup and Celery workers)."""
    import boto3
    from backend.app.config import get_settings

    settings = get_settings()
    s3 = boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
    )
    try:
        s3.head_bucket(Bucket=settings.S3_BUCKET)
    except Exception:
        s3.create_bucket(Bucket=settings.S3_BUCKET)
        logger.info("Created S3 bucket: %s", settings.S3_BUCKET)


async def get_league_dataframe(league_id: int, db: AsyncSession) -> Optional[pd.DataFrame]:
    """Load the latest dataset CSV for a league from S3 or local storage."""
    result = await db.execute(
        select(LeagueDataset)
        .where(LeagueDataset.league_id == league_id)
        .order_by(LeagueDataset.created_at.desc())
        .limit(1)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        return None

    try:
        return await _load_csv_from_s3(dataset.file_path)
    except Exception:
        logger.warning(f"S3 load failed for {dataset.file_path}, trying local fallback")
        return _load_csv_local(dataset.file_path)


def dataset_objects_status(keys):
    """Read-only existence checks. Provider errors are unknown, never missing."""
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    from concurrent.futures import ThreadPoolExecutor
    from backend.app.config import get_settings
    settings = get_settings()
    s3 = boto3.client('s3', endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY, aws_secret_access_key=settings.S3_SECRET_KEY,
        config=Config(connect_timeout=3, read_timeout=3, retries={'max_attempts': 0}))
    def check(key):
        if not key:
            return key, 0
        try:
            result = s3.head_object(Bucket=settings.S3_BUCKET, Key=key)
            return key, int(result.get('ContentLength', 0) > 0)
        except ClientError as exc:
            return key, 0 if exc.response.get('Error', {}).get('Code') in ('404', 'NoSuchKey', 'NotFound') else None
        except Exception:
            return key, None
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(check, set(keys)))


async def _load_csv_from_s3(key: str) -> pd.DataFrame:
    import boto3
    from backend.app.config import get_settings
    settings = get_settings()

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
    )
    response = s3.get_object(Bucket=settings.S3_BUCKET, Key=key)
    body = response["Body"].read()
    return pd.read_csv(io.BytesIO(body))


def _load_csv_local(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


async def upload_csv_to_s3(df: pd.DataFrame, key: str) -> str:
    import boto3
    from backend.app.config import get_settings
    settings = get_settings()

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
    )

    try:
        s3.head_bucket(Bucket=settings.S3_BUCKET)
    except Exception:
        s3.create_bucket(Bucket=settings.S3_BUCKET)

    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=settings.S3_BUCKET, Key=key, Body=buf.read(), ContentType="text/csv")
    return key


async def upload_model_to_s3(model_bytes: bytes, key: str) -> str:
    import boto3
    from backend.app.config import get_settings
    settings = get_settings()

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
    )
    try:
        s3.head_bucket(Bucket=settings.S3_BUCKET)
    except Exception:
        s3.create_bucket(Bucket=settings.S3_BUCKET)

    s3.put_object(Bucket=settings.S3_BUCKET, Key=key, Body=model_bytes, ContentType="application/octet-stream")
    return key


async def delete_model_from_s3(key: str) -> None:
    import boto3
    from backend.app.config import get_settings
    settings = get_settings()

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
    )
    s3.delete_object(Bucket=settings.S3_BUCKET, Key=key)


async def download_model_from_s3(key: str) -> bytes:
    return download_model_from_s3_sync(key)


def _load_csv_from_s3_sync(key: str) -> pd.DataFrame:
    import boto3
    from backend.app.config import get_settings
    settings = get_settings()

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
    )
    response = s3.get_object(Bucket=settings.S3_BUCKET, Key=key)
    body = response["Body"].read()
    return pd.read_csv(io.BytesIO(body))


def download_model_from_s3_sync(key: str) -> bytes:
    import boto3
    from backend.app.config import get_settings
    settings = get_settings()

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
    )
    response = s3.get_object(Bucket=settings.S3_BUCKET, Key=key)
    return response["Body"].read()

