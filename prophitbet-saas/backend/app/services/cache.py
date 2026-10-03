"""
Redis caching service for ProphitBet SaaS.
Provides caching for league metadata, fixtures, and predictions.
"""
import json
import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast

import redis
from pydantic import BaseModel

from backend.app.config import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Cache TTL constants (in seconds)
CACHE_TTL_LEAGUE_METADATA = 3600 * 24  # 24 hours - league info rarely changes
CACHE_TTL_RECENT_FIXTURES = 3600 * 2  # 2 hours - fixtures change periodically
CACHE_TTL_TEAM_STATS = 3600 * 6  # 6 hours - team stats change slowly
CACHE_TTL_API_RESPONSE = 300  # 5 minutes - API responses

# Cache key prefixes
KEY_PREFIX_LEAGUE = "league:"
# Versioned so legacy, non-provenanced fixture responses are never served.
KEY_PREFIX_FIXTURES = "fixtures:v2:"
KEY_PREFIX_PREDICTIONS = "predictions:"
KEY_PREFIX_TEAM_STATS = "team_stats:"


class CacheService:
    """Redis cache service with type-safe operations."""
    
    def __init__(self):
        self._client: Optional[redis.Redis] = None
    
    def _get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            settings = get_settings()
            self._client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        return self._client
    
    def get(self, key: str) -> Optional[str]:
        """Get raw string value from cache."""
        try:
            return self._get_client().get(key)
        except Exception as e:
            logger.warning(f"Cache get failed for {key}: {e}")
            return None
    
    def set(self, key: str, value: str, ttl: int) -> bool:
        """Set string value in cache with TTL."""
        try:
            return self._get_client().setex(key, ttl, value)
        except Exception as e:
            logger.warning(f"Cache set failed for {key}: {e}")
            return False
    
    def get_json(self, key: str, model: type[BaseModel]) -> Optional[BaseModel]:
        """Get and parse JSON value as Pydantic model."""
        data = self.get(key)
        if data:
            try:
                return model.model_validate_json(data)
            except Exception as e:
                logger.warning(f"Cache JSON parse failed for {key}: {e}")
        return None
    
    def set_json(self, key: str, value: BaseModel, ttl: int) -> bool:
        """Serialize and cache Pydantic model as JSON."""
        try:
            return self.set(key, value.model_dump_json(), ttl)
        except Exception as e:
            logger.warning(f"Cache JSON serialize failed for {key}: {e}")
            return False
    
    def get_list(self, key: str, model: type[BaseModel]) -> Optional[list[BaseModel]]:
        """Get and parse JSON array as list of Pydantic models."""
        data = self.get(key)
        if data:
            try:
                items = json.loads(data)
                return [model.model_validate(item) for item in items]
            except Exception as e:
                logger.warning(f"Cache list parse failed for {key}: {e}")
        return None
    
    def set_list(self, key: str, value: list[BaseModel], ttl: int) -> bool:
        """Serialize and cache list of Pydantic models as JSON array."""
        try:
            return self.set(key, json.dumps([item.model_dump() for item in value]), ttl)
        except Exception as e:
            logger.warning(f"Cache list serialize failed for {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a cache key."""
        try:
            return bool(self._get_client().delete(key))
        except Exception as e:
            logger.warning(f"Cache delete failed for {key}: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            client = self._get_client()
            keys = client.keys(pattern)
            if keys:
                return client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache delete pattern failed for {pattern}: {e}")
            return 0
    
    def health_check(self) -> bool:
        """Check if Redis is available."""
        try:
            return self._get_client().ping()
        except Exception:
            return False


# Global cache instance
_cache: Optional[CacheService] = None


def get_cache() -> CacheService:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = CacheService()
    return _cache


def cached(key_prefix: str, ttl: int, key_builder: Optional[Callable[..., str]] = None):
    """
    Decorator for caching async function results.
    
    Args:
        key_prefix: Prefix for cache key
        ttl: Time-to-live in seconds
        key_builder: Optional function to build cache key from function args
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = get_cache()
            
            # Build cache key
            if key_builder:
                cache_key = f"{key_prefix}{key_builder(*args, **kwargs)}"
            else:
                # Default: use function name and args
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = f"{key_prefix}{':'.join(key_parts)}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return json.loads(cached_result)
            
            # Execute function
            logger.debug(f"Cache miss: {cache_key}")
            result = await func(*args, **kwargs)
            
            # Cache result
            try:
                cache.set(cache_key, json.dumps(result, default=str), ttl)
            except Exception as e:
                logger.warning(f"Failed to cache result for {cache_key}: {e}")
            
            return result
        return wrapper
    return decorator


# Convenience functions for common cache operations
def cache_league_metadata(league_id: int, data: dict) -> bool:
    """Cache league metadata."""
    cache = get_cache()
    return cache.set(
        f"{KEY_PREFIX_LEAGUE}metadata:{league_id}",
        json.dumps(data, default=str),
        CACHE_TTL_LEAGUE_METADATA
    )


def get_cached_league_metadata(league_id: int) -> Optional[dict]:
    """Get cached league metadata."""
    cache = get_cache()
    data = cache.get(f"{KEY_PREFIX_LEAGUE}metadata:{league_id}")
    if data:
        return json.loads(data)
    return None


def cache_league_list(data: list[dict]) -> bool:
    """Cache league list."""
    cache = get_cache()
    return cache.set(
        f"{KEY_PREFIX_LEAGUE}list",
        json.dumps(data, default=str),
        CACHE_TTL_LEAGUE_METADATA
    )


def get_cached_league_list() -> Optional[list[dict]]:
    """Get cached league list."""
    cache = get_cache()
    data = cache.get(f"{KEY_PREFIX_LEAGUE}list")
    if data:
        return json.loads(data)
    return None


def cache_fixtures(league_id: int, data: list[dict]) -> bool:
    """Cache fixtures for a league."""
    cache = get_cache()
    return cache.set(
        f"{KEY_PREFIX_FIXTURES}{league_id}",
        json.dumps(data, default=str),
        CACHE_TTL_RECENT_FIXTURES
    )


def get_cached_fixtures(league_id: int) -> Optional[list[dict]]:
    """Get cached fixtures for a league."""
    cache = get_cache()
    data = cache.get(f"{KEY_PREFIX_FIXTURES}{league_id}")
    if data:
        return json.loads(data)
    return None


def invalidate_fixtures_cache(league_id: Optional[int] = None) -> int:
    """Invalidate fixtures cache for a league or all leagues."""
    cache = get_cache()
    if league_id:
        return cache.delete(f"{KEY_PREFIX_FIXTURES}{league_id}")
    return cache.delete_pattern(f"{KEY_PREFIX_FIXTURES}*")


def cache_predictions(league_id: int, data: list[dict]) -> bool:
    """Cache predictions for a league."""
    cache = get_cache()
    return cache.set(
        f"{KEY_PREFIX_PREDICTIONS}{league_id}",
        json.dumps(data, default=str),
        CACHE_TTL_API_RESPONSE
    )


def get_cached_predictions(league_id: int) -> Optional[list[dict]]:
    """Get cached predictions for a league."""
    cache = get_cache()
    data = cache.get(f"{KEY_PREFIX_PREDICTIONS}{league_id}")
    if data:
        return json.loads(data)
    return None


def invalidate_predictions_cache(league_id: Optional[int] = None) -> int:
    """Invalidate predictions cache for a league or all leagues."""
    cache = get_cache()
    if league_id:
        return cache.delete(f"{KEY_PREFIX_PREDICTIONS}{league_id}")
    return cache.delete_pattern(f"{KEY_PREFIX_PREDICTIONS}*")
