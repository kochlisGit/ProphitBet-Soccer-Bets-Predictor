import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def disk_free_mb(path: str) -> float:
    try:
        usage = shutil.disk_usage(Path(path))
        return usage.free / (1024 * 1024)
    except OSError as e:
        logger.warning("disk_usage failed for %s: %s", path, e)
        return float("inf")


def assert_enough_disk_for_sync(min_free_mb: int, path: str) -> tuple[bool, float]:
    if min_free_mb <= 0:
        return True, disk_free_mb(path)
    free = disk_free_mb(path)
    ok = free >= float(min_free_mb)
    if not ok:
        logger.error("Insufficient disk space: %.1f MB free on %s (need %s MB)", free, path, min_free_mb)
    return ok, free
