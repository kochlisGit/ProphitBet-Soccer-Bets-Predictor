"""Process-independent, connection-lifetime guard for prediction generation."""
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

PREDICTION_LOCK = 718311
PREFIX = 'prophitbet-predict:'


@contextmanager
def prediction_run_guard(task_id):
    # Dedicated connection: model-by-model commits cannot release this lock.
    # PostgreSQL releases it even when the worker process is killed.
    from backend.app.config import get_settings
    engine = create_engine(get_settings().DATABASE_URL_SYNC, poolclass=NullPool)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT set_config('application_name', :name, false)"),
                               {'name': PREFIX + str(task_id)})
            acquired = connection.scalar(text('SELECT pg_try_advisory_lock(:key)'), {'key': PREDICTION_LOCK})
            owner = None
            if not acquired:
                owner = connection.scalar(text('''SELECT a.application_name
                    FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid
                    WHERE l.locktype = 'advisory' AND l.classid = 0 AND l.objid = :key
                    AND l.objsubid = 1 AND l.granted LIMIT 1'''), {'key': PREDICTION_LOCK})
                owner = owner.removeprefix(PREFIX) if owner else None
            try:
                yield bool(acquired), owner
            finally:
                if acquired:
                    connection.execute(text('SELECT pg_advisory_unlock(:key)'), {'key': PREDICTION_LOCK})
    finally:
        engine.dispose()
