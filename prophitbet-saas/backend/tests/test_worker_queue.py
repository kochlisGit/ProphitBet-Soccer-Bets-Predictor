from contextlib import contextmanager
import unittest
from unittest.mock import Mock, patch

from celery.exceptions import Ignore
from backend.app.workers.celery_app import celery_app
from backend.app.workers.predict import generate_daily_predictions_task


@contextmanager
def guard(acquired, owner):
    yield acquired, owner


class WorkerQueueTests(unittest.TestCase):
    def test_data_tasks_route_away_from_ml(self):
        router = celery_app.amqp.router
        for task in ('sync_all_leagues_task', 'sync_league_task'):
            route = router.route({}, 'backend.app.workers.data_sync.' + task)
            self.assertEqual(route['queue'].name, 'data_sync')
        self.assertEqual(router.route({}, 'backend.app.workers.predict.generate_daily_predictions_task')['queue'].name, 'celery')
        self.assertEqual(celery_app.conf.broker_transport_options['visibility_timeout'], 86400)
        self.assertTrue(celery_app.conf.worker_deduplicate_successful_tasks)

    def run_guarded(self, acquired, owner, task_id='new'):
        task = generate_daily_predictions_task
        task.push_request(id=task_id)
        try:
            with patch('backend.app.workers.task_guard.prediction_run_guard', return_value=guard(acquired, owner)), \
                 patch('backend.app.workers.predict._generate_daily_predictions', return_value={'total_predictions': 2}) as work:
                result = task.run()
                return result, work.call_count
        finally:
            task.pop_request()

    def test_lock_owner_runs(self):
        result, calls = self.run_guarded(True, None)
        self.assertEqual(calls, 1)
        self.assertEqual(result['total_predictions'], 2)

    def test_overlapping_different_job_does_not_wait_or_work(self):
        result, calls = self.run_guarded(False, 'original')
        self.assertEqual(calls, 0)
        self.assertEqual(result['active_job_id'], 'original')

    def test_same_id_redelivery_cannot_overwrite_active_result(self):
        with self.assertRaises(Ignore):
            self.run_guarded(False, 'same', 'same')

    def test_fixture_horizon_is_bounded(self):
        # Assert the SQL-producing implementation retains both time boundaries.
        import inspect
        from backend.app.workers.predict import _generate_daily_predictions
        source = inspect.getsource(_generate_daily_predictions)
        self.assertIn('Fixture.match_date >= now', source)
        self.assertIn('Fixture.match_date <= now + timedelta(days=7)', source)
