"""Source-evidence and dataset storage checks. Never equate missing with healthy."""
from prometheus_client.core import GaugeMetricFamily
from sqlalchemy import text
from backend.app.services.league_service import dataset_objects_status


def quality_metrics(engine):
    with engine.connect() as db:
        db.execute(text('SET TRANSACTION READ ONLY'))
        issues = dict(db.execute(text("""SELECT 'result_evidence_missing', count(*) FROM predictions
            WHERE actual_result IS NOT NULL AND (actual_score IS NULL OR result_source IS NULL OR result_verified_at IS NULL)
            UNION ALL SELECT 'future_settled_result', count(*) FROM predictions WHERE actual_result IS NOT NULL AND match_date > now()
            UNION ALL SELECT 'fixture_evidence_missing', count(*) FROM fixtures WHERE is_current AND match_date > now()
                AND (source_url IS NULL OR fetched_at IS NULL OR fetched_at > now() OR fetched_at < now()-interval '48 hours')
            UNION ALL SELECT 'duplicate_current_fixture', count(*) FROM
                (SELECT league_id, lower(trim(home_team)), lower(trim(away_team)), match_date FROM fixtures WHERE is_current
                 GROUP BY 1,2,3,4 HAVING count(*)>1) duplicates""")).all())
        rows = db.execute(text('''SELECT l.id, d.file_path, coalesce(d.row_count,0) FROM leagues l
            LEFT JOIN LATERAL (SELECT file_path,row_count FROM league_datasets WHERE league_id=l.id
                ORDER BY created_at DESC LIMIT 1) d ON true WHERE l.is_active''')).all()
    integrity = GaugeMetricFamily('prophitbet_data_integrity_issues', 'Evidence/consistency defects; not a guarantee of factual correctness', labels=['check'])
    for check, count in issues.items():
        integrity.add_metric([check], count)
    objects = dataset_objects_status([row.file_path for row in rows])
    present = GaugeMetricFamily('prophitbet_dataset_object_present', 'Latest CSV object exists and is nonempty; absent on storage error', labels=['league_id'])
    checked = GaugeMetricFamily('prophitbet_dataset_storage_check_up', 'Whether latest CSV existence could be checked', labels=['league_id'])
    counts = GaugeMetricFamily('prophitbet_dataset_rows', 'Latest accepted CSV row count in DB, not number of dataset records', labels=['league_id'])
    for league, key, count in rows:
        checked.add_metric([str(league)], int(objects[key] is not None))
        if objects[key] is not None:
            present.add_metric([str(league)], objects[key])
        counts.add_metric([str(league)], count)
    return [integrity, present, checked, counts]
