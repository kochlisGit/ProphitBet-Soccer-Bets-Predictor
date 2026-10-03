"""Resolve competition identity without substrings or deployment-specific IDs."""
import re
from fastapi import HTTPException
from sqlalchemy import select
from backend.app.db.models import League


def normalize(value):
    return re.sub(r'[\s_-]+', '', value.lower())


ALIASES = {
    'premierleague': ('england', 'premierleague'), 'epl': ('england', 'premierleague'),
    'championship': ('england', 'championship'), 'seriea': ('italy', 'seriea'),
    'psl': ('southafrica', 'betwaypremiership'), 'betwaypremiership': ('southafrica', 'betwaypremiership'),
    'premiership': ('scotland', 'premiership'), 'scottishpremiership': ('scotland', 'premiership'),
    'russianpremierleague': ('russia', 'premierleague'), 'laliga': ('spain', 'laliga'),
    'bundesliga': ('germany', 'bundesliga1'), 'bundesliga1': ('germany', 'bundesliga1'),
    'ligue1': ('france', 'ligue1'),
}


async def resolve_league_id(db, value):
    if re.fullmatch(r'league:\d+', value):
        return int(value.split(':')[1])
    rows = (await db.execute(select(League.id, League.country, League.name))).all()
    target = ALIASES.get(normalize(value))
    ids = [row.id for row in rows if (normalize(row.country), normalize(row.name)) == target] if target else [
        row.id for row in rows if normalize(row.name) == normalize(value)]
    if len(ids) != 1:
        raise HTTPException(422, 'Unknown or ambiguous league; select an explicit league ID.')
    return ids[0]
