"""Read-only slip suggestions from existing house-model predictions, not new ML logic."""
import math
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import noload

from backend.app.db.models import Fixture, Prediction, TrainedModel
from backend.app.services.betting_integrity import MARKETS, valid_odds
from backend.app.services.sa_odds_service import sa_odds_service
from backend.app.services.prediction_markets import market_selection
from src.preprocessing.utils.target import target_labels, parse_target


def rank_slip(rows, quotes, *, bookmaker, legs, min_probability, strategy, draft, min_ev, market_type="result"):
    candidates, seen = [], set()
    for pred, fixture in rows:  # newest prediction first; never cherry-pick older model runs
        market = parse_target(getattr(pred, "market_type", None) or "result").value
        if market_type != "all" and market != parse_target(market_type).value:
            continue
        labels = target_labels(market)
        event = (market, fixture.league_id, fixture.home_team, fixture.away_team, fixture.match_date)
        if event in seen:
            continue
        seen.add(event)
        probabilities = pred.probabilities or {}
        if not isinstance(probabilities, dict) or not all(
            isinstance(probabilities.get(k), (int, float)) and not isinstance(probabilities[k], bool)
            and math.isfinite(probabilities[k]) and 0 <= probabilities[k] <= 1 for k in labels
        ) or abs(sum(probabilities[k] for k in labels) - 1) > .01:
            continue
        quote = quotes.get(str(fixture.id), {}).get(bookmaker.lower())
        picks = []
        for pick in labels:
            probability = probabilities[pick]
            if probability < min_probability:
                continue
            selection = market_selection(market, pick)
            price = quote.get(MARKETS.get(selection)) if quote and not draft else None
            if not draft and (not valid_odds(price) or price > 10000):
                continue
            edge = (probability * price - 1) * 100 if price else None
            if strategy == 'value' and (edge is None or edge < min_ev):
                continue
            picks.append(dict(fixture_id=str(fixture.id), prediction_id=str(pred.id),
                model_id=str(pred.model_id), match_title=f'{fixture.home_team} vs {fixture.away_team}',
                match_date=fixture.match_date.isoformat(), selection=selection, market_type=market, bookmaker=bookmaker,
                odds_taken=price, quote_id=quote['quote_id'] if price else None,
                source_url=quote['source_url'] if price else fixture.source_url,
                quoted_at=quote['scraped_at'] if price else None,
                model_probability=probability, model_ev_pct=round(edge, 2) if edge is not None else None,
                prediction_created_at=pred.created_at.isoformat(),
                explanation=f'House model assigns {probability:.1%} to {selection}.' +
                    (f' At observed odds {price:g}, estimated EV is {edge:.1f}%.' if price else ' No bookmaker price is attached.'),
                teams=(fixture.home_team.strip().casefold(), fixture.away_team.strip().casefold())))
        if picks:
            candidates.append(max(picks, key=lambda p: p['model_ev_pct'] if strategy == 'value' else p['model_probability']))
    candidates.sort(key=lambda p: (-(p['model_ev_pct'] if strategy == 'value' else p['model_probability']), p['match_date'], p['fixture_id']))
    chosen, teams = [], set()
    for pick in candidates:
        if teams.intersection(pick['teams']):
            continue
        if not draft and math.prod(p['odds_taken'] for p in chosen + [pick]) > 1000000:
            continue
        teams.update(pick.pop('teams'))
        chosen.append(pick)
        if len(chosen) == legs:
            break
    priced = bool(chosen) and not draft
    return dict(status=('draft' if draft else 'ready') if len(chosen) == legs else 'insufficient_data',
        requested_legs=legs, legs=chosen, bookmaker=bookmaker, currency='ZAR',
        can_save=priced and len(chosen) == legs,
        combined_odds=math.prod(p['odds_taken'] for p in chosen) if priced else None,
        estimated_win_probability=math.prod(p['model_probability'] for p in chosen) if chosen else None,
        message=(f'Selected {len(chosen)} of {legs} requested legs. ' +
                 ('Prediction-only draft: add actual bookmaker odds separately; no payout is estimated.' if draft else
                  'Review each selection. Saving rechecks quote freshness and prices. Missing quotes are never invented.')),
        warning='Model estimates are not guarantees or measured accuracy. Combined probability assumes independence; outcomes can be correlated. More legs increase the chance of losing the entire stake. No bet is placed automatically.')


async def generate_ai_slip(db, **options):
    now = datetime.now(timezone.utc)
    stmt = select(Prediction, Fixture).join(TrainedModel, Prediction.model_id == TrainedModel.id).join(
        Fixture, (Fixture.league_id == Prediction.league_id) & (Fixture.home_team == Prediction.home_team)
        & (Fixture.away_team == Prediction.away_team) & (Fixture.match_date == Prediction.match_date)
    ).where(TrainedModel.is_house_model.is_(True), TrainedModel.league_id == Prediction.league_id,
        Prediction.actual_result.is_(None),
        Prediction.created_at.between(now - timedelta(hours=48), now),
        Fixture.match_date > now, Fixture.match_date <= now + timedelta(days=7),
        Fixture.is_current.is_(True), Fixture.source_url.isnot(None), Fixture.source_url != '',
        Fixture.fetched_at.between(now - timedelta(hours=48), now)
    ).options(noload(Prediction.mirofish_simulation)).order_by(Prediction.created_at.desc()).limit(1000)
    market = options.get('market_type', 'result')
    if market != 'all':
        normalized = parse_target(market).value
        stmt = stmt.where(TrainedModel.target_type.in_([normalized, 'over_under'] if normalized == 'over-under' else [normalized]))
    rows = (await db.execute(stmt)).all()
    matches = [] if options['draft'] else await sa_odds_service.get_paired_odds_comparison(db, limit=100)
    return rank_slip(rows, {m['fixture_id']: m['bookmakers'] for m in matches}, **options)
