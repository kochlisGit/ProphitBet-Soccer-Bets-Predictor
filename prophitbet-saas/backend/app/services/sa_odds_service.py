"""Observed South African prices only. Missing quotes are never synthesized."""
import math
from datetime import datetime, timezone
from sqlalchemy import select
from backend.app.db.models import SABookmakerOdds, Fixture, Prediction
from backend.app.services.betting_integrity import MARKETS, observed_quote, valid_odds, money, QUOTE_MAX_AGE
from backend.app.services.data_integrity import publishable_prediction


def dutch_stakes(prices, bankroll):
    """A price discrepancy is not a guarantee; include cent-rounding losses."""
    if not prices or not all(valid_odds(price) for price in prices):
        return None
    inverse = sum(1 / price for price in prices)
    if inverse >= 1 or not math.isfinite(bankroll) or bankroll < .01:
        return None
    cents = int(round(bankroll * 100))
    allocated = [int(cents / price / inverse) for price in prices]
    for i in range(cents - sum(allocated)):
        allocated[i % len(allocated)] += 1
    stakes = [value / 100 for value in allocated]
    minimum_return = min(money(stake * price) for stake, price in zip(stakes, prices))
    profit = money(minimum_return - sum(stakes))
    return (stakes, minimum_return, profit) if profit > 0 else None


class SAOddsService:
    @staticmethod
    def calculate_margin(odds):
        if not odds or not all(valid_odds(o) for o in odds):
            return None
        return round((sum(1 / o for o in odds) - 1) * 100, 2)

    async def sync_all_sa_markets(self, db):
        from backend.app.services.bookmaker_feed import import_snapshot
        return await import_snapshot(db)

    async def get_paired_odds_comparison(self, db, league_filter=None, limit=50):
        now = datetime.now(timezone.utc)
        stmt = select(SABookmakerOdds, Fixture).join(Fixture, SABookmakerOdds.fixture_id == Fixture.id).where(
            SABookmakerOdds.match_date > now, SABookmakerOdds.scraped_at >= now - QUOTE_MAX_AGE,
            SABookmakerOdds.bookmaker.in_(["HOLLYWOODBETS", "BETWAY"]), Fixture.is_current.is_(True))
        if league_filter and league_filter.lower() != "all":
            from backend.app.services.league_filter import resolve_league_id
            stmt = stmt.where(Fixture.league_id == await resolve_league_id(db, league_filter))
        rows = (await db.execute(stmt.order_by(SABookmakerOdds.scraped_at.desc()).limit(1000))).all()
        grouped = {}
        for quote, fixture in rows:
            if not observed_quote(quote, fixture, now):
                continue
            key = str(fixture.id)
            match = grouped.setdefault(key, dict(fixture_id=key, match_title=quote.match_title,
                home_team=quote.home_team, away_team=quote.away_team, league_name=quote.league_name,
                match_date=quote.match_date.isoformat(), prediction_id=str(quote.prediction_id) if quote.prediction_id else None,
                bookmakers={}))
            book = quote.bookmaker.lower()
            if book in match["bookmakers"]:
                continue
            prices = {field: getattr(quote, field) if valid_odds(getattr(quote, field)) else None for field in MARKETS.values()}
            match["bookmakers"][book] = dict(prices, bookmaker=quote.bookmaker, quote_id=str(quote.id),
                source_url=quote.source_url, scraped_at=quote.scraped_at.isoformat(),
                margin_pct=self.calculate_margin([prices[k] for k in ("odds_home", "odds_draw", "odds_away")]))
        for match in grouped.values():
            match["best_odds"] = {}
            for selection, field in MARKETS.items():
                options = [dict(odds=book[field], bookmaker=book["bookmaker"], quote_id=book["quote_id"],
                    source_url=book["source_url"], scraped_at=book["scraped_at"])
                    for book in match["bookmakers"].values() if book[field] is not None]
                match["best_odds"][selection] = max(options, key=lambda o: o["odds"]) if options else None
        return sorted(grouped.values(), key=lambda match: match["match_date"])[:limit]

    async def detect_arbitrage_opportunities(self, db, target_bankroll_zar=1000., league_filter=None):
        opportunities = []
        for match in await self.get_paired_odds_comparison(db, league_filter, 100):
            for selections in [("H", "D", "A"), ("Over 2.5", "Under 2.5"), ("BTTS Yes", "BTTS No")]:
                quotes = [match["best_odds"].get(selection) for selection in selections]
                if not all(quotes):
                    continue
                result = dutch_stakes([quote["odds"] for quote in quotes], target_bankroll_zar)
                if not result:
                    continue
                stakes, payout, profit = result
                opportunities.append(dict(match_title=match["match_title"], market=" / ".join(selections),
                    theoretical_minimum_return_zar=payout, theoretical_profit_zar=profit,
                    profit_pct=round(profit / target_bankroll_zar * 100, 2),
                    legs=[dict(quote, selection=selection, stake_zar=stake) for quote, selection, stake in zip(quotes, selections, stakes)],
                    warning="Not guaranteed: prices, limits, acceptance, rounding and settlement rules can differ."))
        return dict(opportunities=opportunities, total_arbitrage_opportunities=len(opportunities), currency="ZAR", currency_symbol="R")

    async def detect_value_bets(self, db, min_ev_pct=3., zar_bankroll=2000., league_filter=None):
        values = []
        for match in await self.get_paired_odds_comparison(db, league_filter, 100):
            if not match["prediction_id"]:
                continue
            import uuid
            pred = (await db.execute(select(Prediction).where(Prediction.id == uuid.UUID(match["prediction_id"]),
                publishable_prediction()))).scalar_one_or_none()
            if not pred or pred.home_team != match["home_team"] or pred.away_team != match["away_team"] or pred.match_date.isoformat() != match["match_date"]:
                continue
            probs = pred.probabilities or {}
            if not all(isinstance(probs.get(k), (int, float)) and math.isfinite(probs[k]) and 0 <= probs[k] <= 1 for k in ("H", "D", "A")) or abs(sum(probs[k] for k in ("H", "D", "A")) - 1) > .01:
                continue
            for pick in ("H", "D", "A"):
                quote = match["best_odds"][pick]
                if not quote:
                    continue
                edge = (probs[pick] * quote["odds"] - 1) * 100
                if edge < min_ev_pct:
                    continue
                values.append(dict(quote, match_title=match["match_title"], match_date=match["match_date"],
                    league_name=match["league_name"], prediction_id=match["prediction_id"], selection=pick,
                    market_odds=quote["odds"], model_probability_pct=round(probs[pick] * 100, 2), ev_pct=round(edge, 2),
                    warning="Model estimate, not a guaranteed edge."))
        return dict(value_bets=values, total_value_bets=len(values), currency="ZAR", currency_symbol="R")

    async def generate_perfect_bet_slips(self, db, bankroll_zar=1000.):
        # Builders require deliberate selections; there is no 'perfect' or risk-free bet.
        return dict(bankroll_zar=bankroll_zar, currency="ZAR", currency_symbol="R",
            slips={}, status="manual_selection_required",
            message="Choose fresh observed quotes from one bookmaker in the slip builder. No fabricated win probabilities or recommended stakes.")

    async def get_bookmaker_margin_analysis(self, db):
        data = await self.get_paired_odds_comparison(db, limit=100)
        books = {}
        for name in ("hollywoodbets", "betway"):
            margins = [m["bookmakers"][name]["margin_pct"] for m in data if name in m["bookmakers"] and m["bookmakers"][name]["margin_pct"] is not None]
            books[name] = {"name": name, "avg_1x2_margin_pct": round(sum(margins) / len(margins), 2) if margins else None}
        return {"bookmakers": books, "status": "ok" if data else "unavailable"}


sa_odds_service = SAOddsService()
