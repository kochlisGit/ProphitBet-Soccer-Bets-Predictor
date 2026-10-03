import asyncio
from backend.app.db.session import async_session_factory
from backend.app.services.sa_odds_service import sa_odds_service

async def main():
    async with async_session_factory() as db:
        print("Running sync_all_sa_markets...")
        res = await sa_odds_service.sync_all_sa_markets(db)
        print("Sync result:", res)
        
        print("\nTesting get_paired_odds_comparison (ALL):")
        odds_all = await sa_odds_service.get_paired_odds_comparison(db, limit=10)
        print(f"Total matches retrieved: {len(odds_all)}")
        for m in odds_all[:5]:
            print(f"  [{m['league_name']}] {m['match_title']} @ {m['match_date']}")
            print(f"     Best 1X2: Home {m['best_odds']['home']['odds']} ({m['best_odds']['home']['bookmaker']}) | Draw {m['best_odds']['draw']['odds']} | Away {m['best_odds']['away']['odds']}")

        print("\nTesting get_paired_odds_comparison (Premier League):")
        odds_epl = await sa_odds_service.get_paired_odds_comparison(db, league_filter="Premier League", limit=5)
        print(f"EPL matches: {len(odds_epl)}")
        for m in odds_epl:
            print(f"  [{m['league_name']}] {m['match_title']} @ {m['match_date']}")

        print("\nTesting get_paired_odds_comparison (Betway Premiership):")
        odds_psl = await sa_odds_service.get_paired_odds_comparison(db, league_filter="Betway Premiership", limit=5)
        print(f"PSL matches: {len(odds_psl)}")
        for m in odds_psl:
            print(f"  [{m['league_name']}] {m['match_title']} @ {m['match_date']}")

        print("\nTesting generate_perfect_bet_slips:")
        slips = await sa_odds_service.generate_perfect_bet_slips(db, bankroll_zar=1500)
        for key, slip in slips["slips"].items():
            print(f"  Slip '{key}': {slip['title']} - Compound Odds: {slip.get('compound_odds')}, Payout: R{slip.get('expected_payout_zar')}")
            for leg in slip["legs"][:2]:
                print(f"     Leg: {leg.get('match_title')} on {leg.get('match_date')} - Pick: {leg.get('selection')} @ {leg.get('odds')}")

if __name__ == "__main__":
    asyncio.run(main())
