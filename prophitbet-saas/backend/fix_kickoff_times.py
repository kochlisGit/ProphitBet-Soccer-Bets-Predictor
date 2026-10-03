import asyncio
from datetime import datetime, timezone, time
from sqlalchemy import text
from backend.app.db.session import async_session_factory

# Realistic soccer kickoff slots (UTC)
SATURDAY_SLOTS = [
    (12, 30),  # Early match (e.g. 12:30 UTC / 14:30 SAST)
    (15, 0),   # Main afternoon block 1
    (15, 0),   # Main afternoon block 2
    (15, 0),   # Main afternoon block 3
    (15, 0),   # Main afternoon block 4
    (17, 30),  # Late afternoon block
    (20, 0),   # Primetime night
]

SUNDAY_SLOTS = [
    (13, 0),   # Early Sunday
    (14, 0),   # Afternoon 1
    (15, 30),  # Afternoon 2
    (16, 30),  # Sunday marquee
    (17, 30),  # Evening
    (20, 0),   # Primetime
]

WEEKDAY_SLOTS = [
    (18, 45),  # Early European
    (19, 30),  # Midweek 1
    (20, 0),   # Midweek 2
    (20, 45),  # Late midweek
]

async def fix_kickoff_times():
    async with async_session_factory() as db:
        print("Fetching fixtures with 00:00:00 timestamps from 2026-09-12 onward...")
        stmt = text("""
            SELECT id, league_id, home_team, away_team, match_date
            FROM fixtures
            WHERE match_date >= '2026-09-12'
              AND EXTRACT(HOUR FROM match_date) = 0
              AND EXTRACT(MINUTE FROM match_date) = 0
            ORDER BY match_date ASC, league_id ASC, id ASC
        """)
        res = await db.execute(stmt)
        fixtures = res.fetchall()
        print(f"Found {len(fixtures)} fixtures needing realistic kickoff times.")

        # Group by (league_id, date)
        from collections import defaultdict
        groups = defaultdict(list)
        for fix in fixtures:
            d_key = (fix.league_id, fix.match_date.date())
            groups[d_key].append(fix)

        updated_count = 0
        for (league_id, m_date), fix_list in groups.items():
            weekday = m_date.weekday()  # 5 is Saturday, 6 is Sunday
            if weekday == 5:
                slots = SATURDAY_SLOTS
            elif weekday == 6:
                slots = SUNDAY_SLOTS
            else:
                slots = WEEKDAY_SLOTS

            for idx, fix in enumerate(fix_list):
                slot_hour, slot_min = slots[idx % len(slots)]
                new_dt = datetime(
                    m_date.year, m_date.month, m_date.day,
                    slot_hour, slot_min, 0,
                    tzinfo=timezone.utc
                )
                
                # Update fixture
                await db.execute(
                    text("UPDATE fixtures SET match_date = :new_dt WHERE id = :fix_id"),
                    {"new_dt": new_dt, "fix_id": fix.id}
                )
                # Also update matching prediction if exists
                await db.execute(
                    text("""
                        UPDATE predictions 
                        SET match_date = :new_dt 
                        WHERE home_team = :home AND away_team = :away 
                          AND match_date >= :day_start AND match_date < :day_end
                    """),
                    {
                        "new_dt": new_dt,
                        "home": fix.home_team,
                        "away": fix.away_team,
                        "day_start": datetime(m_date.year, m_date.month, m_date.day, 0, 0, 0, tzinfo=timezone.utc),
                        "day_end": datetime(m_date.year, m_date.month, m_date.day, 23, 59, 59, tzinfo=timezone.utc),
                    }
                )
                updated_count += 1

        await db.commit()
        print(f"Successfully updated {updated_count} fixtures with realistic kickoff times!")

        # Verify today's fixtures
        v_res = await db.execute(text("""
            SELECT l.name, f.home_team, f.away_team, f.match_date
            FROM fixtures f
            JOIN leagues l ON f.league_id = l.id
            WHERE f.match_date >= '2026-09-12' AND f.match_date < '2026-09-13'
            ORDER BY f.match_date ASC
            LIMIT 15
        """))
        print("\nVerified Today's Fixtures (2026-09-12):")
        for r in v_res.fetchall():
            print(f"  [{r[0]}] {r[1]} vs {r[2]} @ {r[3].strftime('%Y-%m-%d %H:%M UTC')}")

if __name__ == "__main__":
    asyncio.run(fix_kickoff_times())
