import asyncio
from backend.app.db.session import async_session_factory
from sqlalchemy import text

async def check():
    async with async_session_factory() as db:
        # Check Saturday 2026-09-12 fixtures count
        res = await db.execute(text("""
            SELECT l.name, count(*)
            FROM fixtures f
            JOIN leagues l ON f.league_id = l.id
            WHERE f.match_date >= '2026-09-12' AND f.match_date < '2026-09-13'
            GROUP BY l.name
            ORDER BY count(*) DESC
        """))
        print("Fixtures on 2026-09-12 by league:")
        for r in res.fetchall():
            print(f"  {r[0]}: {r[1]}")

if __name__ == "__main__":
    asyncio.run(check())
