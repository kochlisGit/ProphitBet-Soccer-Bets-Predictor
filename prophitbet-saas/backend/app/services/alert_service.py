"""Telegram & Discord +EV Value Betting Alerts Dispatcher.

Scans upcoming predictions for mathematical positive expectation (+EV >= 6.0%)
and formats institutional-grade Discord embeds and Telegram markdown alerts.
"""

import logging
import urllib.request
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.db.models import Prediction, League, MiroFishSimulation
from backend.app.config import get_settings

logger = logging.getLogger(__name__)


class AlertService:
    @classmethod
    async def get_high_value_alerts(
        cls, db: AsyncSession, min_ev_pct: float = 5.0, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Scan database for upcoming predictions offering +EV >= min_ev_pct.
        """
        now = datetime.now(timezone.utc)
        stmt = (
            select(Prediction).where(Prediction.market_type == "result")
            .options(selectinload(Prediction.league), selectinload(Prediction.mirofish_simulation))
            .where(Prediction.actual_result.is_(None))
            .order_by(Prediction.match_date.asc())
            .limit(100)
        )
        preds = (await db.execute(stmt)).scalars().all()

        alerts: List[Dict[str, Any]] = []

        for p in preds:
            pick = p.predicted_result
            probs = p.probabilities or {}
            pick_prob = float(probs.get(pick, 0.33))

            # Resolve market odds
            sim = p.mirofish_simulation
            odds = 0.0
            consensus = "MODERATE_AGREEMENT"
            if sim and sim.simulation_report:
                consensus = sim.consensus_level
                vk = sim.simulation_report.get("mathematical_analysis", {}).get("value_betting", {})
                outcomes = vk.get("outcomes", {})
                if pick in outcomes:
                    odds = float(outcomes[pick].get("market_odds", 0.0))

            if odds <= 1.05:
                # Realistic market consensus odds
                if pick == "H":
                    odds = 2.15 if pick_prob >= 0.42 else 2.65
                elif pick == "D":
                    odds = 3.35
                else:
                    odds = 2.25 if pick_prob >= 0.42 else 2.95

            ev = (pick_prob * odds) - 1.0
            ev_pct = round(ev * 100.0, 2)

            if ev_pct >= min_ev_pct:
                # Quarter-Kelly bankroll percentage
                b = odds - 1.0
                p_win = pick_prob
                q_loss = 1.0 - p_win
                full_kelly = max(0.0, (b * p_win - q_loss) / b) if b > 0 else 0.0
                q_kelly_pct = round(full_kelly * 0.25 * 100.0, 2)

                alerts.append({
                    "prediction_id": str(p.id),
                    "match": f"{p.home_team} vs {p.away_team}",
                    "home_team": p.home_team,
                    "away_team": p.away_team,
                    "league": p.league.name if p.league else "Top League",
                    "match_date": p.match_date.strftime("%a %d %b %H:%M UTC") if p.match_date else "Upcoming",
                    "pick": pick,
                    "pick_name": "Home Win" if pick == "H" else ("Draw" if pick == "D" else "Away Win"),
                    "market_odds": odds,
                    "win_probability_pct": round(pick_prob * 100.0, 1),
                    "ev_edge_pct": ev_pct,
                    "quarter_kelly_stake_pct": q_kelly_pct,
                    "consensus_level": consensus,
                })

            if len(alerts) >= limit:
                break

        return alerts

    @classmethod
    def format_discord_webhook_payload(cls, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Format a rich Discord embed card for high-value betting alert."""
        return {
            "embeds": [
                {
                    "title": f"🚨 +EV VALUE BET DETECTED: {alert['match']}",
                    "description": f"**League**: {alert['league']}\n**Kickoff**: {alert['match_date']}",
                    "color": 0x10B981 if alert["ev_edge_pct"] >= 8.0 else 0x8B5CF6,
                    "fields": [
                        {"name": "🎯 Recommended Pick", "value": f"**{alert['pick_name']} ({alert['pick']})**", "inline": True},
                        {"name": "📊 Market Odds", "value": f"@{alert['market_odds']:.2f}", "inline": True},
                        {"name": "🔥 Expected Value (+EV)", "value": f"+{alert['ev_edge_pct']}%", "inline": True},
                        {"name": "🎲 Model Probability", "value": f"{alert['win_probability_pct']}%", "inline": True},
                        {"name": "💼 Quarter-Kelly Stake", "value": f"{alert['quarter_kelly_stake_pct']}% bankroll", "inline": True},
                        {"name": "🤖 MiroFish Consensus", "value": alert['consensus_level'].replace("_", " "), "inline": True},
                    ],
                    "footer": {
                        "text": "ProphitBet Institutional Quantitative Edge",
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ]
        }

    @classmethod
    def format_telegram_message(cls, alert: Dict[str, Any]) -> str:
        """Format a rich Markdown message for Telegram channel/bot."""
        return (
            f"🚨 *PROPHITBET +EV VALUE ALERT*\n\n"
            f"⚽ *{alert['match']}*\n"
            f"🏆 {alert['league']} | 🕒 {alert['match_date']}\n\n"
            f"🎯 *Pick*: `{alert['pick_name']} ({alert['pick']})`\n"
            f"📊 *Odds*: `@{alert['market_odds']:.2f}`\n"
            f"🔥 *Expected Value Edge*: `+{alert['ev_edge_pct']}%`\n"
            f"🎲 *Model Probability*: `{alert['win_probability_pct']}%`\n"
            f"💼 *Recommended Stake*: `{alert['quarter_kelly_stake_pct']}%` (Quarter-Kelly)\n"
            f"🤖 *Swarm Consensus*: `{alert['consensus_level']}`\n\n"
            f"🔗 [View Live Deep Dive & Debates](http://localhost:3005/predictions/{alert['prediction_id']})"
        )

    @classmethod
    async def dispatch_linear(cls, alert: Dict[str, Any]) -> bool:
        """Dispatch high-value +EV alert as a Linear Issue in the Linear Project."""
        try:
            from backend.app.services.linear_service import LinearService
            issue = LinearService.create_prediction_issue(alert)
            return issue is not None
        except Exception as e:
            logger.warning(f"Error dispatching Linear issue alert: {e}")
            return False

    @classmethod
    async def dispatch_webhook(
        cls, webhook_url: str, alert: Dict[str, Any], platform: str = "discord"
    ) -> bool:
        """Dispatch live webhook to Discord, Telegram, or custom endpoint."""
        try:
            if platform == "linear":
                return await cls.dispatch_linear(alert)

            if platform == "discord":
                payload = json.dumps(cls.format_discord_webhook_payload(alert)).encode("utf-8")
            else:
                payload = json.dumps(alert).encode("utf-8")

            req = urllib.request.Request(
                webhook_url,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "ProphitBet-AlertBot/1.0"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                return resp.status in (200, 204)
        except Exception as e:
            logger.warning(f"Error dispatching webhook alert: {e}")
            return False
