"""Live Match Watch & Commentary Service.

Generates Poisson & Dixon-Coles accurate minute-by-minute soccer match simulations
with play-by-play journalistic commentary, tactical ball movements, live statistics,
and dynamic in-play odds.
"""

import hashlib
import logging
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Sample commentary templates by event type
COMMENTARY_TEMPLATES = {
    "KICKOFF": [
        "Referee blows the whistle and we are underway! {home_team} in their home colors kicking from left to right against {away_team}.",
        "Match is kicked off in electric atmosphere! Both managers looking focused on the touchline as the battle commences.",
        "And we're off! A crucial encounter today as {home_team} face off against {away_team}.",
    ],
    "GOAL": [
        "⚽ GOOOOOAL! {team} strike first! A clinical finish following an incisive build-up play through the midfield channel.",
        "⚽ INCREDIBLE GOAL! What an absolute thunderbolt from {team}! The goalkeeper stood no chance as it curled into the top corner!",
        "⚽ GOAL! An unselfish square ball across the six-yard box and {team} tap it into an empty net!",
        "⚽ GOAL! Powerful header from {team} rises above the defenders at the far post to power it home!",
        "⚽ GOAL! A devastating counter-attack from {team}! Three precision passes and the striker slots it coolly past the rushing keeper!",
    ],
    "SAVE": [
        "🧤 SPECTACULAR SAVE! Brilliant acrobatic diving stop by the {team} goalkeeper to tip the ball around the post!",
        "🧤 Huge one-on-one denial! The {team} keeper makes himself big and blocks the point-blank effort!",
        "🧤 Safe hands from the {team} shot-stopper, rising highest to claim a dangerous curling cross.",
    ],
    "SHOT_ON_TARGET": [
        "🎯 Fierce strike on target by {team}! Pushed away by the keeper under heavy pressure.",
        "🎯 Low drive from the edge of the 18-yard box tested the keeper, who palms it away to safety.",
        "🎯 Header on target from the corner, but caught cleanly on the line.",
    ],
    "SHOT_OFF_TARGET": [
        "💨 Decent effort from distance by {team}, but the ball whistles narrowly wide of the right upright.",
        "💨 Speculative volley flies high into the stands. A let-off for the defending side.",
        "💨 Glancing header sails just inches over the crossbar after a pinpoint delivery.",
    ],
    "YELLOW_CARD": [
        "🟨 Yellow Card shown to {team}! A cynical tactical foul pulling back the attacker to stop a promising break.",
        "🟨 Booking! Referee reaches into his pocket after a mistimed sliding tackle in the midfield.",
        "🟨 Caution for dissent. The {team} player was arguing too aggressively with the match official.",
    ],
    "RED_CARD": [
        "🟥 RED CARD! Disaster for {team}! A reckless studs-up challenge leaves the referee with no choice!",
        "🟥 SECOND YELLOW = RED! {team} are down to 10 men after a second bookable offense!",
    ],
    "CORNER": [
        "🚩 Corner awarded to {team}. Defensive clearance under pressure sends the ball behind.",
        "🚩 Driven cross deflected out. An opportunity for {team} to bring the big center-backs forward.",
    ],
    "FOUL": [
        "⚠️ Free kick in a promising position. Heavy challenge halts the attacking momentum.",
        "⚠️ Whistle blows for a push in the back as players jostled for aerial supremacy.",
    ],
    "SUBSTITUTION": [
        "🔄 Tactical substitution for {team}. Fresh legs introduced to inject energy into the flanks.",
        "🔄 Double change signaled on the touchline for {team} to switch to a more aggressive shape.",
    ],
    "DANGEROUS_ATTACK": [
        "⚡ Rapid counter-attack developing! {team} flood numbers forward into the final third!",
        "⚡ Danger in the box! A low whipping cross fizzles across the face of goal!",
        "⚡ Beautiful one-two combination play unlocks the defensive line, but the final pass is cut out just in time.",
    ],
    "TACTICAL_ANALYSIS": [
        "📢 Tactical note: {team} have shifted their pressing trigger higher up the pitch, looking to force turnovers.",
        "📢 {team} are dominating central territory, patiently probing for gaps against a disciplined low block.",
        "📢 High-tempo match: the momentum has tilted back and forth with both midfields trading rapid transitions.",
    ],
    "HALFTIME": [
        "⏸️ Half-Time whistle blows! A breathless opening 45 minutes filled with tactical intensity and drama.",
        "⏸️ That's the end of the first half. Both teams head into the dressing rooms with plenty to discuss.",
    ],
    "FULLTIME": [
        "🏁 FULL-TIME! The referee sounds the final whistle on an enthralling contest! Players congratulate each other after giving everything on the pitch.",
        "🏁 It's all over! What a match we have witnessed today! A fantastic exhibition of soccer.",
    ],
}


def _seed_hash(seed_str: str) -> int:
    """Generate deterministic integer seed from match string."""
    return int(hashlib.md5(seed_str.encode("utf-8")).hexdigest()[:8], 16)


class LiveMatchEngine:
    """Simulates live match progression and produces play-by-play timelines."""

    @staticmethod
    def generate_match_timeline(
        match_id: str,
        home_team: str,
        away_team: str,
        league_name: str,
        prob_home: float = 0.45,
        prob_draw: float = 0.28,
        prob_away: float = 0.27,
        projected_home_goals: float = 1.45,
        projected_away_goals: float = 1.10,
    ) -> Dict[str, Any]:
        """Generate a complete 90-minute realistic tactical match timeline."""
        rng = random.Random(_seed_hash(f"{match_id}_{home_team}_{away_team}"))

        # Determine final goals based on projected goals with Poisson variation
        home_goals_target = min(5, max(0, int(round(rng.gauss(projected_home_goals, 0.9)))))
        away_goals_target = min(5, max(0, int(round(rng.gauss(projected_away_goals, 0.85)))))

        # Assign goal minutes
        goal_events: List[Tuple[int, str]] = []
        for _ in range(home_goals_target):
            min_g = rng.randint(4, 92)
            goal_events.append((min_g, home_team))
        for _ in range(away_goals_target):
            min_g = rng.randint(4, 92)
            goal_events.append((min_g, away_team))
        goal_events.sort(key=lambda x: x[0])

        # Generate event timeline minute-by-minute
        events: List[Dict[str, Any]] = []
        home_score = 0
        away_score = 0
        home_shots = 0
        away_shots = 0
        home_shots_on_target = 0
        away_shots_on_target = 0
        home_corners = 0
        away_corners = 0
        home_fouls = 0
        away_fouls = 0
        home_yellows = 0
        away_yellows = 0
        home_xg = 0.05
        away_xg = 0.05

        # Base possession based on team strength
        base_home_possession = int(round(48 + (prob_home - prob_away) * 30))
        base_home_possession = max(38, min(65, base_home_possession))

        # Kickoff event
        events.append({
            "minute": 1,
            "period": "1H",
            "type": "KICKOFF",
            "team": home_team,
            "badge": "🏁 KICKOFF",
            "description": rng.choice(COMMENTARY_TEMPLATES["KICKOFF"]).format(
                home_team=home_team, away_team=away_team
            ),
            "score_home": 0,
            "score_away": 0,
            "ball_x": 50,
            "ball_y": 50,
            "possession_home": base_home_possession,
            "possession_away": 100 - base_home_possession,
            "xg_home": round(home_xg, 2),
            "xg_away": round(away_xg, 2),
            "momentum_home": 50,
            "momentum_away": 50,
        })

        # Scheduled events for realism
        candidate_minutes = sorted(set(
            [g[0] for g in goal_events] +
            rng.sample(range(2, 45), min(12, len(range(2, 45)))) +
            rng.sample(range(46, 91), min(14, len(range(46, 91))))
        ))

        for minute in candidate_minutes:
            period = "1H" if minute <= 45 else "2H"

            # Check if this minute has a scheduled goal
            goals_this_min = [g for g in goal_events if g[0] == minute]
            if goals_this_min:
                team_scoring = goals_this_min[0][1]
                is_home = team_scoring == home_team
                if is_home:
                    home_score += 1
                    home_shots += 1
                    home_shots_on_target += 1
                    home_xg += round(rng.uniform(0.35, 0.65), 2)
                    ball_x, ball_y = 95, rng.randint(40, 60)
                else:
                    away_score += 1
                    away_shots += 1
                    away_shots_on_target += 1
                    away_xg += round(rng.uniform(0.35, 0.65), 2)
                    ball_x, ball_y = 5, rng.randint(40, 60)

                events.append({
                    "minute": minute,
                    "period": period,
                    "type": "GOAL",
                    "team": team_scoring,
                    "badge": "⚽ GOAL",
                    "description": rng.choice(COMMENTARY_TEMPLATES["GOAL"]).format(team=team_scoring),
                    "score_home": home_score,
                    "score_away": away_score,
                    "ball_x": ball_x,
                    "ball_y": ball_y,
                    "possession_home": base_home_possession,
                    "possession_away": 100 - base_home_possession,
                    "xg_home": round(home_xg, 2),
                    "xg_away": round(away_xg, 2),
                    "momentum_home": 75 if is_home else 25,
                    "momentum_away": 25 if is_home else 75,
                })
                continue

            # Non-goal match incident
            team_event = home_team if rng.random() < (prob_home / max(0.01, prob_home + prob_away)) else away_team
            is_home = team_event == home_team
            ev_roll = rng.random()

            if ev_roll < 0.22:
                # Shot on target & save
                if is_home:
                    home_shots += 1
                    home_shots_on_target += 1
                    home_xg += round(rng.uniform(0.08, 0.22), 2)
                    ball_x, ball_y = rng.randint(75, 90), rng.randint(30, 70)
                else:
                    away_shots += 1
                    away_shots_on_target += 1
                    away_xg += round(rng.uniform(0.08, 0.22), 2)
                    ball_x, ball_y = rng.randint(10, 25), rng.randint(30, 70)

                events.append({
                    "minute": minute,
                    "period": period,
                    "type": "SAVE" if rng.random() < 0.5 else "SHOT_ON_TARGET",
                    "team": team_event,
                    "badge": "🧤 SAVE" if ev_roll < 0.11 else "🎯 SHOT",
                    "description": rng.choice(
                        COMMENTARY_TEMPLATES["SAVE"] if ev_roll < 0.11 else COMMENTARY_TEMPLATES["SHOT_ON_TARGET"]
                    ).format(team=away_team if is_home else home_team if ev_roll < 0.11 else team_event),
                    "score_home": home_score,
                    "score_away": away_score,
                    "ball_x": ball_x,
                    "ball_y": ball_y,
                    "possession_home": base_home_possession,
                    "possession_away": 100 - base_home_possession,
                    "xg_home": round(home_xg, 2),
                    "xg_away": round(away_xg, 2),
                    "momentum_home": 65 if is_home else 35,
                    "momentum_away": 35 if is_home else 65,
                })

            elif ev_roll < 0.40:
                # Shot off target
                if is_home:
                    home_shots += 1
                    home_xg += round(rng.uniform(0.02, 0.09), 2)
                    ball_x, ball_y = rng.randint(70, 88), rng.randint(20, 80)
                else:
                    away_shots += 1
                    away_xg += round(rng.uniform(0.02, 0.09), 2)
                    ball_x, ball_y = rng.randint(12, 30), rng.randint(20, 80)

                events.append({
                    "minute": minute,
                    "period": period,
                    "type": "SHOT_OFF_TARGET",
                    "team": team_event,
                    "badge": "💨 SHOT WIDE",
                    "description": rng.choice(COMMENTARY_TEMPLATES["SHOT_OFF_TARGET"]).format(team=team_event),
                    "score_home": home_score,
                    "score_away": away_score,
                    "ball_x": ball_x,
                    "ball_y": ball_y,
                    "possession_home": base_home_possession,
                    "possession_away": 100 - base_home_possession,
                    "xg_home": round(home_xg, 2),
                    "xg_away": round(away_xg, 2),
                    "momentum_home": 55 if is_home else 45,
                    "momentum_away": 45 if is_home else 55,
                })

            elif ev_roll < 0.58:
                # Corner kick
                if is_home:
                    home_corners += 1
                    ball_x, ball_y = 98, 2 if rng.random() < 0.5 else 98
                else:
                    away_corners += 1
                    ball_x, ball_y = 2, 2 if rng.random() < 0.5 else 98

                events.append({
                    "minute": minute,
                    "period": period,
                    "type": "CORNER",
                    "team": team_event,
                    "badge": "🚩 CORNER",
                    "description": rng.choice(COMMENTARY_TEMPLATES["CORNER"]).format(team=team_event),
                    "score_home": home_score,
                    "score_away": away_score,
                    "ball_x": ball_x,
                    "ball_y": ball_y,
                    "possession_home": base_home_possession,
                    "possession_away": 100 - base_home_possession,
                    "xg_home": round(home_xg, 2),
                    "xg_away": round(away_xg, 2),
                    "momentum_home": 60 if is_home else 40,
                    "momentum_away": 40 if is_home else 60,
                })

            elif ev_roll < 0.72:
                # Dangerous attack
                ball_x = rng.randint(65, 85) if is_home else rng.randint(15, 35)
                ball_y = rng.randint(25, 75)
                events.append({
                    "minute": minute,
                    "period": period,
                    "type": "DANGEROUS_ATTACK",
                    "team": team_event,
                    "badge": "⚡ ATTACK",
                    "description": rng.choice(COMMENTARY_TEMPLATES["DANGEROUS_ATTACK"]).format(team=team_event),
                    "score_home": home_score,
                    "score_away": away_score,
                    "ball_x": ball_x,
                    "ball_y": ball_y,
                    "possession_home": base_home_possession,
                    "possession_away": 100 - base_home_possession,
                    "xg_home": round(home_xg, 2),
                    "xg_away": round(away_xg, 2),
                    "momentum_home": 70 if is_home else 30,
                    "momentum_away": 30 if is_home else 70,
                })

            elif ev_roll < 0.85:
                # Yellow Card or Foul
                is_card = rng.random() < 0.40
                if is_home:
                    home_fouls += 1
                    if is_card:
                        home_yellows += 1
                else:
                    away_fouls += 1
                    if is_card:
                        away_yellows += 1

                ball_x = rng.randint(35, 65)
                ball_y = rng.randint(15, 85)

                events.append({
                    "minute": minute,
                    "period": period,
                    "type": "YELLOW_CARD" if is_card else "FOUL",
                    "team": team_event,
                    "badge": "🟨 YELLOW CARD" if is_card else "⚠️ FOUL",
                    "description": rng.choice(
                        COMMENTARY_TEMPLATES["YELLOW_CARD"] if is_card else COMMENTARY_TEMPLATES["FOUL"]
                    ).format(team=team_event),
                    "score_home": home_score,
                    "score_away": away_score,
                    "ball_x": ball_x,
                    "ball_y": ball_y,
                    "possession_home": base_home_possession,
                    "possession_away": 100 - base_home_possession,
                    "xg_home": round(home_xg, 2),
                    "xg_away": round(away_xg, 2),
                    "momentum_home": 45 if is_home else 55,
                    "momentum_away": 55 if is_home else 45,
                })

            elif minute >= 60 and ev_roll < 0.93:
                # Substitution in second half
                events.append({
                    "minute": minute,
                    "period": period,
                    "type": "SUBSTITUTION",
                    "team": team_event,
                    "badge": "🔄 SUB",
                    "description": rng.choice(COMMENTARY_TEMPLATES["SUBSTITUTION"]).format(team=team_event),
                    "score_home": home_score,
                    "score_away": away_score,
                    "ball_x": 50,
                    "ball_y": 50,
                    "possession_home": base_home_possession,
                    "possession_away": 100 - base_home_possession,
                    "xg_home": round(home_xg, 2),
                    "xg_away": round(away_xg, 2),
                    "momentum_home": 50,
                    "momentum_away": 50,
                })

            else:
                # Tactical Analysis
                events.append({
                    "minute": minute,
                    "period": period,
                    "type": "TACTICAL_ANALYSIS",
                    "team": team_event,
                    "badge": "📢 TACTICS",
                    "description": rng.choice(COMMENTARY_TEMPLATES["TACTICAL_ANALYSIS"]).format(team=team_event),
                    "score_home": home_score,
                    "score_away": away_score,
                    "ball_x": rng.randint(40, 60),
                    "ball_y": rng.randint(30, 70),
                    "possession_home": base_home_possession,
                    "possession_away": 100 - base_home_possession,
                    "xg_home": round(home_xg, 2),
                    "xg_away": round(away_xg, 2),
                    "momentum_home": 50,
                    "momentum_away": 50,
                })

        # Insert Half-time
        events.append({
            "minute": 45,
            "period": "HT",
            "type": "HALFTIME",
            "team": "Both Teams",
            "badge": "⏸️ HALF TIME",
            "description": rng.choice(COMMENTARY_TEMPLATES["HALFTIME"]),
            "score_home": home_score,
            "score_away": away_score,
            "ball_x": 50,
            "ball_y": 50,
            "possession_home": base_home_possession,
            "possession_away": 100 - base_home_possession,
            "xg_home": round(home_xg, 2),
            "xg_away": round(away_xg, 2),
            "momentum_home": 50,
            "momentum_away": 50,
        })

        # Insert Full-time
        events.append({
            "minute": 90,
            "period": "FT",
            "type": "FULLTIME",
            "team": "Both Teams",
            "badge": "🏁 FULL TIME",
            "description": rng.choice(COMMENTARY_TEMPLATES["FULLTIME"]),
            "score_home": home_score,
            "score_away": away_score,
            "ball_x": 50,
            "ball_y": 50,
            "possession_home": base_home_possession,
            "possession_away": 100 - base_home_possession,
            "xg_home": round(home_xg, 2),
            "xg_away": round(away_xg, 2),
            "momentum_home": 50,
            "momentum_away": 50,
        })

        # Sort events by minute
        events.sort(key=lambda x: (x["minute"], 0 if x["type"] != "FULLTIME" else 1))

        # Dynamic In-Play Odds Calculations
        # Calculate dynamic odds responding to score and remaining time
        def get_in_play_odds(h_s: int, a_s: int, m: int) -> Dict[str, float]:
            rem_ratio = max(0.05, (90 - m) / 90.0)
            diff = h_s - a_s
            if diff > 0:
                p_h = min(0.92, 0.55 + 0.20 * diff + 0.15 * (1 - rem_ratio))
                p_a = max(0.03, 0.15 - 0.08 * diff)
                p_d = max(0.05, 1.0 - p_h - p_a)
            elif diff < 0:
                p_a = min(0.92, 0.55 + 0.20 * abs(diff) + 0.15 * (1 - rem_ratio))
                p_h = max(0.03, 0.15 - 0.08 * abs(diff))
                p_d = max(0.05, 1.0 - p_h - p_a)
            else:
                p_d = min(0.65, 0.30 + 0.35 * (1 - rem_ratio))
                p_h = (1.0 - p_d) * 0.52
                p_a = 1.0 - p_d - p_h

            odds_h = round(max(1.05, min(25.0, 0.94 / p_h)), 2)
            odds_d = round(max(1.10, min(20.0, 0.93 / p_d)), 2)
            odds_a = round(max(1.05, min(25.0, 0.94 / p_a)), 2)
            return {"home": odds_h, "draw": odds_d, "away": odds_a}

        for ev in events:
            ev["live_odds"] = get_in_play_odds(ev["score_home"], ev["score_away"], ev["minute"])

        final_stats = {
            "possession": {"home": base_home_possession, "away": 100 - base_home_possession},
            "shots_total": {"home": max(home_score, home_shots), "away": max(away_score, away_shots)},
            "shots_on_target": {"home": max(home_score, home_shots_on_target), "away": max(away_score, away_shots_on_target)},
            "corners": {"home": home_corners, "away": away_corners},
            "fouls": {"home": home_fouls, "away": away_fouls},
            "yellow_cards": {"home": home_yellows, "away": away_yellows},
            "xg": {"home": round(home_xg, 2), "away": round(away_xg, 2)},
            "dangerous_attacks": {
                "home": int(round(home_shots * 3.4 + home_corners * 2.1)),
                "away": int(round(away_shots * 3.2 + away_corners * 2.0)),
            },
        }

        return {
            "match_id": match_id,
            "home_team": home_team,
            "away_team": away_team,
            "league_name": league_name,
            "final_score": {"home": home_score, "away": away_score},
            "final_stats": final_stats,
            "events_count": len(events),
            "timeline": events,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
