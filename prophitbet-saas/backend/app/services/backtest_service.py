import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models import Prediction, Fixture, MiroFishSimulation, League


class BacktestFilter(BaseModel):
    league_id: Optional[int] = None
    min_ev_pct: float = Field(0.0, description="Minimum +EV edge required to place bet")
    min_confidence: float = Field(0.0, description="Minimum model probability threshold")
    consensus_filter: str = Field("ALL", description="ALL, STRONG_CONSENSUS, UPSET_ALERT, MODERATE_AGREEMENT")
    allowed_picks: List[str] = Field(default=["H", "D", "A"], description="Outcomes to wager on")
    staking_strategy: str = Field("quarter_kelly", description="flat, quarter_kelly, half_kelly, proportional")
    flat_stake_amount: float = Field(100.0, ge=5.0, le=10000.0)
    proportional_stake_pct: float = Field(2.5, ge=0.5, le=25.0)
    initial_bankroll: float = Field(10000.0, ge=100.0, le=1000000.0)


class TradeRecord(BaseModel):
    id: str
    date: str
    league: str
    match: str
    pick: str
    actual: str
    is_win: bool
    odds: float
    ev_pct: float
    stake: float
    pnl: float
    bankroll: float
    drawdown_pct: float


class BacktestSummary(BaseModel):
    initial_bankroll: float
    final_bankroll: float
    net_profit: float
    roi_pct: float
    total_turnover: float
    yield_pct: float
    total_bets: int
    winning_bets: int
    losing_bets: int
    win_rate_pct: float
    max_drawdown_pct: float
    profit_factor: float
    sharpe_ratio: float
    avg_odds: float
    breakdown_by_outcome: Dict[str, Dict[str, Any]]
    benchmark_home_roi: float
    equity_curve: List[Dict[str, Any]]
    trades: List[TradeRecord]


PRESET_STRATEGIES = [
    {
        "id": "quarter_kelly_ev_plus_3",
        "title": "Institutional Quarter-Kelly (+EV ≥ 3%)",
        "description": "Standard quantitative bankroll preservation betting only on mathematically positive edges",
        "params": {
            "min_ev_pct": 3.0,
            "min_confidence": 0.35,
            "consensus_filter": "ALL",
            "allowed_picks": ["H", "D", "A"],
            "staking_strategy": "quarter_kelly",
            "initial_bankroll": 10000.0,
        },
    },
    {
        "id": "high_conviction_consensus",
        "title": "High Conviction MiroFish Swarm (Consensus Only)",
        "description": "Filter strictly for fixtures where ML model and 4-agent swarm reached Strong Consensus",
        "params": {
            "min_ev_pct": 1.5,
            "min_confidence": 0.42,
            "consensus_filter": "STRONG_CONSENSUS",
            "allowed_picks": ["H", "D", "A"],
            "staking_strategy": "proportional",
            "proportional_stake_pct": 3.0,
            "initial_bankroll": 10000.0,
        },
    },
    {
        "id": "home_fortress_flat",
        "title": "Home Fortress Value Bias (Flat $150)",
        "description": "Exploit historical home ground supremacy when Dixon-Coles xG supremacy is positive",
        "params": {
            "min_ev_pct": 2.0,
            "min_confidence": 0.40,
            "consensus_filter": "ALL",
            "allowed_picks": ["H"],
            "staking_strategy": "flat",
            "flat_stake_amount": 150.0,
            "initial_bankroll": 10000.0,
        },
    },
    {
        "id": "underdog_value_hunter",
        "title": "Draw & Away Value Hunter (High Odds)",
        "description": "Capitalize on market over-pricing of favorites by taking mispriced Draw & Away positions",
        "params": {
            "min_ev_pct": 5.0,
            "min_confidence": 0.22,
            "consensus_filter": "ALL",
            "allowed_picks": ["D", "A"],
            "staking_strategy": "quarter_kelly",
            "initial_bankroll": 10000.0,
        },
    },
]


class BacktestService:
    @staticmethod
    async def run_backtest(db: AsyncSession, filters: BacktestFilter) -> BacktestSummary:
        """
        Execute full historical simulation over settled predictions.
        """
        # 1. Fetch settled predictions
        stmt = (
            select(Prediction, League.name.label("league_name"))
            .join(League, Prediction.league_id == League.id)
            .where(Prediction.actual_result.isnot(None))
            .order_by(Prediction.match_date.asc())
        )
        if filters.league_id:
            stmt = stmt.where(Prediction.league_id == filters.league_id)

        rows = (await db.execute(stmt)).all()

        # Pre-fetch MiroFish simulations for these predictions
        sim_stmt = select(MiroFishSimulation)
        sim_results = (await db.execute(sim_stmt)).scalars().all()
        sim_map = {sim.prediction_id: sim for sim in sim_results}

        # Initialize simulation variables
        current_bankroll = float(filters.initial_bankroll)
        peak_bankroll = current_bankroll
        max_drawdown = 0.0

        total_turnover = 0.0
        gross_wins = 0.0
        gross_losses = 0.0
        returns_list: List[float] = []

        trades: List[TradeRecord] = []
        equity_curve: List[Dict[str, Any]] = [
            {
                "date": "Start",
                "bankroll": round(current_bankroll, 2),
                "drawdown_pct": 0.0,
                "profit": 0.0,
            }
        ]

        outcome_stats: Dict[str, Dict[str, Any]] = {
            "H": {"bets": 0, "wins": 0, "profit": 0.0, "staked": 0.0},
            "D": {"bets": 0, "wins": 0, "profit": 0.0, "staked": 0.0},
            "A": {"bets": 0, "wins": 0, "profit": 0.0, "staked": 0.0},
        }

        # Benchmark blind home tracker
        bench_bankroll = float(filters.initial_bankroll)

        for pred, league_name in rows:
            if current_bankroll <= 10.0:
                break  # Bankrupt protection

            pick = pred.predicted_result
            actual = pred.actual_result
            probs = pred.probabilities or {}
            pick_prob = float(probs.get(pick, 0.33))

            # Filter 1: Allowed picks
            if pick not in filters.allowed_picks:
                continue

            # Filter 2: Min model confidence
            if pick_prob < filters.min_confidence:
                continue

            # Filter 3: MiroFish consensus
            sim = sim_map.get(pred.id)
            sim_consensus = sim.consensus_level if sim else "MODERATE_AGREEMENT"
            if filters.consensus_filter != "ALL" and sim_consensus != filters.consensus_filter:
                continue

            # Resolve Market Odds
            odds = 0.0
            if sim and sim.simulation_report:
                vk = sim.simulation_report.get("mathematical_analysis", {}).get("value_and_kelly", {})
                outcomes = vk.get("outcomes", {})
                if pick in outcomes:
                    odds = float(outcomes[pick].get("market_odds", 0.0))

            if odds <= 1.05:
                continue

            # Calculate Expected Value (+EV)
            ev = (pick_prob * odds) - 1.0
            ev_pct = round(ev * 100.0, 2)

            # Filter 4: Min EV Edge
            if ev_pct < filters.min_ev_pct:
                continue

            # Determine Stake Amount
            stake = 0.0
            if filters.staking_strategy == "flat":
                stake = min(current_bankroll, filters.flat_stake_amount)
            elif filters.staking_strategy in ("quarter_kelly", "half_kelly"):
                mult = 0.50 if filters.staking_strategy == "half_kelly" else 0.25
                if odds > 1.0:
                    full_kelly = (pick_prob * odds - 1.0) / (odds - 1.0)
                    rec_kelly = max(0.005, min(0.12, full_kelly * mult))
                else:
                    rec_kelly = 0.02
                stake = min(current_bankroll, current_bankroll * rec_kelly)
            elif filters.staking_strategy == "proportional":
                stake = min(current_bankroll, current_bankroll * (filters.proportional_stake_pct / 100.0))
            else:
                stake = min(current_bankroll, 100.0)

            stake = round(max(5.0, stake), 2)
            if stake > current_bankroll:
                stake = current_bankroll

            # Determine Outcome
            is_win = (pick == actual)
            if is_win:
                pnl = round(stake * (odds - 1.0), 2)
                gross_wins += pnl
            else:
                pnl = round(-stake, 2)
                gross_losses += abs(pnl)

            # Update bankrolls
            current_bankroll = round(current_bankroll + pnl, 2)
            total_turnover += stake
            trade_return = pnl / stake if stake > 0 else 0.0
            returns_list.append(trade_return)

            # Track Drawdown
            if current_bankroll > peak_bankroll:
                peak_bankroll = current_bankroll
            current_dd = round(((peak_bankroll - current_bankroll) / peak_bankroll) * 100.0, 2) if peak_bankroll > 0 else 0.0
            if current_dd > max_drawdown:
                max_drawdown = current_dd

            # Track outcome breakdown
            outcome_stats[pick]["bets"] += 1
            outcome_stats[pick]["staked"] += stake
            outcome_stats[pick]["profit"] += pnl
            if is_win:
                outcome_stats[pick]["wins"] += 1

            # Append Trade
            match_str = f"{pred.home_team} vs {pred.away_team}"
            date_str = pred.match_date.strftime("%Y-%m-%d") if pred.match_date else "Recent"
            trades.append(
                TradeRecord(
                    id=str(pred.id),
                    date=date_str,
                    league=league_name,
                    match=match_str,
                    pick=pick,
                    actual=actual,
                    is_win=is_win,
                    odds=odds,
                    ev_pct=ev_pct,
                    stake=stake,
                    pnl=pnl,
                    bankroll=current_bankroll,
                    drawdown_pct=current_dd,
                )
            )

            equity_curve.append({
                "date": date_str,
                "match": match_str,
                "bankroll": current_bankroll,
                "drawdown_pct": current_dd,
                "profit": round(current_bankroll - filters.initial_bankroll, 2),
            })

        # Calculate Summary Metrics
        net_profit = round(current_bankroll - filters.initial_bankroll, 2)
        roi_pct = round((net_profit / filters.initial_bankroll) * 100.0, 2)
        yield_pct = round((net_profit / total_turnover) * 100.0, 2) if total_turnover > 0 else 0.0

        total_bets = len(trades)
        winning_bets = sum(1 for t in trades if t.is_win)
        losing_bets = total_bets - winning_bets
        win_rate = round((winning_bets / total_bets) * 100.0, 2) if total_bets > 0 else 0.0

        profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else (99.0 if gross_wins > 0 else 1.0)
        avg_odds = round(sum(t.odds for t in trades) / total_bets, 2) if total_bets > 0 else 0.0

        # Sharpe Ratio (annualized proxy based on average trade returns)
        if len(returns_list) > 1:
            mean_ret = sum(returns_list) / len(returns_list)
            var_ret = sum((r - mean_ret) ** 2 for r in returns_list) / (len(returns_list) - 1)
            std_ret = math.sqrt(var_ret) if var_ret > 0 else 1.0
            sharpe = round((mean_ret / std_ret) * math.sqrt(min(250, len(returns_list))), 2)
        else:
            sharpe = 0.0

        # Compute benchmark: flat 100 on Home
        home_rows = [p for p, _ in rows]
        home_wins_count = sum(1 for p in home_rows if p.actual_result == "H")
        home_loss_count = len(home_rows) - home_wins_count
        bench_profit = (home_wins_count * 1.85 * 100) - (len(home_rows) * 100)
        benchmark_home_roi = round((bench_profit / (len(home_rows) * 100)) * 100.0, 2) if home_rows else -12.5

        return BacktestSummary(
            initial_bankroll=filters.initial_bankroll,
            final_bankroll=current_bankroll,
            net_profit=net_profit,
            roi_pct=roi_pct,
            total_turnover=round(total_turnover, 2),
            yield_pct=yield_pct,
            total_bets=total_bets,
            winning_bets=winning_bets,
            losing_bets=losing_bets,
            win_rate_pct=win_rate,
            max_drawdown_pct=round(max_drawdown, 2),
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            avg_odds=avg_odds,
            breakdown_by_outcome=outcome_stats,
            benchmark_home_roi=benchmark_home_roi,
            equity_curve=equity_curve[-150:] if len(equity_curve) > 150 else equity_curve,
            trades=trades[-50:] if len(trades) > 50 else trades,
        )
