"""MiroFish Swarm Intelligence API Routes.

Exposes endpoints for running multi-agent soccer match simulations,
retrieving ensemble forecasts, and monitoring engine status.
"""

import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.dependencies import get_current_user, get_optional_user, require_admin
from backend.app.db.models import User
from backend.app.db.session import get_db
from backend.app.services.mirofish_service import MiroFishService
from backend.app.services.vector_rag_service import QdrantVectorRAGService

router = APIRouter()


class MiroFishSimulationOut(BaseModel):
    id: str
    prediction_id: str
    swarm_predicted_result: str
    swarm_probabilities: Dict[str, float]
    ensemble_predicted_result: str
    ensemble_probabilities: Dict[str, float]
    confidence_score: float
    consensus_level: str
    simulation_report: Dict[str, Any]
    created_at: str


class SimulateRequest(BaseModel):
    force_recompute: bool = False


class SandboxSimulateRequest(BaseModel):
    home_team: str
    away_team: str
    league: Optional[str] = "Premier League"
    home_odds: Optional[float] = 2.10
    draw_odds: Optional[float] = 3.30
    away_odds: Optional[float] = 3.60
    ml_prob_h: Optional[float] = 0.45
    ml_prob_d: Optional[float] = 0.28
    ml_prob_a: Optional[float] = 0.27
    tactical_notes: Optional[str] = None
    weight_ml: Optional[float] = 0.55
    weight_swarm: Optional[float] = 0.45


class ContinuousTestRequest(BaseModel):
    batch_size: Optional[int] = 5
    stress_mode: Optional[str] = "standard"


@router.get("/status")
def get_mirofish_status():
    """Return status and configuration of MiroFish swarm engine."""
    return MiroFishService.get_status()


@router.get("/stats")
async def get_mirofish_stats(
    db: AsyncSession = Depends(get_db),
    _user: Optional[User] = Depends(get_optional_user),
):
    """Return aggregate statistics and telemetry for MiroFish swarm."""
    return await MiroFishService.get_stats(db=db)


@router.get("/calibration")
async def get_agent_calibration(
    db: AsyncSession = Depends(get_db),
    _user: Optional[User] = Depends(get_optional_user),
):
    """Closed-loop Bayesian agent calibration auditing empirical hit-rates on settled matches."""
    return await QdrantVectorRAGService.evaluate_closed_loop_agent_calibration(db=db)


@router.post("/tactical-intel/seed")
def seed_tactical_intel(
    user: User = Depends(require_admin),
):
    """Seed Qdrant vector database with European club tactical intelligence dossiers."""
    count = QdrantVectorRAGService.seed_tactical_intel_dossiers()
    return {"status": "ok", "seeded_points": count, "vector_db": "Qdrant", "collection": "team_profiles"}


@router.get("/tactical-intel/{team_name}")
def get_team_tactical_intel(
    team_name: str,
    _user: Optional[User] = Depends(get_optional_user),
):
    """Retrieve vector RAG tactical dossier for a club from Qdrant."""
    dossier = QdrantVectorRAGService.retrieve_tactical_dossier(team_name)
    return {"team_name": team_name, "tactical_intel": dossier}


@router.get("/simulations")
async def get_mirofish_simulations(
    limit: int = Query(50, ge=1, le=100),
    consensus: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    _user: Optional[User] = Depends(get_optional_user),
):
    """List recent simulations with rich prediction context for the Live Views monitor."""
    return await MiroFishService.get_simulations(
        db=db, limit=limit, consensus=consensus, search=search
    )


@router.post("/sandbox/simulate")
def run_sandbox_simulation(
    payload: SandboxSimulateRequest,
    _user: User = Depends(get_current_user),
):
    """Run interactive on-the-fly sandbox simulation with custom parameters."""
    raise HTTPException(
        status_code=422,
        detail="Sandbox simulation is disabled because its inputs are not verified provider data",
    )


@router.post("/continuous-test/run")
async def run_continuous_test_batch(
    payload: Optional[ContinuousTestRequest] = None,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(require_admin),
):
    """Run automated continuous testing across upcoming match fixtures."""
    batch_size = payload.batch_size if payload and payload.batch_size else 5
    stress_mode = payload.stress_mode if payload and payload.stress_mode else "standard"
    return await MiroFishService.run_continuous_test(
        db=db, batch_size=batch_size, stress_mode=stress_mode
    )


@router.post("/predictions/{prediction_id}/simulate", response_model=MiroFishSimulationOut)
async def simulate_prediction(
    prediction_id: uuid.UUID,
    payload: Optional[SimulateRequest] = None,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Run MiroFish multi-agent swarm simulation for a given prediction."""
    force = payload.force_recompute if payload else False
    try:
        sim = await MiroFishService.simulate_prediction(
            prediction_id=prediction_id, db=db, force_recompute=force
        )
        return MiroFishSimulationOut(
            id=str(sim.id),
            prediction_id=str(sim.prediction_id),
            swarm_predicted_result=sim.swarm_predicted_result,
            swarm_probabilities=sim.swarm_probabilities,
            ensemble_predicted_result=sim.ensemble_predicted_result,
            ensemble_probabilities=sim.ensemble_probabilities,
            confidence_score=sim.confidence_score,
            consensus_level=sim.consensus_level,
            simulation_report=sim.simulation_report,
            created_at=sim.created_at.isoformat() if sim.created_at else "",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Simulation failed")


@router.get("/predictions/{prediction_id}/simulation", response_model=Optional[MiroFishSimulationOut])
async def get_prediction_simulation(
    prediction_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_optional_user),
):
    """Retrieve existing MiroFish swarm simulation for a given prediction."""
    sim = await MiroFishService.get_simulation(prediction_id=prediction_id, db=db)
    if not sim:
        return None

    return MiroFishSimulationOut(
        id=str(sim.id),
        prediction_id=str(sim.prediction_id),
        swarm_predicted_result=sim.swarm_predicted_result,
        swarm_probabilities=sim.swarm_probabilities,
        ensemble_predicted_result=sim.ensemble_predicted_result,
        ensemble_probabilities=sim.ensemble_probabilities,
        confidence_score=sim.confidence_score,
        consensus_level=sim.consensus_level,
        simulation_report=sim.simulation_report,
        created_at=sim.created_at.isoformat() if sim.created_at else "",
    )
