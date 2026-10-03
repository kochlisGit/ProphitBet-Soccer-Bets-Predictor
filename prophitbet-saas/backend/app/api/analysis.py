import base64
import io
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.dependencies import require_plan, get_current_user
from backend.app.db.models import User
from backend.app.db.session import get_db

router = APIRouter()

ANALYSIS_TYPES = [
    "correlation", "distribution", "variance", "boruta",
    "rules", "coefficients", "impurity", "description",
]


class AnalysisRequest(BaseModel):
    league_id: int
    analysis_type: str
    season: Optional[int] = None
    colormap: str = "RdYlGn"
    kwargs: Optional[dict] = None


class AnalysisResponse(BaseModel):
    analysis_type: str
    league_id: int
    image_base64: Optional[str] = None
    data: Optional[dict] = None


def _get_analyzer_class(analysis_type: str):
    from backend.app.config import setup_ml_path
    setup_ml_path()

    from src.analysis import (
        CorrelationAnalyzer,
        DistributionAnalyzer,
        VarianceAnalyzer,
        BorutaAnalyzer,
        RuleExtractorAnalyzer,
        CoefficientAnalyzer,
        GiniImpurityAnalyzer,
        DescriptiveAnalyzer,
    )
    mapping = {
        "correlation": CorrelationAnalyzer,
        "distribution": DistributionAnalyzer,
        "variance": VarianceAnalyzer,
        "boruta": BorutaAnalyzer,
        "rules": RuleExtractorAnalyzer,
        "coefficients": CoefficientAnalyzer,
        "impurity": GiniImpurityAnalyzer,
        "description": DescriptiveAnalyzer,
    }
    return mapping.get(analysis_type)


@router.post("", response_model=AnalysisResponse)
async def run_analysis(
    body: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    plan = "free"
    if user.subscription and user.subscription.status == "active":
        plan = user.subscription.plan

    if body.analysis_type not in ANALYSIS_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid analysis type. Choose from: {ANALYSIS_TYPES}")

    if plan == "free" and body.analysis_type != "description":
        raise HTTPException(status_code=403, detail="Free plan only supports descriptive statistics")

    analyzer_cls = _get_analyzer_class(body.analysis_type)
    if analyzer_cls is None:
        raise HTTPException(status_code=400, detail="Unknown analysis type")

    from backend.app.services.league_service import get_league_dataframe
    df = await get_league_dataframe(body.league_id, db)
    if df is None:
        raise HTTPException(status_code=404, detail="No data for this league")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        analyzer = analyzer_cls(df=df)
        kwargs = body.kwargs or {}
        ax = analyzer.generate_plot(season=body.season, colormap=body.colormap, **kwargs)

        buf = io.BytesIO()
        fig = ax.get_figure()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return AnalysisResponse(
            analysis_type=body.analysis_type,
            league_id=body.league_id,
            image_base64=img_b64,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/types")
async def list_analysis_types(user: User = Depends(get_current_user)):
    plan = "free"
    if user.subscription and user.subscription.status == "active":
        plan = user.subscription.plan
    available = ["description"] if plan == "free" else ANALYSIS_TYPES
    return {"types": available, "plan": plan}


from backend.app.services.backtest_service import (
    BacktestService,
    BacktestFilter,
    BacktestSummary,
    PRESET_STRATEGIES,
)
from backend.app.services.clv_service import CLVTrackerService
import uuid


@router.get("/backtest/presets")
async def get_backtest_presets(user: User = Depends(get_current_user)):
    """Get pre-configured institutional quantitative betting strategies."""
    return {"presets": PRESET_STRATEGIES}


@router.post("/backtest", response_model=BacktestSummary)
async def run_strategy_backtest(
    filters: BacktestFilter,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Run quantitative betting strategy simulation over settled predictions."""
    try:
        return await BacktestService.run_backtest(db, filters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest simulation failed: {str(e)}")


@router.get("/clv/audit")
async def get_clv_audit(
    limit: int = 150,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Audit historical predictions for Closing Line Value (CLV) edge and smart money movement."""
    try:
        return await CLVTrackerService.audit_clv_portfolio(db, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CLV audit failed: {str(e)}")


@router.get("/clv/drift/{prediction_id}")
async def get_match_odds_drift(
    prediction_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Retrieve time-series market odds movement and CLV edge for a match."""
    result = await CLVTrackerService.get_match_odds_drift(prediction_id, db)
    if not result:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return result

