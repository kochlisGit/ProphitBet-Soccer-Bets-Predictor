from fastapi import APIRouter

from backend.app.api import auth, predictions, leagues, models, analysis, fixtures, billing, admin, mirofish, betslip, sa_markets, live, results, linear

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Auth"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
api_router.include_router(leagues.router, prefix="/leagues", tags=["Leagues"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])
api_router.include_router(fixtures.router, prefix="/fixtures", tags=["Fixtures"])
api_router.include_router(billing.router, prefix="/billing", tags=["Billing"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin"])
api_router.include_router(mirofish.router, prefix="/mirofish", tags=["MiroFish Swarm"])
api_router.include_router(betslip.router, prefix="/betslip", tags=["Bet Slip Journal"])
api_router.include_router(sa_markets.router, prefix="/sa-markets", tags=["South African Markets"])
api_router.include_router(live.router, prefix="/live", tags=["Live Match Watch"])
api_router.include_router(results.router, prefix="/results", tags=["Match Results & Learning"])
api_router.include_router(linear.router, prefix="/linear", tags=["Linear Integration"])


