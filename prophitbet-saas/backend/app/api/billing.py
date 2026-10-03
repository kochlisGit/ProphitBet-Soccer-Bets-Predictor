import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.dependencies import get_current_user
from backend.app.config import get_settings
from backend.app.db.models import Subscription, User
from backend.app.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


class CheckoutRequest(BaseModel):
    plan: str  # "pro" or "elite"


class CheckoutResponse(BaseModel):
    checkout_url: str


class UsageResponse(BaseModel):
    plan: str
    models_trained_this_month: int
    models_limit: int


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(
    body: CheckoutRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    import stripe
    stripe.api_key = settings.STRIPE_SECRET_KEY

    if body.plan not in ("pro", "elite"):
        raise HTTPException(status_code=400, detail="Invalid plan. Choose 'pro' or 'elite'.")

    price_id = settings.STRIPE_PRICE_PRO if body.plan == "pro" else settings.STRIPE_PRICE_ELITE

    if not user.stripe_customer_id:
        customer = stripe.Customer.create(email=user.email, metadata={"user_id": str(user.id)})
        user.stripe_customer_id = customer.id
        await db.flush()

    session = stripe.checkout.Session.create(
        customer=user.stripe_customer_id,
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=f"{settings.FRONTEND_URL}/billing?success=true",
        cancel_url=f"{settings.FRONTEND_URL}/billing?canceled=true",
        metadata={"user_id": str(user.id), "plan": body.plan},
        subscription_data={"metadata": {"user_id": str(user.id), "plan": body.plan}},
    )
    return CheckoutResponse(checkout_url=session.url)


@router.post("/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    import stripe
    stripe.api_key = settings.STRIPE_SECRET_KEY

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, settings.STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        logger.error(f"Stripe webhook verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid webhook")

    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        user_id = data.get("metadata", {}).get("user_id")
        plan = data.get("metadata", {}).get("plan", "pro")
        stripe_sub_id = data.get("subscription")
        if user_id:
            from uuid import UUID
            result = await db.execute(select(Subscription).where(Subscription.user_id == UUID(user_id)))
            sub = result.scalar_one_or_none()
            if sub:
                sub.plan = plan
                sub.stripe_subscription_id = stripe_sub_id
                sub.status = "active"
            else:
                sub = Subscription(user_id=UUID(user_id), plan=plan, stripe_subscription_id=stripe_sub_id, status="active")
                db.add(sub)

    elif event_type in ("customer.subscription.updated", "customer.subscription.deleted"):
        stripe_sub_id = data.get("id")
        status = data.get("status")
        result = await db.execute(
            select(Subscription).where(Subscription.stripe_subscription_id == stripe_sub_id)
        )
        sub = result.scalar_one_or_none()
        if sub:
            if event_type == "customer.subscription.deleted" or status == "canceled":
                sub.plan = "free"
                sub.status = "canceled"
            else:
                # Keep the purchased plan during payment problems; access is
                # controlled by status. Prices restore older downgraded rows.
                price_plans = {price: plan for price, plan in [
                    (settings.STRIPE_PRICE_PRO, "pro"), (settings.STRIPE_PRICE_ELITE, "elite")
                ] if price}
                for item in (data.get("items") or {}).get("data", []):
                    plan = price_plans.get((item.get("price") or {}).get("id"))
                    if plan:
                        sub.plan = plan
                        break
                sub.status = status or "incomplete"

    return {"received": True}


@router.get("/portal")
async def get_billing_portal(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    import stripe
    stripe.api_key = settings.STRIPE_SECRET_KEY

    if not user.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No billing account found. Subscribe to a plan first.")

    session = stripe.billing_portal.Session.create(
        customer=user.stripe_customer_id,
        return_url=f"{settings.FRONTEND_URL}/billing",
    )
    return {"url": session.url}


@router.get("/usage", response_model=UsageResponse)
async def get_usage(
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    from datetime import datetime, timezone
    from sqlalchemy import func
    from backend.app.db.models import TrainedModel

    from backend.app.api.models import _user_plan
    plan = _user_plan(user)

    month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    count_result = await db.execute(
        select(func.count())
        .select_from(TrainedModel)
        .where(
            TrainedModel.user_id == user.id,
            TrainedModel.is_house_model == False,
            TrainedModel.created_at >= month_start,
        )
    )
    count = count_result.scalar() or 0
    limits = {"free": 0, "pro": 3, "elite": 999999}

    return UsageResponse(plan=plan, models_trained_this_month=count, models_limit=limits.get(plan, 0))
