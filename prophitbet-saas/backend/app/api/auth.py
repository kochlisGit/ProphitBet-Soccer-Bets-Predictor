from typing import Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.dependencies import get_current_user
from backend.app.auth.jwt import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from backend.app.db.models import Subscription, User
from backend.app.db.session import get_db
from backend.app.config import get_settings

router = APIRouter()


# --------------- Schemas ---------------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


REFRESH_COOKIE = "prophitbet_refresh"


def _set_refresh_cookie(response: Response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        httponly=True,
        secure=settings.ENVIRONMENT.lower() not in {"development", "dev", "test"},
        samesite="lax",
        path="/",
    )


DEFAULT_NOTIFICATION_PREFS = {
    "notify_daily_digest": True,
    "notify_training": True,
    "notify_weekly_report": False,
}


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    avatar_url: Optional[str]
    provider: str
    plan: str
    is_admin: bool
    preferences: dict

    class Config:
        from_attributes = True


def _user_preferences_dict(user: User) -> dict:
    merged = dict(DEFAULT_NOTIFICATION_PREFS)
    if user.preferences and isinstance(user.preferences, dict):
        for k in merged:
            if k in user.preferences:
                merged[k] = bool(user.preferences[k])
    return merged


# --------------- Routes ---------------

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, response: Response, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    if len(body.password) < 8:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Password must be at least 8 characters")

    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        name=body.name,
        provider="email",
    )
    db.add(user)
    await db.flush()

    sub = Subscription(user_id=user.id, plan="free", status="active")
    db.add(sub)
    await db.flush()

    _set_refresh_cookie(response, create_refresh_token(user.id))
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, response: Response, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if user is None or user.password_hash is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    _set_refresh_cookie(response, create_refresh_token(user.id))
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    response: Response,
    refresh_cookie: Optional[str] = Cookie(None, alias=REFRESH_COOKIE),
    db: AsyncSession = Depends(get_db),
):
    payload = decode_token(refresh_cookie) if refresh_cookie else None
    if payload is None or payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    from uuid import UUID
    user_id = UUID(payload["sub"])
    result = await db.execute(select(User).where(User.id == user_id, User.is_active == True))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    _set_refresh_cookie(response, create_refresh_token(user.id))
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response):
    response.delete_cookie(
        key=REFRESH_COOKIE,
        path="/",
        httponly=True,
        samesite="lax",
    )


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    notify_daily_digest: Optional[bool] = None
    notify_training: Optional[bool] = None
    notify_weekly_report: Optional[bool] = None


@router.patch("/me", response_model=UserResponse)
async def update_me(
    body: UpdateProfileRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if body.name is not None:
        user.name = body.name

    merged = _user_preferences_dict(user)
    if body.notify_daily_digest is not None:
        merged["notify_daily_digest"] = body.notify_daily_digest
    if body.notify_training is not None:
        merged["notify_training"] = body.notify_training
    if body.notify_weekly_report is not None:
        merged["notify_weekly_report"] = body.notify_weekly_report
    if any(
        getattr(body, k) is not None
        for k in ("notify_daily_digest", "notify_training", "notify_weekly_report")
    ):
        user.preferences = merged
    await db.flush()

    plan = "free"
    if user.subscription and user.subscription.status == "active":
        plan = user.subscription.plan
    return UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        avatar_url=user.avatar_url,
        provider=user.provider,
        plan=plan,
        is_admin=user.is_admin,
        preferences=_user_preferences_dict(user),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(get_current_user)):
    plan = "free"
    if user.subscription and user.subscription.status == "active":
        plan = user.subscription.plan
    return UserResponse(
        id=str(user.id),
        email=user.email,
        name=user.name,
        avatar_url=user.avatar_url,
        provider=user.provider,
        plan=plan,
        is_admin=user.is_admin,
        preferences=_user_preferences_dict(user),
    )
