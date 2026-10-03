from typing import Optional
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.auth.jwt import decode_token
from backend.app.db.models import User
from backend.app.db.session import get_db

bearer_scheme = HTTPBearer(auto_error=False)

PLAN_HIERARCHY = {"free": 0, "pro": 1, "elite": 2}


async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    if creds is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    payload = decode_token(creds.credentials)
    if payload is None or payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    user_id = UUID(payload["sub"])
    result = await db.execute(select(User).where(User.id == user_id, User.is_active == True))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


async def get_optional_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    if creds is None:
        return None
    try:
        return await get_current_user(creds=creds, db=db)
    except HTTPException:
        return None


def require_plan(minimum: str):
    """Dependency factory: require at least the given plan tier."""
    min_level = PLAN_HIERARCHY[minimum]

    async def checker(user: User = Depends(get_current_user)) -> User:
        if getattr(user, "is_admin", False):
            return user
        plan = "free"
        if user.subscription and user.subscription.status == "active":
            plan = user.subscription.plan
        if PLAN_HIERARCHY.get(plan, 0) < min_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires a {minimum} plan or higher",
            )
        return user

    return checker


async def require_admin(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user
