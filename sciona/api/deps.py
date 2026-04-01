"""Dependency injection for the platform API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

bearer_scheme = HTTPBearer()


class UserProfile(BaseModel):
    """Minimal public.users profile representation."""

    user_id: str
    github_id: int = 0
    github_login: str = ""
    display_name: str = ""
    avatar_url: str = ""
    email: str = ""
    identity_tier: str = "contributor"
    effective_tier: str = "general"
    reputation_score: int = 0
    is_blacklisted: bool = False
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


UserRow = UserProfile


async def get_supabase(request: Request) -> Any:
    """Return the configured Supabase client from app state."""
    client = getattr(request.app.state, "supabase_admin", None)
    if client is None:
        client = getattr(request.app.state, "supabase", None)
    if client is None:
        raise HTTPException(503, "Supabase client not available")
    return client


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> UserProfile:
    """Validate the configured auth token and return the user profile."""
    return await _require_auth_supabase(request, credentials)


async def _require_auth_supabase(
    request: Request,
    credentials: HTTPAuthorizationCredentials,
) -> UserProfile:
    """Validate a Supabase JWT and return the caller's public profile."""
    supabase = await get_supabase(request)
    token = credentials.credentials
    try:
        user_response = await supabase.auth.get_user(token)
    except Exception as exc:  # pragma: no cover - exact SDK exceptions vary
        raise HTTPException(401, "Invalid or expired token") from exc

    auth_user = getattr(user_response, "user", None)
    if auth_user is None:
        raise HTTPException(401, "Invalid token")

    result = (
        await supabase.table("users")
        .select("*")
        .eq("user_id", str(auth_user.id))
        .maybe_single()
        .execute()
    )
    data = getattr(result, "data", None)
    if not data:
        raise HTTPException(401, "User profile not found")
    if data.get("is_blacklisted"):
        raise HTTPException(403, "Account suspended")

    return UserProfile(**data)
