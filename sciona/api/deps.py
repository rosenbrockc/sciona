"""Dependency injection for the platform API."""

from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

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
    is_blacklisted: bool = False


UserRow = UserProfile


def use_supabase_auth() -> bool:
    """Return whether Supabase Auth should be used for request auth."""
    return os.environ.get("SCIONA_USE_SUPABASE_AUTH", "0") == "1"


def use_supabase_db() -> bool:
    """Return whether Supabase-backed database code paths are enabled."""
    return os.environ.get("SCIONA_USE_SUPABASE_DB", "0") == "1"


def dual_write_enabled() -> bool:
    """Return whether legacy writes should be mirrored to Supabase PG."""
    return os.environ.get("SCIONA_DUAL_WRITE", "0") == "1"


def _read_source(router_name: str | None = None) -> str:
    """Resolve the current read source for the router-specific cutover flag."""
    if router_name:
        override = os.environ.get(
            f"SCIONA_READ_SOURCE_{router_name.upper()}",
            "",
        ).strip()
        if override:
            return override
    return os.environ.get("SCIONA_READ_SOURCE", "pg").strip() or "pg"


def _get_jwt_public_key() -> str:
    """Load the JWT public key from environment or file."""
    key = os.environ.get("SCIONA_JWT_PUBLIC_KEY", "")
    if key:
        return key
    key_path = os.environ.get("SCIONA_JWT_PUBLIC_KEY_PATH", "")
    if key_path and os.path.exists(key_path):
        with open(key_path) as f:
            return f.read()
    return ""


async def get_db(request: Request) -> Any:
    """Yield a connection from the asyncpg pool.

    The pool is stored on ``request.app.state.db_pool`` and initialised
    during the FastAPI lifespan.
    """
    pool = getattr(request.app.state, "db_pool", None)
    if pool is None:
        raise HTTPException(503, "Database not available")
    mirror_pool = getattr(request.app.state, "supabase_db_pool", None)
    async with pool.acquire() as conn:
        if dual_write_enabled() and mirror_pool is not None:
            from sciona.api.dual_write import DualWriteConnection

            async with mirror_pool.acquire() as mirror_conn:
                yield DualWriteConnection(conn, mirror_conn)
            return
        yield conn


async def get_read_db(
    request: Request,
    router_name: str | None = None,
) -> Any:
    """Yield the configured read connection for phased read cutover."""
    source = _read_source(router_name)
    if source == "supabase":
        pool = getattr(request.app.state, "supabase_db_pool", None)
        if pool is None:
            raise HTTPException(503, "Supabase read pool not available")
    else:
        pool = getattr(request.app.state, "db_pool", None)
        if pool is None:
            raise HTTPException(503, "Database not available")
    async with pool.acquire() as conn:
        yield conn


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
    if use_supabase_auth():
        return await _require_auth_supabase(request, credentials)
    return await _require_auth_legacy(request, credentials)


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


async def _require_auth_legacy(
    request: Request,
    credentials: HTTPAuthorizationCredentials,
) -> UserProfile:
    """Decode and validate the legacy platform JWT."""
    try:
        import jwt as pyjwt
    except ImportError:
        raise HTTPException(503, "PyJWT not installed")

    public_key = _get_jwt_public_key()
    if not public_key:
        raise HTTPException(503, "JWT public key not configured")

    token = credentials.credentials
    try:
        payload = pyjwt.decode(token, public_key, algorithms=["RS256"])
    except pyjwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired — run `sciona login`")
    except pyjwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")

    db = getattr(request.app.state, "db_pool", None)
    if db is None:
        raise HTTPException(503, "Database not available")

    async with db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM users WHERE user_id = $1::uuid",
            payload["sub"],
        )

    if not row:
        raise HTTPException(401, "User not found")
    if row["is_blacklisted"]:
        raise HTTPException(403, "Account suspended")

    return UserProfile(**dict(row))
