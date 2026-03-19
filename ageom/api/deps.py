"""Dependency injection for the platform API."""

from __future__ import annotations

import os
from typing import Any, AsyncGenerator

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

bearer_scheme = HTTPBearer()


class UserRow(BaseModel):
    """Minimal user representation from the database."""

    user_id: str
    github_id: int = 0
    github_login: str = ""
    display_name: str = ""
    avatar_url: str = ""
    email: str = ""
    identity_tier: str = "contributor"
    is_blacklisted: bool = False


def _get_jwt_public_key() -> str:
    """Load the JWT public key from environment or file."""
    key = os.environ.get("AGEOM_JWT_PUBLIC_KEY", "")
    if key:
        return key
    key_path = os.environ.get("AGEOM_JWT_PUBLIC_KEY_PATH", "")
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
    async with pool.acquire() as conn:
        yield conn


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> UserRow:
    """Decode and validate platform JWT. Returns UserRow or raises 401."""
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
        raise HTTPException(401, "Token expired — run `ageom login`")
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

    return UserRow(**dict(row))
