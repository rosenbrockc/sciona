"""Authentication endpoints for the platform API."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from sciona.api.deps import UserProfile, get_supabase, require_auth, use_supabase_auth
from sciona.api.models import (
    DeviceFlowResponse,
    PendingResponse,
    TokenResponse,
    UserResponse,
)

router = APIRouter()

GITHUB_DEVICE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"


def _request_supabase(request: Request) -> Any | None:
    state = getattr(request.app, "state", None)
    if state is None:
        return None
    return getattr(state, "supabase_admin", None) or getattr(state, "supabase", None)


def _attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _token_response_from_session(session_response: Any) -> TokenResponse | None:
    session = _attr_or_key(session_response, "session")
    if session is None:
        data = _attr_or_key(session_response, "data")
        session = _attr_or_key(data, "session")
    if session is None:
        return None

    access_token = _attr_or_key(session, "access_token")
    if not access_token:
        return None

    refresh_token = _attr_or_key(session, "refresh_token", "")
    expires_in = _attr_or_key(session, "expires_in")
    if expires_in is None:
        expires_at = _attr_or_key(session, "expires_at")
        if expires_at is not None:
            expires_in = max(int(float(expires_at) - time.time()), 0)
    if expires_in is None:
        expires_in = 30 * 24 * 3600

    return TokenResponse(
        access_token=str(access_token),
        refresh_token=str(refresh_token or ""),
        expires_in=int(expires_in),
    )


@router.get("/auth/login")
async def login_redirect(supabase=Depends(get_supabase)) -> dict[str, str]:
    """Return a Supabase GitHub OAuth URL for browser-based login."""
    redirect_to = os.environ.get(
        "SCIONA_SUPABASE_REDIRECT_URL",
        os.environ.get("SUPABASE_REDIRECT_URL", "http://localhost:5173/auth/callback"),
    )
    try:
        response = await supabase.auth.sign_in_with_oauth(
            {"provider": "github", "options": {"redirect_to": redirect_to}}
        )
    except Exception as exc:
        raise HTTPException(503, "Supabase OAuth is not available") from exc

    url = _attr_or_key(response, "url")
    if not url:
        data = _attr_or_key(response, "data")
        url = _attr_or_key(data, "url")
    if not url:
        raise HTTPException(500, "Supabase OAuth URL not returned")
    return {"url": str(url)}


@router.get("/auth/github/device")
async def github_device_start() -> DeviceFlowResponse:
    """Start the legacy GitHub device flow used by the CLI."""
    try:
        import httpx
    except ImportError:
        raise HTTPException(503, "httpx not installed")

    client_id = os.environ.get("GITHUB_OAUTH_CLIENT_ID", "")
    if not client_id:
        raise HTTPException(503, "GitHub OAuth not configured")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GITHUB_DEVICE_URL,
            data={"client_id": client_id, "scope": "read:user user:email"},
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

    return DeviceFlowResponse(
        device_code=data["device_code"],
        user_code=data["user_code"],
        verification_uri=data["verification_uri"],
        expires_in=data.get("expires_in", 900),
        interval=data.get("interval", 5),
    )


@router.post("/auth/github/device/poll")
async def github_device_poll(
    device_code: str,
    request: Request,
) -> TokenResponse | PendingResponse:
    """Poll the device flow and return a Supabase or legacy session token."""
    try:
        import httpx
    except ImportError:
        raise HTTPException(503, "httpx not installed")

    client_id = os.environ.get("GITHUB_OAUTH_CLIENT_ID", "")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id": client_id,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

    if "error" in data:
        if data["error"] == "authorization_pending":
            return PendingResponse(interval=data.get("interval", 5))
        if data["error"] == "slow_down":
            return PendingResponse(
                status="slow_down", interval=data.get("interval", 10)
            )
        raise HTTPException(400, f"GitHub OAuth error: {data['error']}")

    github_token = data["access_token"]

    try:
        import httpx as httpx_mod
    except ImportError:
        raise HTTPException(503, "httpx not installed")

    async with httpx_mod.AsyncClient() as client:
        user_resp = await client.get(
            GITHUB_USER_URL,
            headers={
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/json",
            },
        )
        user_resp.raise_for_status()
        gh_user = user_resp.json()

    if use_supabase_auth():
        supabase = _request_supabase(request)
        if supabase is not None:
            try:
                session_response = await supabase.auth.sign_in_with_id_token(
                    {"provider": "github", "token": github_token}
                )
            except Exception:
                session_response = None
            token_response = _token_response_from_session(session_response)
            if token_response is not None:
                return token_response

    jwt_token = await _upsert_user_and_issue_jwt(gh_user)
    return TokenResponse(
        access_token=jwt_token,
        refresh_token="",
        expires_in=30 * 24 * 3600,
    )


@router.get("/auth/me")
async def get_me(user: UserProfile = Depends(require_auth)) -> UserResponse:
    """Return the current authenticated user."""
    return UserResponse(
        user_id=UUID(str(user.user_id)),
        github_login=user.github_login,
        display_name=user.display_name,
        avatar_url=user.avatar_url,
        identity_tier=user.identity_tier,
        effective_tier=user.effective_tier,
        reputation_score=0,
        created_at=datetime.now(timezone.utc),
    )


async def _upsert_user_and_issue_jwt(gh_user: dict[str, Any]) -> str:
    """Upsert the GitHub user into the legacy JWT identity flow."""
    try:
        import jwt as pyjwt
    except ImportError:
        raise HTTPException(503, "PyJWT not installed")

    private_key = os.environ.get("SCIONA_JWT_PRIVATE_KEY", "")
    if not private_key:
        key_path = os.environ.get("SCIONA_JWT_PRIVATE_KEY_PATH", "")
        if key_path and os.path.exists(key_path):
            with open(key_path) as f:
                private_key = f.read()
    if not private_key:
        raise HTTPException(503, "JWT private key not configured")

    now = int(time.time())
    payload = {
        "sub": str(gh_user.get("id", "")),
        "ghid": gh_user.get("id", 0),
        "login": gh_user.get("login", ""),
        "tier": "contributor",
        "iat": now,
        "exp": now + 30 * 24 * 3600,
    }

    return pyjwt.encode(payload, private_key, algorithm="RS256")
