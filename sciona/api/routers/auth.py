"""Authentication endpoints for the platform API."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from sciona.api.deps import UserProfile, require_auth
from sciona.api.models import (
    DeviceFlowResponse,
    PendingResponse,
    TokenResponse,
    UserResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)

GITHUB_DEVICE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"


def _request_supabase(request: Request) -> Any | None:
    state = getattr(request.app, "state", None)
    if state is None:
        return None
    return getattr(state, "supabase", None) or getattr(state, "supabase_admin", None)


def _attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _current_span():
    try:
        from opentelemetry import trace
    except ImportError:
        return None
    try:
        return trace.get_current_span()
    except Exception:
        return None


def _annotate_span(**attributes: Any) -> None:
    span = _current_span()
    if span is None:
        return
    for key, value in attributes.items():
        if value is None:
            continue
        try:
            span.set_attribute(key, value)
        except Exception:
            logger.debug("Failed to set span attribute %s", key, exc_info=True)


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
async def login_redirect(request: Request) -> dict[str, str]:
    """Return a Supabase GitHub OAuth URL for browser-based login."""
    _annotate_span(**{"auth.flow": "supabase_oauth", "auth.provider": "github"})
    supabase = _request_supabase(request)
    if supabase is None:
        raise HTTPException(503, "Supabase OAuth is not available")
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
    _annotate_span(**{"auth.flow": "github_device"})
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
    """Poll the device flow and return a Supabase session token."""
    _annotate_span(
        **{
            "auth.flow": "github_device",
            "auth.device_code_present": bool(device_code),
        }
    )
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
        user_resp.json()

    supabase = _request_supabase(request)
    if supabase is None:
        raise HTTPException(503, "Supabase Auth is not available")

    try:
        session_response = await supabase.auth.sign_in_with_id_token(
            {"provider": "github", "token": github_token}
        )
    except Exception as exc:
        raise HTTPException(503, "Supabase device login failed") from exc

    token_response = _token_response_from_session(session_response)
    if token_response is None:
        raise HTTPException(503, "Supabase session was not returned")
    return token_response


@router.get("/auth/me")
async def get_me(user: UserProfile = Depends(require_auth)) -> UserResponse:
    """Return the current authenticated user."""
    _annotate_span(**{"auth.flow": "me", "user.id": str(user.user_id)})
    return UserResponse(
        user_id=UUID(str(user.user_id)),
        github_login=user.github_login,
        display_name=user.display_name,
        avatar_url=user.avatar_url,
        identity_tier=user.identity_tier,
        effective_tier=user.effective_tier,
        reputation_score=user.reputation_score,
        created_at=user.created_at,
    )
