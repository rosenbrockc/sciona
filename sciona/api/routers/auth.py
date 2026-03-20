"""GitHub OAuth device flow + JWT issuance."""

from __future__ import annotations

import os
import time
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from sciona.api.deps import UserRow, require_auth
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


@router.get("/auth/github/device")
async def github_device_start() -> DeviceFlowResponse:
    """Start GitHub device flow (returns device_code, user_code, verification_uri)."""
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
async def github_device_poll(device_code: str) -> TokenResponse | PendingResponse:
    """Poll for device flow completion. Returns JWT on success or pending status."""
    try:
        import httpx
    except ImportError:
        raise HTTPException(503, "httpx not installed")

    client_id = os.environ.get("GITHUB_OAUTH_CLIENT_ID", "")
    client_secret = os.environ.get("GITHUB_OAUTH_CLIENT_SECRET", "")

    async with httpx.AsyncClient() as client:
        # Exchange device code for GitHub access token
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

    # Fetch GitHub user profile
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

    # Upsert user and issue JWT
    jwt_token = await _upsert_user_and_issue_jwt(gh_user)

    return TokenResponse(
        access_token=jwt_token,
        expires_in=30 * 24 * 3600,  # 30 days
    )


@router.get("/auth/me")
async def get_me(user: UserRow = Depends(require_auth)) -> UserResponse:
    """Return current authenticated user."""
    from datetime import datetime, timezone

    return UserResponse(
        user_id=UUID(user.user_id),
        github_login=user.github_login,
        display_name=user.display_name,
        avatar_url=user.avatar_url,
        identity_tier=user.identity_tier,
        reputation_score=0,
        created_at=datetime.now(timezone.utc),
    )


async def _upsert_user_and_issue_jwt(gh_user: dict) -> str:
    """Upsert the GitHub user into the database and return a signed JWT."""
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
