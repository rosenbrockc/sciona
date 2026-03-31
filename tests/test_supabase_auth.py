from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from sciona.api import deps
from sciona.api.routers import auth as auth_router


class _FakeSupabaseQuery:
    def __init__(self, data):
        self._data = data

    def select(self, _fields: str):
        return self

    def eq(self, _field: str, _value):
        return self

    def maybe_single(self):
        return self

    async def execute(self):
        return SimpleNamespace(data=self._data)


class _FakeSupabase:
    def __init__(self, *, profile: dict | None = None, session=None, user_id: str = "user-1"):
        self._profile = profile
        self._session = session
        self._user_id = user_id
        self.auth = SimpleNamespace(
            get_user=self._get_user,
            sign_in_with_id_token=self._sign_in_with_id_token,
        )

    async def _get_user(self, _token: str):
        return SimpleNamespace(user=SimpleNamespace(id=self._user_id))

    async def _sign_in_with_id_token(self, _payload: dict):
        return SimpleNamespace(session=self._session)

    def table(self, _name: str):
        return _FakeSupabaseQuery(self._profile)


class _Acquire:
    def __init__(self, row):
        self.row = row

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def fetchrow(self, _sql: str, _user_id: str):
        return self.row


class _Pool:
    def __init__(self, row):
        self.row = row

    def acquire(self):
        return _Acquire(self.row)


@pytest.mark.asyncio
async def test_require_auth_supabase_valid_token(monkeypatch) -> None:
    monkeypatch.setenv("SCIONA_USE_SUPABASE_AUTH", "1")
    profile = {
        "user_id": "user-1",
        "github_id": 1,
        "github_login": "alice",
        "display_name": "Alice",
        "avatar_url": "",
        "email": "alice@example.com",
        "identity_tier": "contributor",
        "effective_tier": "early_access",
        "is_blacklisted": False,
    }
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supabase=_FakeSupabase(profile=profile))))
    credentials = SimpleNamespace(credentials="token")

    result = await deps.require_auth(request, credentials=credentials)

    assert result.github_login == "alice"
    assert result.effective_tier == "early_access"


@pytest.mark.asyncio
async def test_require_auth_supabase_blacklisted(monkeypatch) -> None:
    monkeypatch.setenv("SCIONA_USE_SUPABASE_AUTH", "1")
    profile = {
        "user_id": "user-1",
        "github_id": 1,
        "github_login": "alice",
        "display_name": "Alice",
        "avatar_url": "",
        "email": "alice@example.com",
        "identity_tier": "contributor",
        "effective_tier": "general",
        "is_blacklisted": True,
    }
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supabase=_FakeSupabase(profile=profile))))
    credentials = SimpleNamespace(credentials="token")

    with pytest.raises(HTTPException) as exc:
        await deps.require_auth(request, credentials=credentials)

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_require_auth_legacy_still_works(monkeypatch) -> None:
    monkeypatch.setenv("SCIONA_USE_SUPABASE_AUTH", "0")
    monkeypatch.setattr(deps, "_get_jwt_public_key", lambda: "pubkey")

    class _JWT:
        class ExpiredSignatureError(Exception):
            pass

        class InvalidTokenError(Exception):
            pass

        @staticmethod
        def decode(token: str, key: str, algorithms: list[str]):
            assert token == "legacy-token"
            assert key == "pubkey"
            assert algorithms == ["RS256"]
            return {"sub": "user-legacy"}

    import sys

    monkeypatch.setitem(sys.modules, "jwt", _JWT)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                db_pool=_Pool(
                    {
                        "user_id": "user-legacy",
                        "github_id": 2,
                        "github_login": "legacy",
                        "display_name": "Legacy",
                        "avatar_url": "",
                        "email": "legacy@example.com",
                        "identity_tier": "contributor",
                        "effective_tier": "general",
                        "is_blacklisted": False,
                    }
                )
            )
        )
    )
    credentials = SimpleNamespace(credentials="legacy-token")

    result = await deps.require_auth(request, credentials=credentials)

    assert result.user_id == "user-legacy"
    assert result.github_login == "legacy"


@pytest.mark.asyncio
async def test_device_flow_returns_supabase_session(monkeypatch) -> None:
    monkeypatch.setenv("SCIONA_USE_SUPABASE_AUTH", "1")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "client-id")
    session = SimpleNamespace(
        access_token="supabase-access",
        refresh_token="supabase-refresh",
        expires_in=3600,
    )
    supabase = _FakeSupabase(profile=None, session=session)

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self._payload

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, **_kwargs):
            if "access_token" in url:
                return _Response({"access_token": "github-token"})
            return _Response({"access_token": "github-token"})

        async def get(self, _url: str, **_kwargs):
            return _Response({"id": 123, "login": "alice"})

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: _Client())
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(supabase=supabase)))

    result = await auth_router.github_device_poll("device-code", request)

    assert result.access_token == "supabase-access"
    assert result.refresh_token == "supabase-refresh"


@pytest.mark.asyncio
async def test_auth_me_returns_effective_tier() -> None:
    user = deps.UserProfile(
        user_id="11111111-1111-1111-1111-111111111111",
        github_id=1,
        github_login="alice",
        display_name="Alice",
        avatar_url="",
        email="alice@example.com",
        identity_tier="contributor",
        effective_tier="internal",
        is_blacklisted=False,
    )

    result = await auth_router.get_me(user=user)

    assert result.effective_tier == "internal"
