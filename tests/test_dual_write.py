from __future__ import annotations

import pytest

from sciona.api import deps
from sciona.api.dual_write import (
    DualWriteConnection,
    get_dual_write_metrics,
    reset_dual_write_metrics,
)


class _Conn:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, str, tuple]] = []
        self.fail_on = fail_on

    async def fetch(self, query: str, *args):
        self.calls.append(("fetch", query, args))
        return [{"ok": True}]

    async def fetchrow(self, query: str, *args):
        self.calls.append(("fetchrow", query, args))
        if self.fail_on == "fetchrow":
            raise RuntimeError("mirror failed")
        return {"ok": True}

    async def fetchval(self, query: str, *args):
        self.calls.append(("fetchval", query, args))
        if self.fail_on == "fetchval":
            raise RuntimeError("mirror failed")
        return 1

    async def execute(self, query: str, *args):
        self.calls.append(("execute", query, args))
        if self.fail_on == "execute":
            raise RuntimeError("mirror failed")
        return "OK"


@pytest.mark.asyncio
async def test_dual_write_execute_mirrors_mutations() -> None:
    reset_dual_write_metrics()
    primary = _Conn()
    mirror = _Conn()
    conn = DualWriteConnection(primary, mirror)

    result = await conn.execute("UPDATE atoms SET description = $1", "x")

    assert result == "OK"
    assert primary.calls == [("execute", "UPDATE atoms SET description = $1", ("x",))]
    assert mirror.calls == [("execute", "UPDATE atoms SET description = $1", ("x",))]
    metrics = get_dual_write_metrics()
    assert metrics["dual_write_attempts"] == 1
    assert metrics["dual_write_failures"] == 0


@pytest.mark.asyncio
async def test_dual_write_fetch_does_not_mirror_reads() -> None:
    reset_dual_write_metrics()
    primary = _Conn()
    mirror = _Conn()
    conn = DualWriteConnection(primary, mirror)

    rows = await conn.fetch("SELECT * FROM atoms")

    assert rows == [{"ok": True}]
    assert primary.calls == [("fetch", "SELECT * FROM atoms", ())]
    assert mirror.calls == []
    assert get_dual_write_metrics()["dual_write_attempts"] == 0


@pytest.mark.asyncio
async def test_dual_write_mirror_failures_do_not_raise() -> None:
    reset_dual_write_metrics()
    primary = _Conn()
    mirror = _Conn(fail_on="execute")
    conn = DualWriteConnection(primary, mirror)

    result = await conn.execute("DELETE FROM atoms WHERE atom_id = $1", "a1")

    assert result == "OK"
    metrics = get_dual_write_metrics()
    assert metrics["dual_write_attempts"] == 1
    assert metrics["dual_write_failures"] == 1


def test_read_source_resolution(monkeypatch) -> None:
    monkeypatch.delenv("SCIONA_READ_SOURCE", raising=False)
    monkeypatch.delenv("SCIONA_READ_SOURCE_CATALOG", raising=False)
    assert deps._read_source() == "pg"

    monkeypatch.setenv("SCIONA_READ_SOURCE", "supabase")
    assert deps._read_source() == "supabase"

    monkeypatch.setenv("SCIONA_READ_SOURCE_CATALOG", "pg")
    assert deps._read_source("catalog") == "pg"
