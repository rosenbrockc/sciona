"""Shared context primitives for cross-agent prompt augmentation."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_TABLE_NAME_RE = re.compile(r"[a-z_][a-z0-9_]*$")


@dataclass(frozen=True)
class ContextRecord:
    """A single memory record used for prompt-time retrieval."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SharedContextStore(Protocol):
    """Protocol for shared context stores."""

    async def put(
        self,
        namespace: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write a context record into a namespace."""
        ...

    async def recent(self, namespace: str, *, limit: int = 5) -> list[ContextRecord]:
        """Return the most recent records for a namespace."""
        ...

    async def search(
        self,
        namespace: str,
        query: str,
        *,
        limit: int = 5,
    ) -> list[ContextRecord]:
        """Return the best-matching records for a query within a namespace."""
        ...


@dataclass
class SharedContextMetrics:
    """Lightweight counters for shared-context usage."""

    backend: str = "memory"
    searches_total: int = 0
    search_hits: int = 0
    search_misses: int = 0
    search_errors: int = 0
    search_latency_ms_total: float = 0.0
    puts_total: int = 0
    put_errors: int = 0
    put_latency_ms_total: float = 0.0
    recents_total: int = 0
    recent_latency_ms_total: float = 0.0
    injected_blocks: int = 0
    injected_records: int = 0
    injected_chars: int = 0
    template_searches_total: int = 0
    template_search_hits: int = 0
    template_puts_total: int = 0
    template_injected_blocks: int = 0
    template_injected_records: int = 0
    template_injected_chars: int = 0
    failure_searches_total: int = 0
    failure_search_hits: int = 0
    failure_puts_total: int = 0
    failure_injected_blocks: int = 0
    failure_injected_records: int = 0
    failure_injected_chars: int = 0
    duplicate_candidates: int = 0
    duplicates_suppressed: int = 0
    match_with_context_total: int = 0
    match_with_context_success: int = 0
    match_without_context_total: int = 0
    match_without_context_success: int = 0
    promotions_total: int = 0

    def record_search(self, *, latency_ms: float, hits: int, error: bool = False) -> None:
        self.searches_total += 1
        self.search_latency_ms_total += max(0.0, latency_ms)
        if error:
            self.search_errors += 1
            self.search_misses += 1
            return
        if hits > 0:
            self.search_hits += 1
        else:
            self.search_misses += 1

    def record_put(self, *, latency_ms: float, error: bool = False) -> None:
        self.puts_total += 1
        self.put_latency_ms_total += max(0.0, latency_ms)
        if error:
            self.put_errors += 1

    def record_recent(self, *, latency_ms: float) -> None:
        self.recents_total += 1
        self.recent_latency_ms_total += max(0.0, latency_ms)

    def record_injection(self, *, chars: int, records: int) -> None:
        if chars <= 0:
            return
        self.injected_blocks += 1
        self.injected_chars += chars
        self.injected_records += max(0, records)

    def record_template_search(self, *, hits: int) -> None:
        self.template_searches_total += 1
        if hits > 0:
            self.template_search_hits += 1

    def record_template_put(self) -> None:
        self.template_puts_total += 1

    def record_template_injection(self, *, chars: int, records: int) -> None:
        if chars <= 0:
            return
        self.template_injected_blocks += 1
        self.template_injected_chars += chars
        self.template_injected_records += max(0, records)

    def record_failure_search(self, *, hits: int) -> None:
        self.failure_searches_total += 1
        if hits > 0:
            self.failure_search_hits += 1

    def record_failure_put(self) -> None:
        self.failure_puts_total += 1

    def record_failure_injection(self, *, chars: int, records: int) -> None:
        if chars <= 0:
            return
        self.failure_injected_blocks += 1
        self.failure_injected_chars += chars
        self.failure_injected_records += max(0, records)

    def record_duplicate_filter(self, *, candidates: int, suppressed: int) -> None:
        self.duplicate_candidates += max(0, candidates)
        self.duplicates_suppressed += max(0, suppressed)

    def record_match_outcome(self, *, used_context: bool, success: bool) -> None:
        if used_context:
            self.match_with_context_total += 1
            if success:
                self.match_with_context_success += 1
            return
        self.match_without_context_total += 1
        if success:
            self.match_without_context_success += 1

    def record_promotion(self) -> None:
        self.promotions_total += 1

    def snapshot(self) -> dict[str, float | int | str]:
        avg_search_ms = (
            self.search_latency_ms_total / self.searches_total
            if self.searches_total
            else 0.0
        )
        avg_put_ms = (
            self.put_latency_ms_total / self.puts_total if self.puts_total else 0.0
        )
        hit_rate = self.search_hits / self.searches_total if self.searches_total else 0.0
        duplicate_suppression_rate = (
            self.duplicates_suppressed / self.duplicate_candidates
            if self.duplicate_candidates
            else 0.0
        )
        success_with_context_rate = (
            self.match_with_context_success / self.match_with_context_total
            if self.match_with_context_total
            else 0.0
        )
        success_without_context_rate = (
            self.match_without_context_success / self.match_without_context_total
            if self.match_without_context_total
            else 0.0
        )
        template_hit_rate = (
            self.template_search_hits / self.template_searches_total
            if self.template_searches_total
            else 0.0
        )
        failure_hit_rate = (
            self.failure_search_hits / self.failure_searches_total
            if self.failure_searches_total
            else 0.0
        )
        return {
            "backend": self.backend,
            "searches_total": self.searches_total,
            "search_hits": self.search_hits,
            "search_misses": self.search_misses,
            "search_errors": self.search_errors,
            "search_hit_rate": hit_rate,
            "search_latency_ms_avg": avg_search_ms,
            "puts_total": self.puts_total,
            "put_errors": self.put_errors,
            "put_latency_ms_avg": avg_put_ms,
            "recents_total": self.recents_total,
            "injected_blocks": self.injected_blocks,
            "injected_records": self.injected_records,
            "injected_chars": self.injected_chars,
            "template_searches_total": self.template_searches_total,
            "template_search_hits": self.template_search_hits,
            "template_hit_rate": template_hit_rate,
            "template_puts_total": self.template_puts_total,
            "template_injected_blocks": self.template_injected_blocks,
            "template_injected_records": self.template_injected_records,
            "template_injected_chars": self.template_injected_chars,
            "failure_searches_total": self.failure_searches_total,
            "failure_search_hits": self.failure_search_hits,
            "failure_hit_rate": failure_hit_rate,
            "failure_puts_total": self.failure_puts_total,
            "failure_injected_blocks": self.failure_injected_blocks,
            "failure_injected_records": self.failure_injected_records,
            "failure_injected_chars": self.failure_injected_chars,
            "duplicate_candidates": self.duplicate_candidates,
            "duplicates_suppressed": self.duplicates_suppressed,
            "duplicate_suppression_rate": duplicate_suppression_rate,
            "match_with_context_total": self.match_with_context_total,
            "match_with_context_success": self.match_with_context_success,
            "match_without_context_total": self.match_without_context_total,
            "match_without_context_success": self.match_without_context_success,
            "success_rate_with_context": success_with_context_rate,
            "success_rate_without_context": success_without_context_rate,
            "match_success_delta": (
                success_with_context_rate - success_without_context_rate
            ),
            "promotions_total": self.promotions_total,
        }


class InMemorySharedContextStore:
    """Simple append-only in-memory context store.

    This is intended as a default local backend and test utility.
    """

    def __init__(self, *, max_records_per_namespace: int = 500) -> None:
        self._data: dict[str, list[ContextRecord]] = {}
        self._lock = asyncio.Lock()
        self._max = max_records_per_namespace

    async def put(
        self,
        namespace: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not namespace.strip() or not text.strip():
            return
        rec = ContextRecord(text=text.strip(), metadata=dict(metadata or {}))
        async with self._lock:
            bucket = self._data.setdefault(namespace, [])
            bucket.append(rec)
            if len(bucket) > self._max:
                self._data[namespace] = bucket[-self._max :]

    async def recent(self, namespace: str, *, limit: int = 5) -> list[ContextRecord]:
        async with self._lock:
            bucket = list(self._data.get(namespace, []))
        if limit <= 0:
            return []
        return list(reversed(bucket[-limit:]))

    async def search(
        self,
        namespace: str,
        query: str,
        *,
        limit: int = 5,
    ) -> list[ContextRecord]:
        if limit <= 0:
            return []
        q_tokens = _tokenize(query)
        async with self._lock:
            bucket = list(self._data.get(namespace, []))
        if not bucket:
            return []
        if not q_tokens:
            return list(reversed(bucket[-limit:]))

        scored: list[tuple[float, ContextRecord]] = []
        n = len(bucket)
        for idx, rec in enumerate(bucket):
            base = _overlap_score(q_tokens, _tokenize(rec.text))
            if base <= 0.0:
                continue
            # Slight recency bias: more recent entries rank higher on ties.
            recency = (idx + 1) / max(n, 1)
            scored.append((base + (0.01 * recency), rec))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [rec for _, rec in scored[:limit]]


class PostgresSharedContextStore:
    """PostgreSQL-backed shared context store.

    Uses simple append-only writes and recency-biased overlap scoring.
    """

    def __init__(
        self,
        postgres_uri: str,
        *,
        table_name: str = "ageom_shared_context",
        scan_limit: int = 120,
        max_records_per_namespace: int = 500,
        ttl_hours: int = 0,
    ) -> None:
        self._postgres_uri = postgres_uri.strip()
        self._table_name = table_name.strip().lower()
        if not self._postgres_uri:
            raise ValueError("postgres_uri must be non-empty")
        if not _TABLE_NAME_RE.fullmatch(self._table_name):
            raise ValueError(
                f"Invalid shared-context table name: {table_name!r}; use [a-z0-9_]"
            )
        self._scan_limit = max(20, int(scan_limit))
        self._max = max(50, int(max_records_per_namespace))
        self._ttl_hours = max(0, int(ttl_hours))

    async def setup(self) -> None:
        import psycopg
        from psycopg import sql

        table = sql.Identifier(self._table_name)
        idx_ns = sql.Identifier(self._index_name("namespace_id_idx"))
        idx_tokens = sql.Identifier(self._index_name("tokens_gin_idx"))
        async with await psycopg.AsyncConnection.connect(self._postgres_uri) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            id BIGSERIAL PRIMARY KEY,
                            namespace TEXT NOT NULL,
                            text TEXT NOT NULL,
                            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                            tokens TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                            access_count BIGINT NOT NULL DEFAULT 0,
                            last_accessed TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    ).format(table)
                )
                await cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (namespace, id DESC)").format(
                        idx_ns,
                        table,
                    )
                )
                await cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING GIN(tokens)").format(
                        idx_tokens,
                        table,
                    )
                )
                await cur.execute(
                    sql.SQL(
                        "ALTER TABLE {} "
                        "ADD COLUMN IF NOT EXISTS access_count BIGINT NOT NULL DEFAULT 0"
                    ).format(table)
                )
                await cur.execute(
                    sql.SQL(
                        "ALTER TABLE {} "
                        "ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ "
                        "NOT NULL DEFAULT NOW()"
                    ).format(table)
                )
            await conn.commit()

    async def put(
        self,
        namespace: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        namespace = namespace.strip()
        text = text.strip()
        if not namespace or not text:
            return
        import psycopg
        from psycopg import sql
        from psycopg.types.json import Jsonb

        tokens = sorted(_tokenize(text))
        table = sql.Identifier(self._table_name)
        async with await psycopg.AsyncConnection.connect(self._postgres_uri) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql.SQL(
                        "INSERT INTO {} (namespace, text, metadata, tokens, access_count, last_accessed) "
                        "VALUES (%s, %s, %s, %s, 0, NOW())"
                    ).format(table),
                    (namespace, text, Jsonb(dict(metadata or {})), tokens),
                )
                if self._ttl_hours > 0:
                    await cur.execute(
                        sql.SQL(
                            "DELETE FROM {} WHERE namespace = %s "
                            "AND created_at < NOW() - (%s || ' hours')::interval"
                        ).format(table),
                        (namespace, int(self._ttl_hours)),
                    )
                await cur.execute(
                    sql.SQL(
                        """
                        DELETE FROM {}
                        WHERE namespace = %s
                        AND id NOT IN (
                            SELECT id FROM {}
                            WHERE namespace = %s
                            ORDER BY
                                COALESCE((metadata->>'confidence')::double precision, 0.0) DESC,
                                access_count DESC,
                                created_at DESC
                            LIMIT %s
                        )
                        """
                    ).format(table, table),
                    (namespace, namespace, self._max),
                )
            await conn.commit()

    async def recent(self, namespace: str, *, limit: int = 5) -> list[ContextRecord]:
        namespace = namespace.strip()
        if not namespace or limit <= 0:
            return []
        import psycopg
        from psycopg import sql

        table = sql.Identifier(self._table_name)
        async with await psycopg.AsyncConnection.connect(self._postgres_uri) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql.SQL(
                        "SELECT id, text, metadata "
                        "FROM {} WHERE namespace = %s "
                        "ORDER BY id DESC LIMIT %s"
                    ).format(table),
                    (namespace, int(limit)),
                )
                rows = await cur.fetchall()
                ids = [int(r[0]) for r in rows if r and r[0] is not None]
                if ids:
                    await cur.execute(
                        sql.SQL(
                            "UPDATE {} SET access_count = access_count + 1, "
                            "last_accessed = NOW() WHERE id = ANY(%s)"
                        ).format(table),
                        (ids,),
                    )
            await conn.commit()
        return [
            ContextRecord(text=str(r[1]), metadata=_normalize_metadata(r[2])) for r in rows
        ]

    async def search(
        self,
        namespace: str,
        query: str,
        *,
        limit: int = 5,
    ) -> list[ContextRecord]:
        namespace = namespace.strip()
        if not namespace or limit <= 0:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return await self.recent(namespace, limit=limit)

        import psycopg
        from psycopg import sql

        table = sql.Identifier(self._table_name)
        fetch_limit = max(self._scan_limit, int(limit) * 20)
        async with await psycopg.AsyncConnection.connect(self._postgres_uri) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql.SQL(
                        "SELECT id, text, metadata, tokens "
                        "FROM {} WHERE namespace = %s "
                        "ORDER BY id DESC LIMIT %s"
                    ).format(table),
                    (namespace, fetch_limit),
                )
                rows = await cur.fetchall()

        if not rows:
            return []

        scored: list[tuple[float, ContextRecord]] = []
        selected_ids: list[int] = []
        n = len(rows)
        for idx, row in enumerate(rows):
            row_id = int(row[0])
            text = str(row[1])
            metadata = _normalize_metadata(row[2])
            tokens = set(str(t) for t in (row[3] or []))
            base = _overlap_score(q_tokens, tokens)
            if base <= 0.0:
                continue
            recency = (n - idx) / max(n, 1)
            score = base + (0.01 * recency)
            scored.append((score, ContextRecord(text=text, metadata=metadata)))
            selected_ids.append(row_id)

        scored.sort(key=lambda item: item[0], reverse=True)
        out = [rec for _, rec in scored[:limit]]
        if selected_ids:
            async with await psycopg.AsyncConnection.connect(self._postgres_uri) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        sql.SQL(
                            "UPDATE {} SET access_count = access_count + 1, "
                            "last_accessed = NOW() WHERE id = ANY(%s)"
                        ).format(table),
                        (selected_ids,),
                    )
                await conn.commit()
        return out

    def _index_name(self, suffix: str) -> str:
        return f"{self._table_name}_{suffix}"[:63]


class InstrumentedSharedContextStore:
    """SharedContextStore wrapper that records operational metrics."""

    def __init__(self, inner: SharedContextStore, metrics: SharedContextMetrics) -> None:
        self._inner = inner
        self._metrics = metrics

    async def put(
        self,
        namespace: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        t0 = time.perf_counter()
        try:
            await self._inner.put(namespace, text, metadata=metadata)
        except Exception:
            self._metrics.record_put(
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=True,
            )
            raise
        self._metrics.record_put(latency_ms=(time.perf_counter() - t0) * 1000.0)

    async def recent(self, namespace: str, *, limit: int = 5) -> list[ContextRecord]:
        t0 = time.perf_counter()
        rows = await self._inner.recent(namespace, limit=limit)
        self._metrics.record_recent(latency_ms=(time.perf_counter() - t0) * 1000.0)
        return rows

    async def search(
        self,
        namespace: str,
        query: str,
        *,
        limit: int = 5,
    ) -> list[ContextRecord]:
        t0 = time.perf_counter()
        try:
            rows = await self._inner.search(namespace, query, limit=limit)
        except Exception:
            self._metrics.record_search(
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                hits=0,
                error=True,
            )
            raise
        self._metrics.record_search(
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            hits=len(rows),
        )
        return rows


class PolicySharedContextStore:
    """Policy wrapper for promotion and provenance metadata."""

    def __init__(
        self,
        inner: SharedContextStore,
        *,
        promotion_enabled: bool,
        promotion_min_confidence: float,
        repo_namespace: str,
        metrics: SharedContextMetrics | None = None,
    ) -> None:
        self._inner = inner
        self._promotion_enabled = bool(promotion_enabled)
        self._promotion_min_conf = float(promotion_min_confidence)
        self._repo_ns = repo_namespace.strip().strip("/")
        self._metrics = metrics

    async def put(
        self,
        namespace: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        namespace = namespace.strip()
        if not namespace:
            return
        md = dict(metadata or {})
        md.setdefault("created_at_unix", time.time())
        md.setdefault("source_namespace", namespace)
        channel = namespace.split("/")[-1] if "/" in namespace else namespace
        md.setdefault("source_channel", channel)
        confidence = _coerce_confidence(md.get("confidence"), namespace)
        md["confidence"] = confidence
        await self._inner.put(namespace, text, metadata=md)

        if not self._promotion_enabled or not self._repo_ns:
            return
        if namespace.startswith(self._repo_ns + "/"):
            return
        if confidence < self._promotion_min_conf:
            return
        promoted_md = dict(md)
        promoted_md["promoted_from"] = namespace
        promoted_md["promoted_at_unix"] = time.time()
        promoted_ns = f"{self._repo_ns}/{channel}"
        await self._inner.put(promoted_ns, text, metadata=promoted_md)
        if self._metrics is not None:
            self._metrics.record_promotion()

    async def recent(self, namespace: str, *, limit: int = 5) -> list[ContextRecord]:
        return await self._inner.recent(namespace, limit=limit)

    async def search(
        self,
        namespace: str,
        query: str,
        *,
        limit: int = 5,
    ) -> list[ContextRecord]:
        return await self._inner.search(namespace, query, limit=limit)


async def create_shared_context_store(
    *,
    enabled: bool,
    backend: str = "auto",
    postgres_uri: str = "",
    postgres_table: str = "ageom_shared_context",
    max_records_per_namespace: int = 500,
    ttl_hours: int = 0,
    promotion_enabled: bool = False,
    promotion_min_confidence: float = 0.9,
    repo_namespace: str = "repo/default",
    metrics: SharedContextMetrics | None = None,
) -> SharedContextStore | None:
    """Build a shared-context store with optional Postgres persistence.

    Backend selection:
    - ``postgres``: force Postgres (fallbacks to memory on failure).
    - ``memory``: force in-memory.
    - ``auto``: use Postgres when ``postgres_uri`` is set, else memory.
    """
    if not enabled:
        return None

    resolved = (backend or "auto").strip().lower()
    if resolved not in {"auto", "memory", "postgres"}:
        resolved = "auto"

    store: SharedContextStore | None = None
    wants_postgres = resolved == "postgres" or (
        resolved == "auto" and bool(postgres_uri.strip())
    )

    if wants_postgres and postgres_uri.strip():
        try:
            pg_store = PostgresSharedContextStore(
                postgres_uri=postgres_uri,
                table_name=postgres_table,
                max_records_per_namespace=max_records_per_namespace,
                ttl_hours=ttl_hours,
            )
            await pg_store.setup()
            store = pg_store
            if metrics is not None:
                metrics.backend = "postgres"
        except Exception as exc:
            logger.warning(
                "Failed to initialize Postgres shared context; using memory backend: %s",
                exc,
            )

    if store is None:
        store = InMemorySharedContextStore(
            max_records_per_namespace=max_records_per_namespace
        )
        if metrics is not None:
            metrics.backend = "memory"

    store = PolicySharedContextStore(
        store,
        promotion_enabled=promotion_enabled,
        promotion_min_confidence=promotion_min_confidence,
        repo_namespace=repo_namespace,
        metrics=metrics,
    )

    if metrics is not None:
        return InstrumentedSharedContextStore(store, metrics)
    return store


def format_context_block(
    title: str,
    records: list[ContextRecord],
    *,
    max_chars: int = 900,
    metrics: SharedContextMetrics | None = None,
    include_provenance: bool | None = None,
) -> str:
    """Render shared context records into a compact prompt block."""
    if not records:
        return ""
    if include_provenance is None:
        env_val = os.getenv("AGEOM_SHARED_CONTEXT_INCLUDE_PROVENANCE", "1").strip().lower()
        include_provenance = env_val not in {"0", "false", "no", "off"}
    deduped = _dedupe_records(records)
    if metrics is not None:
        metrics.record_duplicate_filter(
            candidates=len(records),
            suppressed=max(0, len(records) - len(deduped)),
        )
    lines = [f"## {title}"]
    for rec in deduped:
        text = " ".join(rec.text.split())
        if len(text) > 220:
            text = text[:217] + "..."
        if include_provenance:
            prov = _provenance_label(rec.metadata)
            if prov:
                lines.append(f"- {prov} {text}")
            else:
                lines.append(f"- {text}")
        else:
            lines.append(f"- {text}")
    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[: max_chars - 3] + "..."
    if metrics is not None:
        metrics.record_injection(chars=len(block), records=len(deduped))
    return block


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _overlap_score(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


def _normalize_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _coerce_confidence(raw: Any, namespace: str) -> float:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        lower_ns = namespace.lower()
        if lower_ns.endswith("/success"):
            return 0.95
        if lower_ns.endswith("/failure"):
            return 0.35
        if lower_ns.endswith("/critique"):
            return 0.75
        return 0.6
    return min(1.0, max(0.0, val))


def _dedupe_records(records: list[ContextRecord]) -> list[ContextRecord]:
    seen: set[str] = set()
    out: list[ContextRecord] = []
    for rec in records:
        key = " ".join(rec.text.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def _provenance_label(metadata: dict[str, Any]) -> str:
    if not metadata:
        return ""
    channel = str(metadata.get("source_channel", "")).strip()
    source_ns = str(metadata.get("source_namespace", "")).strip()
    confidence = metadata.get("confidence")
    parts: list[str] = []
    if channel:
        parts.append(f"ch:{channel}")
    elif source_ns:
        parts.append(f"ns:{source_ns.split('/')[-1]}")
    try:
        if confidence is not None:
            parts.append(f"conf:{float(confidence):.2f}")
    except (TypeError, ValueError):
        pass
    promoted_from = str(metadata.get("promoted_from", "")).strip()
    if promoted_from:
        parts.append("promoted")
    if not parts:
        return ""
    return "[" + " ".join(parts) + "]"
