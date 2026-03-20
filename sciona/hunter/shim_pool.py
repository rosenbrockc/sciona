"""Pre-warming subprocess pool for CLI-based LLM agents (claude, codex, gemini).

Instead of spawning a new process per LLM call (~30-60s startup), this module
keeps a pool of idle subprocesses that block on stdin, ready to serve a prompt
instantly.  After each call the pool eagerly pre-warms a replacement.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from asyncio.subprocess import PIPE, Process
from typing import Dict, List, Tuple

from sciona.hunter.llm import SubprocessCLIClient

logger = logging.getLogger(__name__)


class ShimPoolClient:
    """LLM client backed by a pool of pre-warmed CLI subprocesses.

    Supports the same CLI variants as ``SubprocessCLIClient`` (claude, codex,
    gemini) but amortises process-startup cost by keeping idle processes ready.
    """

    def __init__(
        self,
        cli: str,
        model: str,
        max_tokens: int,
        *,
        pool_size: int = 2,
        use_agent_layer: bool = False,
    ) -> None:
        self._delegate = SubprocessCLIClient(
            cli=cli,
            model=model,
            max_tokens=max_tokens,
            use_agent_layer=use_agent_layer,
        )
        self._pool_size = pool_size
        # Pool: argv-hash → list of idle processes
        self._pool: Dict[str, List[Process]] = {}
        self._env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        self._warming_tasks: List[asyncio.Task] = []  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Pool helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cmd_key(cmd: list[str]) -> str:
        """Deterministic hash of a command argv for pool keying."""
        return hashlib.sha256(" ".join(cmd).encode()).hexdigest()[:16]

    async def _spawn(self, cmd: list[str]) -> Process:
        """Spawn a subprocess that blocks on stdin."""
        return await asyncio.create_subprocess_exec(
            *cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, env=self._env,
        )

    def _prewarm_bg(self, system: str) -> None:
        """Schedule background pre-warming for the given system prompt."""
        task = asyncio.get_event_loop().create_task(self._prewarm(system))
        self._warming_tasks.append(task)
        task.add_done_callback(lambda t: self._warming_tasks.remove(t) if t in self._warming_tasks else None)

    async def _prewarm(self, system: str) -> None:
        """Pre-warm up to ``pool_size`` idle processes for *system*."""
        cmd = self._delegate._build_cmd(system)
        key = self._cmd_key(cmd)
        bucket = self._pool.setdefault(key, [])

        # Only fill up to pool_size
        needed = self._pool_size - len(bucket)
        for _ in range(needed):
            try:
                proc = await self._spawn(cmd)
                bucket.append(proc)
                logger.debug("Pre-warmed %s process (pool %s, size %d)", self._delegate._cli, key[:8], len(bucket))
            except Exception:
                logger.warning("Failed to pre-warm %s process", self._delegate._cli, exc_info=True)

    def _acquire(self, cmd: list[str]) -> Process | None:
        """Pop an idle pre-warmed process for *cmd*, or return None."""
        key = self._cmd_key(cmd)
        bucket = self._pool.get(key, [])
        while bucket:
            proc = bucket.pop(0)
            if proc.returncode is None:  # still alive
                return proc
            logger.debug("Discarding dead pre-warmed process")
        return None

    # ------------------------------------------------------------------
    # LLMClient protocol
    # ------------------------------------------------------------------

    async def complete(self, system: str, user: str) -> str:
        cmd = self._delegate._build_cmd(system)
        stdin_text = self._delegate._build_stdin(system, user)
        attempts = max(1, self._delegate._max_retries + 1)
        last_error = ""

        for attempt in range(1, attempts + 1):
            proc = self._acquire(cmd)
            if proc is not None:
                logger.debug("Using pre-warmed %s process", self._delegate._cli)
            else:
                logger.debug(
                    "No pre-warmed process available; cold-starting %s",
                    self._delegate._cli,
                )
                proc = await self._spawn(cmd)

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(stdin_text.encode()),
                    timeout=self._delegate._timeout_s,
                )
            except asyncio.TimeoutError as exc:
                await self._delegate._terminate_process(proc)
                self._delegate._log_subprocess_event(
                    "PROMPT_SUBPROCESS_TIMEOUT",
                    {
                        "cli": self._delegate._cli,
                        "model": self._delegate._model,
                        "attempt": attempt,
                        "attempts_total": attempts,
                        "timeout_s": self._delegate._timeout_s,
                        "client": "shim_pool",
                    },
                )
                last_error = (
                    f"{self._delegate._cli} CLI timed out after "
                    f"{self._delegate._timeout_s:.1f}s (attempt {attempt}/{attempts})"
                )
                if attempt >= attempts:
                    raise RuntimeError(last_error) from exc
                await self._delegate._sleep_before_retry(attempt)
                continue

            stderr_text = stderr.decode().strip()
            if proc.returncode != 0:
                last_error = (
                    f"{self._delegate._cli} CLI exited with code "
                    f"{proc.returncode}: {stderr_text}"
                )
                if (
                    attempt < attempts
                    and self._delegate._is_transient_error(stderr_text)
                ):
                    self._delegate._log_subprocess_event(
                        "PROMPT_SUBPROCESS_RETRY",
                        {
                            "cli": self._delegate._cli,
                            "model": self._delegate._model,
                            "attempt": attempt,
                            "attempts_total": attempts,
                            "reason": stderr_text[:200],
                            "client": "shim_pool",
                        },
                    )
                    await self._delegate._sleep_before_retry(attempt)
                    continue
                raise RuntimeError(last_error)

            # Eagerly pre-warm a replacement.
            self._prewarm_bg(system)
            return self._delegate._parse_output(stdout.decode())

        raise RuntimeError(last_error or f"{self._delegate._cli} CLI failed without output")

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        # CLI tools have no GBNF support; fall back to plain completion.
        return await self.complete(system, user)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def warm(self, system_prompts: list[str] | None = None) -> None:
        """Explicitly pre-warm the pool for known system prompts."""
        if not system_prompts:
            return
        await asyncio.gather(*(self._prewarm(s) for s in system_prompts))

    async def close(self) -> None:
        """Kill all idle pre-warmed processes and cancel warming tasks."""
        for task in list(self._warming_tasks):
            task.cancel()
        self._warming_tasks.clear()

        for bucket in self._pool.values():
            for proc in bucket:
                if proc.returncode is None:
                    try:
                        proc.kill()
                        await proc.wait()
                    except ProcessLookupError:
                        pass
        self._pool.clear()
