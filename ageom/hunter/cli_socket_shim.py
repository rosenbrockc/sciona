"""Daemon-backed socket shim for Claude and Codex CLI providers."""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


def _env_int(name: str, default: int, *, min_value: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, value)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass
class _SocketWorker:
    process: asyncio.subprocess.Process
    socket_path: Path
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    pid: int | None = None
    pending: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    request_id: int = 0


class CLISocketShimClient:
    """Persistent socket shim around CLI-backed daemon workers."""

    def __init__(
        self,
        *,
        cli: str,
        model: str,
        max_tokens: int,
        pool_size: int | None = None,
        use_agent_layer: bool = False,
    ) -> None:
        self._cli = cli
        self._model = model
        self._max_tokens = max_tokens
        self._use_agent_layer = use_agent_layer
        self._telemetry_provider = f"{cli}_shim"
        self._telemetry_model = model
        self._pool_size = pool_size or _env_int("AGEOM_CLI_SHIM_POOL_SIZE", 2)
        self._timeout_s = _env_float("AGEOM_SUBPROCESS_TIMEOUT_S", 90.0)
        runtime_root = os.getenv("AGEOM_CLI_SHIM_RUNTIME_DIR", "/tmp")
        self._runtime_dir = Path(tempfile.mkdtemp(prefix=f"acs-{cli}-", dir=runtime_root))
        self._daemon_script = Path(__file__).with_name("cli_daemon.py")
        self._python_bin = sys.executable or "python"
        self._init_lock = asyncio.Lock()
        self._workers: list[_SocketWorker] = []
        self._next_worker = itertools.count()
        self._closed = False
        self._last_completion_metadata: dict[str, object] = {}
        self._last_error_metadata: dict[str, object] = {}

    async def complete(self, system: str, user: str) -> str:
        return await self._complete_rpc("complete", system, user, "")

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self._complete_rpc("complete_with_grammar", system, user, grammar)

    def get_last_completion_metadata(self) -> dict[str, object]:
        return dict(self._last_completion_metadata)

    def get_last_error_metadata(self) -> dict[str, object]:
        return dict(self._last_error_metadata)

    async def warmup(self) -> None:
        """Spawn and connect the persistent worker pool up front."""
        await self._ensure_workers()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in list(self._workers):
            await self._shutdown_worker(worker)
        self._workers.clear()
        shutil.rmtree(self._runtime_dir, ignore_errors=True)

    async def _complete_rpc(self, method: str, system: str, user: str, grammar: str) -> str:
        self._last_error_metadata = {}
        worker = await self._acquire_worker()
        try:
            result = await self._rpc(
                worker,
                method,
                {
                    "system": system,
                    "user": user,
                    "grammar": grammar,
                },
            )
        except Exception as exc:
            if self._should_retry_transport_error(exc):
                worker = await self._replace_worker(worker)
                try:
                    result = await self._rpc(
                        worker,
                        method,
                        {
                            "system": system,
                            "user": user,
                            "grammar": grammar,
                        },
                    )
                except Exception as retry_exc:
                    self._last_error_metadata = {
                        "provider_error_phase": "rpc",
                        "provider_transport": "socket_shim",
                        "provider_cli": self._cli,
                        "provider_model": self._model,
                        "shim_worker_pid": worker.pid,
                        "shim_pool_size": self._pool_size,
                        "provider_error_excerpt": str(retry_exc)[:240],
                    }
                    raise
            else:
                self._last_error_metadata = {
                    "provider_error_phase": "rpc",
                    "provider_transport": "socket_shim",
                    "provider_cli": self._cli,
                    "provider_model": self._model,
                    "shim_worker_pid": worker.pid,
                    "shim_pool_size": self._pool_size,
                    "provider_error_excerpt": str(exc)[:240],
                }
                raise
        self._last_completion_metadata = {
            "shim_provider": f"{self._cli}_shim",
            "shim_worker_pid": result.get("pid"),
            "shim_request_count": result.get("requestCount"),
            "shim_pool_size": self._pool_size,
            "shim_was_cold_start": bool(result.get("coldStart", False)),
        }
        text = result.get("text", "")
        return text if isinstance(text, str) else str(text)

    async def _ensure_workers(self) -> None:
        if self._workers:
            return
        async with self._init_lock:
            if self._workers:
                return
            for index in range(self._pool_size):
                self._workers.append(await self._spawn_worker(index))

    async def _acquire_worker(self) -> _SocketWorker:
        await self._ensure_workers()
        start = next(self._next_worker) % len(self._workers)
        ordered = self._workers[start:] + self._workers[:start]
        worker = min(ordered, key=lambda item: item.pending)
        worker.pending += 1
        return worker

    async def _replace_worker(self, worker: _SocketWorker) -> _SocketWorker:
        """Replace a failed worker in-place and return the new one."""
        try:
            index = self._workers.index(worker)
        except ValueError:
            index = len(self._workers)
        try:
            await self._shutdown_worker(worker)
        except Exception:
            pass
        replacement = await self._spawn_worker(index)
        if index < len(self._workers):
            self._workers[index] = replacement
        else:
            self._workers.append(replacement)
        replacement.pending += 1
        return replacement

    def _should_retry_transport_error(self, exc: Exception) -> bool:
        """Return True when a socket transport failure should trigger a retry."""
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionError, OSError)):
            return True
        message = str(exc).lower()
        return (
            "socket closed unexpectedly" in message
            or "broken pipe" in message
            or "connection reset" in message
            or "transport endpoint is not connected" in message
        )

    async def _spawn_worker(self, index: int) -> _SocketWorker:
        socket_path = self._runtime_dir / f"worker-{index}.sock"
        process = await asyncio.create_subprocess_exec(
            self._python_bin,
            str(self._daemon_script),
            "--socket",
            str(socket_path),
            "--cli",
            self._cli,
            "--model",
            self._model,
            "--max-tokens",
            str(self._max_tokens),
            *(
                ["--use-agent-layer"]
                if self._use_agent_layer
                else []
            ),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=dict(os.environ),
        )
        try:
            reader, writer = await self._wait_for_socket(process, socket_path)
        except Exception:
            await self._terminate_process(process)
            stderr_text = await self._read_stream(process.stderr)
            self._last_error_metadata = {
                "provider_error_phase": "startup",
                "provider_transport": "socket_shim",
                "provider_cli": self._cli,
                "provider_model": self._model,
                "provider_exit_code": process.returncode,
                "provider_stderr_excerpt": stderr_text.strip()[:240],
            }
            raise RuntimeError(
                f"Failed to start {self._cli} shim worker: {stderr_text.strip()}"
            ) from None

        worker = _SocketWorker(
            process=process,
            socket_path=socket_path,
            reader=reader,
            writer=writer,
            pid=process.pid,
        )
        await self._rpc(worker, "ping", {})
        return worker

    async def _wait_for_socket(
        self,
        process: asyncio.subprocess.Process,
        socket_path: Path,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        deadline = asyncio.get_running_loop().time() + self._timeout_s
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                stderr_text = await self._read_stream(process.stderr)
                raise RuntimeError(
                    f"{self._cli} shim worker exited early with code {process.returncode}: {stderr_text.strip()}"
                )
            if socket_path.exists():
                try:
                    return await asyncio.open_unix_connection(str(socket_path))
                except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
                    last_error = exc
            await asyncio.sleep(0.05)
        raise RuntimeError(f"Timed out waiting for {self._cli} shim socket {socket_path}: {last_error}")

    async def _rpc(
        self,
        worker: _SocketWorker,
        method: str,
        params: dict[str, object],
    ) -> dict[str, object]:
        async with worker.lock:
            worker.request_id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": worker.request_id,
                "method": method,
                "params": params,
            }
            try:
                worker.writer.write((json.dumps(payload) + "\n").encode())
                await worker.writer.drain()
                raw = await asyncio.wait_for(worker.reader.readline(), timeout=self._timeout_s)
            finally:
                worker.pending = max(0, worker.pending - 1)
            if not raw:
                raise RuntimeError(f"{self._cli} shim socket closed unexpectedly")
            response = json.loads(raw.decode())
            if "error" in response:
                error = response["error"]
                message = error.get("message", "unknown error") if isinstance(error, dict) else str(error)
                raise RuntimeError(message)
            result = response.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"Invalid {self._cli} shim response: {response!r}")
            return result

    async def _shutdown_worker(self, worker: _SocketWorker) -> None:
        try:
            worker.writer.close()
            await worker.writer.wait_closed()
        except Exception:
            pass
        await self._terminate_process(worker.process)
        try:
            worker.socket_path.unlink(missing_ok=True)
        except Exception:
            pass

    async def _terminate_process(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                return
            await process.wait()

    async def _read_stream(self, stream: asyncio.StreamReader | None) -> str:
        if stream is None:
            return ""
        try:
            data = await asyncio.wait_for(stream.read(), timeout=0.2)
        except Exception:
            return ""
        return data.decode(errors="replace")
