"""Isolated, monitored execution of a hash-bound ARC source binary.

The caller supplies a reviewed build identity, never an end-user executable.
Compared with safe_run.py, every timed-out/over-memory process is killed and
reaped. Raw task data, source diagnostics and temporary outputs are not retained.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import threading
import time

import psutil

from sciona.arc_contract import Candidate, Task, parse_answers


@dataclass(frozen=True)
class Attempt:
    status: str
    argument: int
    seconds: float
    peak_rss_mib: float
    candidates: tuple[Candidate, ...]


class ProcessBudget:
    """Shared sampled-RSS cap; select the largest live process as in source."""

    def __init__(self, seconds: float, memory_mib: float):
        for value in (seconds, memory_mib):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Global limits must be finite and positive")
        self.deadline = time.monotonic() + seconds
        self.memory_mib = memory_mib
        self._lock = threading.Lock()
        self._rss: dict[object, float] = {}
        self._victims: set[object] = set()
        self._cancelled = False

    def observe(self, token: object, rss: float) -> str | None:
        with self._lock:
            if self._cancelled:
                return "cancelled"
            if time.monotonic() >= self.deadline:
                return "time_limit"
            self._rss[token] = rss
            active = {key: value for key, value in self._rss.items() if key not in self._victims}
            if sum(active.values()) > self.memory_mib:
                self._victims.add(max(active, key=active.get))
            return "memory_limit" if token in self._victims else None

    def release(self, token: object) -> None:
        with self._lock:
            self._rss.pop(token, None)
            self._victims.discard(token)

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True


def run_once(binary: Path, expected_sha256: str, task: Task, test_index: int,
             argument: int, *, timeout_seconds: float, memory_mib: float,
             budget: ProcessBudget | None = None) -> Attempt:
    if type(argument) is not int or argument not in {2, 3, 4, 12, 23, 33}:
        raise ValueError("Unsupported reviewed source argument")
    for limit in (timeout_seconds, memory_mib):
        if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
            raise ValueError("Process limits must be finite and positive")
    payload = task.source_task(test_index)
    binary = Path(binary).resolve(strict=True)
    data = binary.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("ARC binary hash mismatch")
    with tempfile.TemporaryDirectory(prefix="sciona_arc_process_") as temp:
        work = Path(temp)
        # Copy verified bytes so changing the build cache cannot race execution.
        executable = work / "solver"
        executable.write_bytes(data)
        executable.chmod(0o700)
        task_dir = work / "dataset/evaluation"
        task_dir.mkdir(parents=True)
        (work / "output").mkdir()
        (task_dir / "00000000.json").write_text(json.dumps(payload, separators=(",", ":")))
        started = time.monotonic()
        peak = 0.0
        token = object()
        if budget is not None:
            reason = budget.observe(token, 0.0)
            if reason:
                budget.release(token)
                return Attempt(reason, argument, 0.0, 0.0, ())
        try:
            process = subprocess.Popen([str(executable), "0", str(argument)], cwd=work,
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
        except BaseException:
            if budget is not None:
                budget.release(token)
            raise
        status = "running"
        try:
            monitor = psutil.Process(process.pid)
            while process.poll() is None:
                elapsed = time.monotonic() - started
                try:
                    rss = monitor.memory_info().rss / 2**20
                except psutil.NoSuchProcess:
                    # Reap through Popen; disappearance alone is not success.
                    process.wait()
                    break
                peak = max(peak, rss)
                if budget is not None:
                    reason = budget.observe(token, rss)
                    if reason:
                        status = reason
                        break
                if rss > memory_mib:
                    status = "memory_limit"
                    break
                if elapsed >= timeout_seconds:
                    status = "time_limit"
                    break
                time.sleep(min(0.01, max(0.0001, timeout_seconds - elapsed)))
            if status != "running":
                if process.poll() is None:
                    process.kill()
                process.wait()
                return Attempt(status, argument, time.monotonic() - started, peak, ())
            code = process.wait()
            if code != 0:
                raise RuntimeError(f"ARC solver exited with code {code}")
            output = work / f"output/answer_0_{argument}.csv"
            if not output.is_file() or output.stat().st_size > 4096:
                raise RuntimeError("ARC solver did not produce a valid answer file")
            try:
                candidates = parse_answers(output.read_text())
            except (ValueError, UnicodeError) as error:
                raise RuntimeError("ARC solver produced corrupt answers") from error
            return Attempt("success", argument, time.monotonic() - started, peak, candidates)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            if budget is not None:
                budget.release(token)
