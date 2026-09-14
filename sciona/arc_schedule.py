"""Four-phase adaptive ARC schedule derived from pinned safe_run.py (MIT).

Keeps source order 3,23,33,4; concurrency 4,4,4,2; baseline-derived estimates
and slack. Fixes orphaned processes and missing-baseline crashes at global
deadline. Resource-limited results are explicit; a missing result is never a
fabricated zero-grid prediction. No claim of identical wall-clock decisions.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
import time

from sciona.arc_contract import Candidate, Task, merge_answers
from sciona.arc_process import Attempt, ProcessBudget, run_once


SOURCE_SECONDS = 9 * 60 * 60 * .95
SOURCE_MEMORY_MIB = 4 * 4096 * .95


@dataclass(frozen=True)
class Command:
    test_index: int
    argument: int
    expected_seconds: float
    expected_memory_mib: float
    slack: float


@dataclass(frozen=True)
class ScheduleResult:
    attempts: tuple[tuple[int, Attempt], ...]
    predictions: tuple[tuple[Candidate, ...], ...]
    missing_test_indices: tuple[int, ...]
    completed_all_runs: bool


def commands_for_phase(argument: int, count: int, baseline: dict[int, Attempt],
                       seconds: float, memory_mib: float) -> tuple[list[Command], int]:
    if argument == 3:
        return [Command(index, 3, seconds, memory_mib, 1.5) for index in range(count)], 4
    if argument not in (23, 33, 4):
        raise ValueError("Unknown source phase")
    scale, slack, threads = (20, 2, 2) if argument == 4 else (2, 100, 4)
    commands = [Command(index, argument, item.seconds * scale, item.peak_rss_mib * scale, slack)
                for index, item in baseline.items()]
    # Python sorted in original is stable; baseline is inserted in input order.
    return sorted(commands, key=lambda command: command.expected_seconds), threads


def run_schedule(binary: Path, expected_sha256: str, task: Task, *,
                 seconds: float = SOURCE_SECONDS,
                 memory_mib: float = SOURCE_MEMORY_MIB) -> ScheduleResult:
    budget = ProcessBudget(seconds, memory_mib)
    results: list[tuple[int, Attempt]] = []
    baseline: dict[int, Attempt] = {}
    try:
        for argument in (3, 23, 33, 4):
            if time.monotonic() >= budget.deadline:
                break
            commands, concurrency = commands_for_phase(argument, len(task.test), baseline, seconds, memory_mib)
            iterator = iter(commands)
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                pending = {}

                def fill():
                    while len(pending) < concurrency and time.monotonic() < budget.deadline:
                        command = next(iterator, None)
                        if command is None:
                            break
                        timeout = command.expected_seconds * command.slack
                        memory = command.expected_memory_mib * command.slack
                        # A process completing between RSS samples can have zero
                        # measured memory; preserve zero derived cap as a limit
                        # failure rather than increasing source budgets silently.
                        if timeout <= 0 or memory <= 0:
                            status = "time_limit" if timeout <= 0 else "memory_limit"
                            results.append((command.test_index, Attempt(status, argument, 0, 0, ())))
                            continue
                        future = pool.submit(run_once, binary, expected_sha256, task,
                                             command.test_index, argument, timeout_seconds=timeout,
                                             memory_mib=memory, budget=budget)
                        pending[future] = command.test_index

                try:
                    fill()
                    while pending:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            index = pending.pop(future)
                            results.append((index, future.result()))
                        fill()
                except BaseException:
                    # Cancel before executor context waits for other workers.
                    budget.cancel()
                    raise
            if argument == 3:
                baseline = dict(sorted((index, attempt) for index, attempt in results))
    finally:
        budget.cancel()
    predictions = []
    missing = []
    for index in range(len(task.test)):
        successful = [attempt.candidates for i, attempt in results if i == index and attempt.status == "success"]
        if successful:
            predictions.append(merge_answers(successful))
        else:
            predictions.append(())
            missing.append(index)
    return ScheduleResult(tuple(results), tuple(predictions), tuple(missing),
                          len(results) == 4 * len(task.test) and all(item.status == "success" for _, item in results))
