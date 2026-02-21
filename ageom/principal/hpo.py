"""Hyperparameter optimisation via Optuna for the Principal role."""

from __future__ import annotations

import logging
import math
from typing import Any

import optuna
from optuna.importance import get_param_importances

from ageom.principal.models import BenchmarkResult
from ageom.synthesizer.ghost_sim import GhostSimReport

logger = logging.getLogger(__name__)


class TrialPrunedEarly(optuna.exceptions.TrialPruned):
    """Raised when a ghost-sim pre-check indicates the trial is hopeless."""


class OptunaManager:
    """Wraps an ``optuna.Study`` to drive structural HPO for the Principal.

    The manager provides:
    * Early pruning based on ``GhostSimReport`` — skip compilation when
      the ghost simulation already detected structural mismatches or
      infinite error bounds.
    * A thin reporting layer around ``optuna.importance.get_param_importances``
      for fANOVA-based structural importance analysis.
    """

    def __init__(
        self,
        study_name: str = "principal",
        *,
        direction: str = "minimize",
        storage: str | None = None,
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
    ) -> None:
        self._study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
        )

    @property
    def study(self) -> optuna.Study:
        """Access the underlying Optuna study."""
        return self._study

    # ------------------------------------------------------------------
    # Early pruning
    # ------------------------------------------------------------------

    @staticmethod
    def check_early_prune(sim_report: GhostSimReport) -> None:
        """Raise ``TrialPrunedEarly`` if the ghost report is hopeless.

        Conditions that trigger pruning:
        1. The simulation ran and *failed* (structural mismatch).
        2. Any node's precision gradient is infinite or NaN.
        """
        if sim_report.ran and not sim_report.passed:
            raise TrialPrunedEarly(f"Ghost simulation failed: {sim_report.error}")

        for nid, pg in sim_report.precision_gradients.items():
            if math.isinf(pg) or math.isnan(pg):
                raise TrialPrunedEarly(f"Infinite/NaN error bound at node '{nid}'")

    # ------------------------------------------------------------------
    # Importance analysis
    # ------------------------------------------------------------------

    def param_importances(
        self,
        *,
        evaluator: Any | None = None,
    ) -> dict[str, float]:
        """Return fANOVA parameter importances from completed trials.

        Args:
            evaluator: Optional custom importance evaluator.  Defaults to
                the fANOVA-based evaluator shipped with Optuna.

        Returns:
            Mapping of parameter name to importance score (0-1).
            Empty dict when fewer than 2 completed trials exist.
        """
        completed = [
            t for t in self._study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(completed) < 2:
            logger.info(
                "Fewer than 2 completed trials (%d); skipping importance.",
                len(completed),
            )
            return {}

        kwargs: dict[str, Any] = {"study": self._study}
        if evaluator is not None:
            kwargs["evaluator"] = evaluator

        return get_param_importances(**kwargs)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def report_trial(
        self,
        trial: optuna.Trial,
        benchmark: BenchmarkResult,
    ) -> None:
        """Report intermediate benchmark loss to the Optuna pruner."""
        trial.report(benchmark.global_loss, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
