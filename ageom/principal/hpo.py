"""Hyperparameter optimisation via Optuna for the Principal role."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

try:
    import optuna
    from optuna.importance import get_param_importances
except ImportError:  # pragma: no cover - exercised in runtime envs without optuna
    optuna = None

from ageom.principal.models import BenchmarkResult
from ageom.architect.handoff import CDGExport
from ageom.architect.catalog import PrimitiveCatalog
from ageom.synthesizer.ghost_sim import GhostSimReport

logger = logging.getLogger(__name__)


class TrialPrunedEarly(Exception):
    """Raised when a ghost-sim pre-check indicates the trial is hopeless."""


@dataclass(frozen=True)
class SuggestedParams:
    """Concrete parameter sample for a single Principal trial."""

    signature: str
    trial_number: int
    assignments: dict[str, dict[str, Any]]


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
        sampler: Any | None = None,
        pruner: Any | None = None,
    ) -> None:
        self._study_name = study_name
        self._direction = direction
        self._storage = storage
        self._sampler = sampler
        self._pruner = pruner
        self._studies: dict[str, Any] = {}
        self._active_trials: dict[tuple[str, int], Any] = {}
        self._study = self._ensure_study("global")

    @property
    def study(self) -> Any:
        """Access the underlying Optuna study."""
        return self._study

    def _ensure_study(self, signature: str) -> Any:
        existing = self._studies.get(signature)
        if existing is not None:
            return existing
        study_name = (
            self._study_name
            if signature == "global"
            else f"{self._study_name}:{signature}"
        )
        if optuna is None:
            study = _FallbackStudy(study_name=study_name)
        else:
            study = optuna.create_study(
                study_name=study_name,
                direction=self._direction,
                storage=self._storage,
                sampler=self._sampler,
                pruner=self._pruner,
                load_if_exists=True,
            )
        self._studies[signature] = study
        return study

    def suggest_node_params(
        self,
        *,
        signature: str,
        cdg: CDGExport,
        catalog: PrimitiveCatalog,
    ) -> SuggestedParams:
        """Sample tunables for all atomic nodes in *cdg* from the scoped study."""
        study = self._ensure_study(signature)
        trial = study.ask()
        self._active_trials[(signature, trial.number)] = trial

        assignments: dict[str, dict[str, Any]] = {}
        for node in cdg.nodes:
            primitive_name = str(getattr(node, "matched_primitive", "") or "").strip()
            if not primitive_name:
                continue
            primitive = catalog.get(primitive_name)
            if primitive is None or not primitive.tunable_params:
                continue
            node_assignment: dict[str, Any] = {}
            for spec in primitive.tunable_params:
                suggested = self._suggest_param(
                    trial,
                    name=f"{node.node_id}.{spec.name}",
                    spec=spec,
                )
                node_assignment[spec.name] = suggested
            if node_assignment:
                assignments[node.node_id] = node_assignment

        return SuggestedParams(
            signature=signature,
            trial_number=trial.number,
            assignments=assignments,
        )

    def complete_trial(
        self,
        *,
        signature: str,
        trial_number: int | None,
        loss: float,
    ) -> None:
        """Tell the scoped study the final loss for a completed trial."""
        if trial_number is None:
            return
        key = (signature, trial_number)
        trial = self._active_trials.pop(key, None)
        if trial is None:
            return
        study = self._ensure_study(signature)
        study.tell(trial, loss)

    def prune_trial(
        self,
        *,
        signature: str,
        trial_number: int | None,
    ) -> None:
        """Mark a scoped trial as pruned."""
        if trial_number is None:
            return
        key = (signature, trial_number)
        trial = self._active_trials.pop(key, None)
        if trial is None:
            return
        study = self._ensure_study(signature)
        if optuna is None:
            study.tell(trial, state="pruned")
            return
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)

    @staticmethod
    def _suggest_param(trial: Any, *, name: str, spec: Any) -> Any:
        """Sample a single param from an Optuna trial, falling back to defaults."""
        use_default = getattr(trial, "number", 0) == 0
        if spec.kind == "bool":
            choices = list(spec.choices or [True, False])
            suggested = trial.suggest_categorical(name, choices)
            return spec.default if use_default and spec.default in choices else suggested
        if spec.kind == "categorical":
            choices = list(spec.choices or [spec.default])
            suggested = trial.suggest_categorical(name, choices)
            return spec.default if use_default and spec.default in choices else suggested
        if spec.kind == "int":
            if spec.min_value is None or spec.max_value is None:
                return spec.default
            suggested = trial.suggest_int(
                name,
                int(spec.min_value),
                int(spec.max_value),
                step=int(spec.step or 1),
                log=bool(spec.log_scale),
            )
            default_value = int(spec.default)
            if use_default and int(spec.min_value) <= default_value <= int(spec.max_value):
                return default_value
            return suggested
        if spec.kind == "float":
            if spec.min_value is None or spec.max_value is None:
                return spec.default
            kwargs: dict[str, Any] = {"log": bool(spec.log_scale)}
            if spec.step is not None:
                kwargs["step"] = float(spec.step)
            suggested = trial.suggest_float(
                name,
                float(spec.min_value),
                float(spec.max_value),
                **kwargs,
            )
            default_value = float(spec.default)
            if use_default and float(spec.min_value) <= default_value <= float(spec.max_value):
                return default_value
            return suggested
        return spec.default

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
                conf = sim_report.node_confidence.get(nid, 1.0)
                if conf > 0.3:
                    raise TrialPrunedEarly(
                        f"Infinite/NaN error bound at node '{nid}'"
                    )
                logger.warning(
                    "Infinite/NaN error bound at node '%s' (confidence=%.2f, "
                    "below 0.3 threshold — not pruning)",
                    nid,
                    conf,
                )

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
        if optuna is None:
            return {}

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
        trial: Any,
        benchmark: BenchmarkResult,
    ) -> None:
        """Report intermediate benchmark loss to the Optuna pruner."""
        if optuna is None:
            return
        trial.report(benchmark.global_loss, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


class _FallbackStudy:
    def __init__(self, *, study_name: str) -> None:
        self.study_name = study_name
        self.trials: list[Any] = []

    def ask(self) -> "_FallbackTrial":
        trial = _FallbackTrial(number=len(self.trials))
        self.trials.append(trial)
        return trial

    def tell(self, trial: "_FallbackTrial", value: float | None = None, state: Any | None = None) -> None:
        trial.value = value
        trial.state = state or "complete"


class _FallbackTrial:
    def __init__(self, *, number: int) -> None:
        self.number = number
        self.params: dict[str, Any] = {}
        self.value: float | None = None
        self.state: str | None = None

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        value = choices[0]
        self.params[name] = value
        return value

    def suggest_int(
        self,
        name: str,
        low: int,
        high: int,
        *,
        step: int = 1,
        log: bool = False,
    ) -> int:
        del high, step, log
        self.params[name] = low
        return low

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        del high, step, log
        self.params[name] = low
        return low
