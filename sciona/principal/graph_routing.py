"""Pure routing helpers for the Principal graph."""

from __future__ import annotations

from sciona.principal.graph_types import PrincipalState


def route_after_gradients(state: PrincipalState) -> str:
    """Decide whether to continue optimising or stop."""
    if state.done:
        return "end"
    if len(state.trial_history) >= state.max_trials and state.current_trial > 1:
        return "end"
    if state.error and "pruned" not in state.error.lower():
        return "end"
    if state.pending_param_search and state.param_trials_remaining > 0:
        return "suggest_params"
    return "select_proposal"


def route_after_proposal(state: PrincipalState) -> str:
    """After proposal comparison, either evaluate the chosen branch or fall back."""
    if state.done:
        return "end"
    if len(state.trial_history) >= state.max_trials:
        return "end"
    if state.selected_proposal:
        return "suggest_params"
    return "time_travel"


def route_after_admissibility(state: PrincipalState) -> str:
    """After admissibility, either refine immediately or continue to gradients."""
    if state.done:
        return "end"
    if state.admissibility_hard_rejected or state.admissibility_requires_refinement:
        return "select_proposal"
    return "gradients"


def route_after_update(state: PrincipalState) -> str:
    """After time-travel update, loop back or stop."""
    if state.done:
        return "end"
    if len(state.trial_history) >= state.max_trials:
        return "end"
    return "suggest_params"


def route_after_forward(state: PrincipalState) -> str:
    """After forward pass, evaluate or loop back (if pruned early)."""
    if state.done:
        return "end"
    if state.error:
        return "time_travel"
    return "evaluate"
