"""Ghost Witness system - lightweight abstract interpretation for graph verification.

A Ghost Witness is an executable shadow of a heavy atom that propagates
metadata (shape, dtype, sampling_rate, domain) instead of computing actual
values.  Running witnesses through a graph catches shape mismatches, type
errors, and domain violations *before* any expensive computation runs.
"""

from .abstract import AbstractSignal, AbstractBeatPool
from .registry import REGISTRY, register_atom
from .simulator import simulate_graph, PlanError

__all__ = [
    "AbstractSignal",
    "AbstractBeatPool",
    "REGISTRY",
    "register_atom",
    "simulate_graph",
    "PlanError",
]
