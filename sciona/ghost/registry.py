"""Ghost Witness registry - binds heavy atom implementations to their witnesses.

The registry is the central lookup table that the simulator uses to find
the witness for each atom in a computation graph.  Heavy functions are
registered via the ``@register_atom`` decorator; the simulator never
calls the heavy function, only its witness.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict

import numpy as np


# Global registry: function_name -> { impl, witness, doc, signature }
REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_atom(witness: Callable, *, name: str | None = None) -> Callable:
    """Decorator that binds a heavy implementation to a Ghost Witness.

    The heavy function is stored in the registry alongside its witness.
    The agent's simulator reads the *witness* signature to understand
    what abstract types flow in and out.

    Args:
        witness: A callable that accepts and returns abstract value types
            (e.g. ``AbstractSignal``).  Must have type annotations.
        name: Optional explicit registry key. If omitted, the heavy function
            name is used.

    Returns:
        A decorator that registers the heavy function and returns it
        unchanged.

    Example::

        def witness_fft(sig: AbstractSignal) -> AbstractSignal:
            sig.assert_domain("time")
            return AbstractSignal(shape=sig.shape, dtype="complex128", ...)

        @register_atom(witness=witness_fft)
        def fft(a: np.ndarray, ...) -> np.ndarray:
            return np.fft.fft(a, ...)
    """
    def decorator(heavy_func: Callable) -> Callable:
        atom_name = name or heavy_func.__name__
        REGISTRY[atom_name] = {
            "impl": heavy_func,
            "witness": witness,
            "doc": heavy_func.__doc__ or "",
            "signature": dict(witness.__annotations__),
            "heavy_signature": dict(heavy_func.__annotations__),
            "module": getattr(heavy_func, "__module__", ""),
            "name": atom_name,
        }
        return heavy_func
    return decorator


def get_witness(name: str) -> Callable:
    """Look up the ghost witness for a registered atom.

    Args:
        name: The function name (as registered via ``@register_atom``).

    Returns:
        The witness callable.

    Raises:
        KeyError: If no atom with that name is registered.
    """
    if name not in REGISTRY:
        raise KeyError(
            f"No ghost witness registered for '{name}'. "
            f"Available: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[name]["witness"]


def list_registered() -> list[str]:
    """Return sorted list of all registered atom names."""
    return sorted(REGISTRY.keys())
