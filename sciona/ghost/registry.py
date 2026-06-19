"""Ghost Witness registry - binds heavy atom implementations to their witnesses.

The registry is the central lookup table that the simulator uses to find
the witness for each atom in a computation graph.  Heavy functions are
registered via the ``@register_atom`` decorator; the simulator never
calls the heavy function, only its witness.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, Dict

import numpy as np

if TYPE_CHECKING:
    from sciona.ghost.dimensions import DimensionalSignature


# Global registry: function_name -> { impl, witness, doc, signature, ... }
REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_atom(
    witness: Callable,
    *,
    name: str | None = None,
    dim_map: dict[str, "DimensionalSignature"] | None = None,
) -> Callable:
    """Decorator that binds a heavy implementation to a Ghost Witness.

    The heavy function is stored in the registry alongside its witness.
    The agent's simulator reads the *witness* signature to understand
    what abstract types flow in and out.

    Args:
        witness: A callable that accepts and returns abstract value types
            (e.g. ``AbstractSignal``).  Must have type annotations.
        name: Optional explicit registry key. If omitted, the heavy function
            name is used.
        dim_map: Optional mapping of parameter/output names to their
            ``DimensionalSignature``.  When provided, the compiler can
            enforce dimensional consistency on CDG edges.

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
        fqdn = f"{heavy_func.__module__}.{heavy_func.__name__}"
        atom_name = name or fqdn
        REGISTRY[atom_name] = {
            "impl": heavy_func,
            "witness": witness,
            "doc": heavy_func.__doc__ or "",
            "signature": dict(witness.__annotations__),
            "heavy_signature": dict(heavy_func.__annotations__),
            "module": getattr(heavy_func, "__module__", ""),
            "name": atom_name,
            "dim_signature": dim_map or {},
            "symbolic": None,
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
    if name in REGISTRY:
        return REGISTRY[name]["witness"]

    # Suffix fallback for short name lookups
    short_name = name.rsplit(".", 1)[-1]
    matches = []
    for key in REGISTRY:
        if key == short_name or key.endswith("." + short_name):
            matches.append((key, REGISTRY[key]["witness"]))

    if len(matches) == 1:
        return matches[0][1]
    elif len(matches) > 1:
        raise KeyError(
            f"Ambiguous witness lookup for '{name}'; matches multiple registered keys: "
            f"{[item[0] for item in matches]}"
        )

    raise KeyError(
        f"No ghost witness registered for '{name}'. "
        f"Available: {sorted(REGISTRY.keys())}"
    )


def list_registered() -> list[str]:
    """Return sorted list of all registered atom names."""
    return sorted(REGISTRY.keys())
