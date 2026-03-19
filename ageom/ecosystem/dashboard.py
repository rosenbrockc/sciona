"""Dashboard helpers — impact factor, BibTeX, compute-preserved metrics."""

from __future__ import annotations

from typing import Sequence

from ageom.ecosystem.models import ComputePreserved, OriginatorImpact


def compute_h_index(bounty_values: Sequence[float]) -> int:
    """Compute h-index analog for algorithmic impact.

    The h-index is the largest k such that k bounties are each worth >= k.
    """
    sorted_values = sorted(bounty_values, reverse=True)
    h = 0
    for i, value in enumerate(sorted_values):
        if value >= i + 1:
            h = i + 1
        else:
            break
    return h


def compute_impact_factor(
    bounty_values: Sequence[float],
    atom_count: int = 0,
    *,
    originator_id: str = "",
    github_username: str = "",
    affiliation: str = "",
) -> OriginatorImpact:
    """Compute the Algorithmic Impact Factor for an originator."""
    return OriginatorImpact(
        originator_id=originator_id,
        github_username=github_username,
        affiliation=affiliation,
        bounty_count=len(bounty_values),
        total_bounty_value=sum(bounty_values),
        atom_count=atom_count,
        h_index=compute_h_index(bounty_values),
    )


def estimate_compute_preserved(
    bounties: Sequence[dict],
    tokens_per_step: int = 2000,
    avg_attempts: int = 5,
    cost_per_million_tokens: float = 3.0,
) -> ComputePreserved:
    """Estimate aggregate compute preserved by settled bounties.

    Parameters
    ----------
    bounties
        Sequence of dicts with keys: ``escrow_amount``, ``cdg_node_count``.
    tokens_per_step
        Average tokens per agentic step.
    avg_attempts
        Average number of agentic attempts per CDG node.
    cost_per_million_tokens
        Estimated cost per million tokens.
    """
    total_tokens = 0
    total_escrow = 0.0

    for b in bounties:
        node_count = b.get("cdg_node_count", 0)
        tokens = node_count * tokens_per_step * avg_attempts
        total_tokens += tokens
        total_escrow += b.get("escrow_amount", 0.0)

    cost_saved = total_tokens * cost_per_million_tokens / 1_000_000

    return ComputePreserved(
        total_bounties_settled=len(bounties),
        total_escrow_value=total_escrow,
        estimated_tokens_saved=total_tokens,
        estimated_cost_saved_usd=cost_saved,
    )


def generate_bibtex(
    atom_fqdn: str,
    authors: Sequence[str],
    year: int = 2025,
    description: str = "",
) -> str:
    """Generate a BibTeX entry for an atom.

    Parameters
    ----------
    atom_fqdn
        Fully qualified domain name of the atom.
    authors
        List of author names (e.g., GitHub usernames).
    year
        Publication year.
    description
        Short description of the atom.
    """
    # Sanitize the key (replace dots with underscores)
    bib_key = atom_fqdn.replace(".", "_")
    author_str = " and ".join(authors) if authors else "Unknown"
    title = description or atom_fqdn

    return (
        f"@misc{{{bib_key},\n"
        f"  author = {{{author_str}}},\n"
        f"  title = {{{title}}},\n"
        f"  year = {{{year}}},\n"
        f"  howpublished = {{Algorithmic Commons Registry}},\n"
        f"  note = {{fqdn:{atom_fqdn}}}\n"
        f"}}"
    )
