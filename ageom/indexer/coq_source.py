"""Declaration source for Coq/Rocq via coqpyt."""

from __future__ import annotations

import logging
from pathlib import Path

from ageom.types import Declaration, Prover

logger = logging.getLogger(__name__)


class CoqDeclarationSource:
    """Extracts declarations from Coq .v files using coqpyt."""

    def get_declarations_from_file(self, filepath: str | Path) -> list[Declaration]:
        """Extract all declarations from a single .v file."""
        from coqpyt.coq_file import CoqFile

        filepath = Path(filepath)
        declarations: list[Declaration] = []

        coq_file = CoqFile(str(filepath))
        try:
            for step in coq_file.steps:
                text = step.text.strip()
                # Extract declarations (Theorem, Lemma, Definition, etc.)
                for keyword in (
                    "Theorem",
                    "Lemma",
                    "Definition",
                    "Fixpoint",
                    "Axiom",
                    "Corollary",
                ):
                    if text.startswith(keyword):
                        name = _extract_name(text, keyword)
                        type_sig = _extract_type(text)
                        declarations.append(
                            Declaration(
                                name=name,
                                type_signature=type_sig,
                                source_lib=str(filepath),
                                prover=Prover.COQ,
                                raw_code=text,
                            )
                        )
                        break
        finally:
            coq_file.close()

        return declarations

    def get_all_declarations(self, directory: str | Path) -> list[Declaration]:
        """Recursively extract declarations from all .v files in a directory."""
        directory = Path(directory)
        declarations: list[Declaration] = []
        for v_file in sorted(directory.rglob("*.v")):
            try:
                declarations.extend(self.get_declarations_from_file(v_file))
            except Exception:
                logger.warning("Failed to parse %s, skipping", v_file, exc_info=True)
                continue
        return declarations


def _extract_name(text: str, keyword: str) -> str:
    """Extract the declaration name from a Coq statement."""
    rest = text[len(keyword) :].strip()
    # Name is the first token after the keyword
    name = rest.split()[0] if rest.split() else ""
    # Remove trailing colon or period
    return name.rstrip(":.")


def _extract_type(text: str) -> str:
    """Extract the type signature from a Coq declaration."""
    # Type is between the first ':' and ':=' or '.'
    if ":" not in text:
        return ""
    after_colon = text.split(":", 1)[1]
    for terminator in (":=", "."):
        if terminator in after_colon:
            return after_colon.split(terminator, 1)[0].strip()
    return after_colon.strip()
