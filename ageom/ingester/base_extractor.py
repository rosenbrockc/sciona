"""Base extractor protocol and language dispatch types."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from ageom.ingester.models import RawDataFlowGraph


class SourceLanguage(str, Enum):
    """Supported source languages for extraction."""

    PYTHON = "python"
    CPP = "cpp"
    JULIA = "julia"
    RUST = "rust"


EXTENSION_MAP: dict[str, SourceLanguage] = {
    ".py": SourceLanguage.PYTHON,
    ".cpp": SourceLanguage.CPP,
    ".cc": SourceLanguage.CPP,
    ".cxx": SourceLanguage.CPP,
    ".h": SourceLanguage.CPP,
    ".hpp": SourceLanguage.CPP,
    ".jl": SourceLanguage.JULIA,
    ".rs": SourceLanguage.RUST,
}


@runtime_checkable
class BaseExtractor(Protocol):
    """Protocol for language-specific source code extractors."""

    async def extract_class(
        self, source_path: str, class_name: str
    ) -> RawDataFlowGraph:
        """Extract data-flow graph for a class/struct from a source file."""
        ...

    async def extract_procedural(
        self, source_path: str, pipeline_name: str | None = None
    ) -> RawDataFlowGraph:
        """Extract data-flow graph for procedural/top-level code."""
        ...
