"""Base extractor protocol and language dispatch types."""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from sciona.ingester.models import RawDataFlowGraph


class SourceLanguage(str, Enum):
    """Supported source languages for extraction."""

    PYTHON = "python"
    CPP = "cpp"
    JULIA = "julia"
    RUST = "rust"
    HASKELL = "haskell"


EXTENSION_MAP: dict[str, SourceLanguage] = {
    ".py": SourceLanguage.PYTHON,
    ".cpp": SourceLanguage.CPP,
    ".cc": SourceLanguage.CPP,
    ".cxx": SourceLanguage.CPP,
    ".h": SourceLanguage.CPP,
    ".hpp": SourceLanguage.CPP,
    ".ipp": SourceLanguage.CPP,
    ".jl": SourceLanguage.JULIA,
    ".rs": SourceLanguage.RUST,
    ".hs": SourceLanguage.HASKELL,
}


@runtime_checkable
class BaseExtractor(Protocol):
    """Protocol for language-specific source code extractors."""

    async def extract_class(
        self, source_path: str, class_name: str
    ) -> RawDataFlowGraph:
        """Extract data-flow graph for a class/struct from a source file."""
        ...

    async def extract_function(
        self, source_path: str, function_name: str
    ) -> RawDataFlowGraph:
        """Extract data-flow graph for a named function and its call tree."""
        ...

    async def extract_procedural(
        self, source_path: str, pipeline_name: str | None = None
    ) -> RawDataFlowGraph:
        """Extract data-flow graph for procedural/top-level code."""
        ...
