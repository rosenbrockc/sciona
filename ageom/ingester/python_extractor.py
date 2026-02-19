"""Python AST extractor — thin adapter wrapping existing extraction functions."""

from __future__ import annotations

from ageom.ingester.extractor import extract_data_flow, extract_procedural_data_flow
from ageom.ingester.models import RawDataFlowGraph


class PythonASTExtractor:
    """Adapter delegating to the existing Python AST extraction functions."""

    async def extract_class(
        self, source_path: str, class_name: str
    ) -> RawDataFlowGraph:
        return await extract_data_flow(source_path, class_name)

    async def extract_procedural(
        self, source_path: str, pipeline_name: str | None = None
    ) -> RawDataFlowGraph:
        return await extract_procedural_data_flow(source_path, pipeline_name)
