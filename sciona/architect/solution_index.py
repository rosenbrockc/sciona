"""Searchable index of solution CDG templates for prompt-to-template matching."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SolutionTemplate:
    """A solution CDG template indexed for retrieval."""

    name: str
    family: str = ""
    paradigm: str = ""
    summary: str = ""
    dejargonized_summary: str = ""
    use_when: list[str] = field(default_factory=list)
    do_not_use_when: list[str] = field(default_factory=list)
    key_insight: str = ""
    critical_stages: list[str] = field(default_factory=list)
    swappable_stages: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    scaling_notes: str = ""
    stage_names: list[str] = field(default_factory=list)
    stage_descriptions: str = ""
    grounding_rate: float = 0.0
    cdg_path: Path = field(default_factory=Path)
    raw_cdg: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tokenizer (shared with catalog._tokenize style)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "the", "of", "in", "for", "and", "or", "to", "is", "by", "on", "at",
    "an", "it", "as", "be", "if", "so", "no", "do", "up", "we", "he",
    "compute", "apply", "return", "returns", "input", "output",
    "data", "array", "value", "values", "result", "results",
    "function", "method", "using", "from", "with", "into",
    "each", "that", "this", "given", "based", "use", "used",
    "set", "get", "new", "per", "one", "two", "all", "can",
    "when", "then", "than", "also", "same", "only", "not", "are",
    "has", "have", "will", "been", "were", "was", "its", "may",
})


def _tokenize(text: str) -> frozenset[str]:
    tokens = set(re.split(r"[\s_\-/,;:()]+", text.lower())) - _STOP_WORDS
    tokens.discard("")
    return frozenset(t for t in tokens if len(t) >= 2)


# ---------------------------------------------------------------------------
# SolutionTemplateIndex
# ---------------------------------------------------------------------------


class SolutionTemplateIndex:
    """Searchable index of solution CDG templates."""

    def __init__(self, templates: list[SolutionTemplate]) -> None:
        self._templates = {t.name: t for t in templates}
        self._idf: dict[str, float] = {}
        self._template_tokens: dict[str, frozenset[str]] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Build IDF table from template searchable text."""
        n = len(self._templates)
        if n == 0:
            return

        df: dict[str, int] = {}
        for template in self._templates.values():
            tokens = self._searchable_tokens(template)
            self._template_tokens[template.name] = tokens
            for token in tokens:
                df[token] = df.get(token, 0) + 1

        self._idf = {
            token: math.log(n / count)
            for token, count in df.items()
        }

    def _searchable_tokens(self, template: SolutionTemplate) -> frozenset[str]:
        """Build searchable token set from template metadata."""
        parts = [
            template.dejargonized_summary or template.summary,
            " ".join(template.use_when),
            template.family,
            template.paradigm,
            template.key_insight,
            template.stage_descriptions,
            " ".join(template.stage_names),
        ]
        return _tokenize(" ".join(parts))

    @classmethod
    def from_directory(cls, cdg_dir: str | Path) -> SolutionTemplateIndex:
        """Load all solution CDGs + bindings from a directory."""
        cdg_dir = Path(cdg_dir)
        templates: list[SolutionTemplate] = []

        for cdg_path in sorted(cdg_dir.glob("*.json")):
            if "_bindings" in cdg_path.name:
                continue
            try:
                data = json.loads(cdg_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            applicability = data.get("applicability", {})
            stages = data.get("stages", [])

            # Compute grounding rate from bindings
            bindings_path = cdg_path.with_name(cdg_path.stem + "_bindings.json")
            grounding_rate = 0.0
            if bindings_path.exists():
                try:
                    bd = json.loads(bindings_path.read_text())
                    total = len(bd.get("bindings", []))
                    resolved = sum(
                        1 for b in bd.get("bindings", [])
                        if b.get("status") in ("active", "approximate")
                        or b.get("action_class") in (
                            "orchestration", "trivial_inline",
                            "external_knowledge", "external_tool",
                        )
                    )
                    grounding_rate = resolved / total if total else 0.0
                except (json.JSONDecodeError, OSError):
                    pass

            stage_descs = " ".join(
                s.get("description", "") + " " + s.get("name", "")
                for s in stages
            )

            templates.append(SolutionTemplate(
                name=cdg_path.stem,
                family=data.get("family", ""),
                paradigm=data.get("paradigm", ""),
                summary=data.get("summary", ""),
                dejargonized_summary=data.get("dejargonized_summary", ""),
                use_when=applicability.get("use_when", []),
                do_not_use_when=applicability.get("do_not_use_when", []),
                key_insight=applicability.get("key_insight", ""),
                critical_stages=applicability.get("critical_stages", []),
                swappable_stages=applicability.get("swappable_stages", []),
                failure_modes=applicability.get("failure_modes", []),
                scaling_notes=applicability.get("scaling_notes", ""),
                stage_names=[s.get("stage_id", "") for s in stages],
                stage_descriptions=stage_descs,
                grounding_rate=grounding_rate,
                cdg_path=cdg_path,
                raw_cdg=data,
            ))

        return cls(templates)

    @property
    def size(self) -> int:
        return len(self._templates)

    def get(self, name: str) -> SolutionTemplate | None:
        return self._templates.get(name)

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        family_filter: str | None = None,
    ) -> list[tuple[SolutionTemplate, float]]:
        """Keyword search against dejargonized summaries + use_when.

        Returns top-k (template, score) pairs sorted by score descending.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[SolutionTemplate, float]] = []
        for template in self._templates.values():
            template_tokens = self._template_tokens.get(template.name, frozenset())
            overlap = query_tokens & template_tokens
            if not overlap:
                continue

            # TF-IDF weighted overlap
            score = sum(self._idf.get(t, 1.0) for t in overlap)

            # Family bonus
            if family_filter and template.family == family_filter:
                score += 5.0

            # Grounding bonus — prefer fully grounded templates
            score += template.grounding_rate * 2.0

            scored.append((template, score))

        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def search_dejargonized(
        self,
        dejargonized_prompt: str,
        original_prompt: str = "",
        k: int = 10,
    ) -> list[tuple[SolutionTemplate, float]]:
        """Search using both dejargonized and original prompt text.

        The dejargonized prompt provides canonical vocabulary overlap,
        while the original retains domain-specific terms that may appear
        in template stage descriptions.
        """
        combined = dejargonized_prompt
        if original_prompt:
            combined = dejargonized_prompt + " " + original_prompt
        return self.search(combined, k=k)
