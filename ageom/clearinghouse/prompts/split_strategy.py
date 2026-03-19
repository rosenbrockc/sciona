"""LLM prompt template for data stratification strategy."""

from __future__ import annotations

SPLIT_STRATEGY_PROMPT = """\
You are analyzing a dataset for stratified train/test splitting.

Dataset schema (ageom.yml):
{ageom_yml_content}

Dataset statistics:
{statistics_json}

Identify the best stratification axis for splitting this dataset into
a 20% public / 80% blind partition. The split must ensure:
1. No data leakage between partitions (e.g., same subject in both)
2. Representative distribution of key characteristics
3. Deterministic assignment from the axis values

Return JSON: {{"stratify_by": "<field>", "method": "hash", "reason": "..."}}
"""


def format_split_strategy_prompt(
    ageom_yml_content: str,
    statistics_json: str,
) -> str:
    """Format the split strategy prompt with dataset-specific content."""
    return SPLIT_STRATEGY_PROMPT.format(
        ageom_yml_content=ageom_yml_content,
        statistics_json=statistics_json,
    )
