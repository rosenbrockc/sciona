"""Configuration for AGEO-Matcher via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgeomConfig(BaseSettings):
    """Central configuration loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_prefix="AGEOM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Embedding model
    embedding_model: str = "microsoft/unixcoder-base"
    embedding_dim: int = 768
    embedding_batch_size: int = 32

    # FAISS index
    index_dir: Path = Field(default=Path("data/index"))

    # LLM (for Hunter agent)
    llm_provider: str = "anthropic"  # "anthropic" | "codex"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""
    llm_model: str = "claude-sonnet-4-5-20250929"
    llm_max_tokens: int = 4096

    # Lean 4
    lean_toolchain: str = "leanprover/lean4:v4.14.0"
    mathlib_path: Path = Field(default=Path(""))

    # Coq
    coq_project_path: Path = Field(default=Path(""))

    # Hunter agent parameters
    hunter_max_iterations: int = 5
    hunter_top_k_verify: int = 3
    hunter_search_k: int = 20

    # Architect (Round 1)
    skill_index_dir: Path = Field(default=Path("data/skill_index"))
    clrs_path: Path = Field(default=Path(""))
    coq100_path: Path = Field(default=Path(""))
    postgres_uri: str = ""
    architect_max_depth: int = 8
    architect_llm_provider: str = ""  # falls back to llm_provider when empty
    architect_llm_model: str = "claude-sonnet-4-5-20250929"

    # Synthesizer (Round 3)
    synthesizer_max_iterations: int = 10
    synthesizer_llm_provider: str = ""  # falls back to llm_provider
    synthesizer_llm_model: str = "claude-sonnet-4-5-20250929"

    # Python target
    python_path: str = "python"
    python_mypy_path: str = "mypy"
    python_packages: str = "numpy,scipy"

    # Extractor (Round 3 Phase 3)
    export_output_dir: Path = Field(default=Path("export"))
    lean_lake_path: str = "lake"  # path to lake binary
    optimize_by_default: bool = False
