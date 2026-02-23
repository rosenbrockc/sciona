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

    # Atom sources
    sources_file: Path = Field(default=Path("sources.yml"))

    # Embedding model
    embedding_model: str = "microsoft/unixcoder-base"
    embedding_dim: int = 768
    embedding_batch_size: int = 32

    # FAISS index
    index_dir: Path = Field(default=Path("data/index"))

    # LLM (global defaults / shared credentials)
    llm_provider: str = "anthropic"  # "anthropic" | "codex"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""
    llm_model: str = "claude-sonnet-4-5-20250929"
    llm_max_tokens: int = 4096
    llama_cpp_base_url: str = "http://127.0.0.1:8080/v1"
    llama_cpp_api_key: str = "local"

    # Lean 4
    lean_toolchain: str = "leanprover/lean4:v4.14.0"
    mathlib_path: Path = Field(default=Path(""))

    # Coq
    coq_project_path: Path = Field(default=Path(""))

    # Hunter agent parameters
    hunter_llm_provider: str = "llama_cpp"  # default local quantized worker
    hunter_llm_model: str = "llama-3.1-8b-instruct"
    hunter_llm_max_tokens: int = 1024
    hunter_mode: str = "speculative_local"  # "standard" | "speculative_local"
    hunter_use_gbnf: bool = True
    hunter_max_iterations: int = 5
    hunter_top_k_verify: int = 10
    hunter_search_k: int = 20
    hunter_query_batch_size: int = 40
    hunter_top_k_per_query: int = 50
    hunter_max_candidates_total: int = 3000
    hunter_verify_concurrency: int = 1  # >1 enables parallel verification

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

    # Ingester (Round 0)
    ingester_llm_provider: str = ""  # falls back to llm_provider
    ingester_llm_model: str = "claude-sonnet-4-5-20250929"

    # --- Per-prompt LLM overrides (empty = fall back to agent-level) ---

    # Architect per-prompt
    architect_strategy_llm_provider: str = ""
    architect_strategy_llm_model: str = ""
    architect_decompose_llm_provider: str = ""
    architect_decompose_llm_model: str = ""
    architect_critique_llm_provider: str = ""
    architect_critique_llm_model: str = ""

    # Hunter per-prompt
    hunter_score_llm_provider: str = ""
    hunter_score_llm_model: str = ""
    hunter_reformulate_llm_provider: str = ""
    hunter_reformulate_llm_model: str = ""
    hunter_analyze_failure_llm_provider: str = ""
    hunter_analyze_failure_llm_model: str = ""

    # Synthesizer per-prompt
    synthesizer_repair_llm_provider: str = ""
    synthesizer_repair_llm_model: str = ""
    synthesizer_tactic_llm_provider: str = ""
    synthesizer_tactic_llm_model: str = ""

    # Ingester per-prompt
    ingester_chunk_llm_provider: str = ""
    ingester_chunk_llm_model: str = ""
    ingester_hoist_state_llm_provider: str = ""
    ingester_hoist_state_llm_model: str = ""
    ingester_abstract_llm_provider: str = ""
    ingester_abstract_llm_model: str = ""
    ingester_fix_type_llm_provider: str = ""
    ingester_fix_type_llm_model: str = ""
    ingester_fix_ghost_llm_provider: str = ""
    ingester_fix_ghost_llm_model: str = ""
    ingester_opaque_witness_llm_provider: str = ""
    ingester_opaque_witness_llm_model: str = ""
    ingester_fix_message_cycle_llm_provider: str = ""
    ingester_fix_message_cycle_llm_model: str = ""

    # Orchestrator per-prompt
    orchestrator_refine_llm_provider: str = ""
    orchestrator_refine_llm_model: str = ""

    # Python target
    python_path: str = "python"
    python_mypy_path: str = "mypy"
    python_packages: str = "numpy,scipy"

    # Extractor (Round 3 Phase 3)
    export_output_dir: Path = Field(default=Path("export"))
    lean_lake_path: str = "lake"  # path to lake binary
    optimize_by_default: bool = False
