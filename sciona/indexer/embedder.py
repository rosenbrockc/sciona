"""Embedding backends for semantic retrieval."""

from __future__ import annotations

from typing import Protocol
import warnings

import numpy as np

from sciona.julia_runtime import configure_juliacall_env
from sciona.types import Declaration

DEFAULT_EMBEDDING_BACKEND = "fastembed"
DEFAULT_EMBEDDING_MODELS = {
    "fastembed": "BAAI/bge-small-en-v1.5",
    "unixcoder": "microsoft/unixcoder-base",
}


class Embedder(Protocol):
    """Common protocol for local embedding backends."""

    backend: str
    model_name: str

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...

    def embed_declaration(self, decl: Declaration) -> np.ndarray: ...


def normalize_embedding_backend(backend: str | None) -> str:
    """Normalize embedding backend names."""
    value = str(backend or DEFAULT_EMBEDDING_BACKEND).strip().lower()
    if value in {"fastembed", "fast"}:
        return "fastembed"
    if value in {"unixcoder", "transformers"}:
        return "unixcoder"
    raise ValueError(f"Unsupported embedding backend: {backend!r}")


def default_embedding_model(backend: str | None) -> str:
    """Return the default model for *backend*."""
    normalized = normalize_embedding_backend(backend)
    return DEFAULT_EMBEDDING_MODELS[normalized]


def create_embedder(
    backend: str | None = None,
    model_name: str | None = None,
) -> Embedder:
    """Create an embedder for the requested backend."""
    normalized = normalize_embedding_backend(backend)
    selected_model = str(model_name or "").strip() or default_embedding_model(normalized)
    if normalized == "fastembed":
        return FastEmbedEmbedder(selected_model)
    return UniXcoderEmbedder(selected_model)


def _declaration_text(decl: Declaration) -> str:
    text = f"{decl.name} : {decl.type_signature}"
    if decl.docstring:
        text += f"\n{decl.docstring}"
    if decl.conceptual_summary:
        text += f"\n{decl.conceptual_summary}"
    return text


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = arr / norm
    return arr


def _prefer_juliacall_before_torch() -> None:
    """Best-effort import ordering guard for environments that use juliacall.

    Some environments warn or segfault if torch is imported before juliacall.
    UniXcoder loads through transformers/torch, so opportunistically import
    juliacall first when available. Failures here should not block embedding.
    """
    try:
        configure_juliacall_env()
        import juliacall  # noqa: F401
    except Exception:
        return


class FastEmbedEmbedder:
    """Embeds text locally using the FastEmbed ONNX runtime stack."""

    backend = "fastembed"

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODELS["fastembed"]) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed("__sciona_dim_probe__").shape[0])
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        batch = self.embed_batch([text], batch_size=1)
        return batch[0]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            dim = self._dim or 0
            return np.zeros((0, dim), dtype=np.float32)

        raw_vectors = list(self._model.embed(texts, batch_size=batch_size))
        vectors = [_l2_normalize(np.asarray(vec, dtype=np.float32)) for vec in raw_vectors]
        if vectors and self._dim is None:
            self._dim = int(vectors[0].shape[0])
        return np.vstack(vectors)

    def embed_declaration(self, decl: Declaration) -> np.ndarray:
        return self.embed(_declaration_text(decl))


class UniXcoderEmbedder:
    """Embeds text using microsoft/unixcoder-base through transformers/torch."""

    backend = "unixcoder"

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODELS["unixcoder"]) -> None:
        # Lazy imports so torch/transformers aren't required at import time.
        _prefer_juliacall_before_torch()
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name

        # UniXcoder currently triggers a tokenizers deprecation inside the
        # upstream Roberta loader path. There is no replacement API at this
        # layer yet, so suppress the narrow known warning locally.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Deprecated in 0\.9\.0: BPE\.__init__ will not create from files anymore, try `BPE\.from_file` instead",
                category=DeprecationWarning,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._dim = 768

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string into an L2-normalized vector."""
        import torch

        tokens = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            outputs = self._model(**tokens)
        return _l2_normalize(outputs.last_hidden_state[:, 0, :].squeeze(0).numpy())

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of texts. Returns array of shape (N, dim)."""
        import torch

        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        all_vecs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            with torch.no_grad():
                outputs = self._model(**tokens)
            vecs = outputs.last_hidden_state[:, 0, :].numpy()
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            all_vecs.append((vecs / norms).astype(np.float32))
        return np.vstack(all_vecs)

    def embed_declaration(self, decl: Declaration) -> np.ndarray:
        return self.embed(_declaration_text(decl))
