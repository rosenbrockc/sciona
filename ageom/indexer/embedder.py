"""UniXcoder-based embedder for formal code declarations."""

from __future__ import annotations

import numpy as np

from ageom.types import Declaration


class UniXcoderEmbedder:
    """Embeds formal code using microsoft/unixcoder-base.

    Produces L2-normalized 768-dimensional vectors suitable for
    cosine similarity search via inner product on normalized vectors.
    """

    def __init__(self, model_name: str = "microsoft/unixcoder-base") -> None:
        # Lazy imports so torch/transformers aren't required at import time
        from transformers import AutoModel, AutoTokenizer

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
        # Use CLS token embedding
        vec = outputs.last_hidden_state[:, 0, :].squeeze(0).numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed a batch of texts. Returns array of shape (N, 768)."""
        import torch

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
            # L2-normalize each vector
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            vecs = vecs / norms
            all_vecs.append(vecs)
        return np.vstack(all_vecs)

    def embed_declaration(self, decl: Declaration) -> np.ndarray:
        """Embed a declaration using its name, type signature, and docstring."""
        text = f"{decl.name} : {decl.type_signature}"
        if decl.docstring:
            text += f"\n{decl.docstring}"
        return self.embed(text)
