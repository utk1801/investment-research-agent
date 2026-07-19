"""Encode text chunks to embeddings using sentence-transformers."""

from __future__ import annotations

import logging
from typing import Sequence

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDING_DIM

log = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        log.info("Model loaded. Embedding dim: %s", EMBEDDING_DIM)
    return _model


def embed_texts(texts: Sequence[str], normalize: bool = True) -> list[list[float]]:
    """Encode a list of strings to embedding vectors."""
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(list(texts), normalize_embeddings=normalize, show_progress_bar=False)
    return [v.tolist() for v in vectors]


class Embedder:
    def __init__(self):
        self.model = get_embedding_model()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return embed_texts(texts, normalize=True)

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM