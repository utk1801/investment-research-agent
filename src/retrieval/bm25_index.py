"""BM25 keyword search index over chunk texts."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import DATA_DIR

log = logging.getLogger(__name__)
BM25_PATH = DATA_DIR / "bm25_index.pkl"


class BM25Index:
    def __init__(self):
        self._index: BM25Okapi | None = None
        self._chunks: list[dict] = []
        self._tokenized_corpus: list[list[str]] = []

    def build(self, chunks: list[dict]) -> None:
        """Build BM25 from chunk list. Each chunk must have 'chunk_text' key."""
        self._chunks = chunks
        tokenized = [self._tokenize(c["chunk_text"]) for c in chunks]
        self._index = BM25Okapi(tokenized)
        self._tokenized_corpus = tokenized
        log.info("BM25 index built: %d documents", len(self._chunks))

    def search(self, query: str, top_k: int = 20) -> list[tuple[dict, float]]:
        """Search BM25. Returns list of (chunk, score) sorted desc."""
        if self._index is None:
            return []
        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._chunks[i], float(scores[i])) for i in top_indices]

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def save(self, path: Path | None = None) -> None:
        path = path or BM25_PATH
        with open(path, "wb") as f:
            pickle.dump({"index": self._index, "chunks": self._chunks,
                         "tokenized": self._tokenized_corpus}, f)
        log.info("BM25 index saved to %s", path)

    def load(self, path: Path | None = None) -> bool:
        path = path or BM25_PATH
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._index = data["index"]
            self._chunks = data["chunks"]
            self._tokenized_corpus = data["tokenized"]
            log.info("BM25 index loaded from %s (%d docs)", path, len(self._chunks))
            return True
        except Exception as exc:
            log.warning("Failed to load BM25 index: %s", exc)
            return False

    # ── singleton ──────────────────────────────────────────────────────────────
    _instance: "BM25Index | None" = None

    @classmethod
    def get_instance(cls) -> "BM25Index":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.load()
        return cls._instance