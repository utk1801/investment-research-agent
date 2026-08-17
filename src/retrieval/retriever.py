"""Hybrid retriever — BM25 + ChromaDB vector, RRF fusion, cross-encoder rerank."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import chromadb
import numpy as np
from sentence_transformers import CrossEncoder

from src.config import (
    CHROMA_DIR, COLLECTION_NAME, EMBEDDING_DIM,
    RETRIEVAL_TOP_K_HYBRID, RETRIEVAL_TOP_K_FINAL, RRF_K,
    EMBEDDING_MODEL, RERANKER_MODEL, DEFAULT_RETRIEVAL_APPROACH,
    ENABLE_QUERY_REWRITING, ALL_TICKERS,
)
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embed import Embedder
from src.query_rewriting import rewrite_query

log = logging.getLogger(__name__)

# ── ticker id map ──────────────────────────────────────────────────────────────
_TICKER_ID_MAP: dict[str, int] = {}
_XC_MODEL: CrossEncoder | None = None


def _init_ticker_map() -> None:
    global _TICKER_ID_MAP
    try:
        from sqlalchemy import create_engine, text
        from src.config import get_db_url
        engine = create_engine(get_db_url())
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT id, symbol FROM tickers")).fetchall()
        _TICKER_ID_MAP = {sym: tid for tid, sym in rows}
        log.info("Ticker map loaded: %d tickers", len(_TICKER_ID_MAP))
    except Exception as exc:
        log.warning("Could not load ticker map: %s", exc)


class HybridRetriever:
    def __init__(self, ticker_filter: str | None = None):
        self.ticker_filter = ticker_filter
        self.embedder = Embedder()
        self._chroma = None
        self._bm25: BM25Index | None = None
        self._xc_model: CrossEncoder | None = None
        self.last_rewritten_query: str | None = None
        self.last_retrieval_metadata: dict = {}
        _init_ticker_map()
        self._load()

    # Query-type classifier: determines optimal retrieval strategy
    @staticmethod
    def _classify_query(query: str) -> dict:
        """Classify query type to guide retrieval weighting."""
        q = query.lower()

        is_metric = any(kw in q for kw in [
            "free cash flow", "revenue", "profit margin", "pe ratio", "dividend",
            "p/e", "market cap", "net income", "ebitda", "earnings per share",
            "eps", "debt to equity", "current ratio", "quick ratio",
        ])
        is_comparison = any(kw in q for kw in [
            "vs ", " versus ", "compare", "comparison", "higher than", "lower than",
            "best ", "worst ", "top ", "highest", "lowest", "largest", "smallest",
            "which", "than", " between ",
        ])
        has_ticker = bool(HybridRetriever._extract_tickers(q))

        # Metric + any specificity anchor = keyword-heavy
        if is_metric:
            return {"strategy": "keyword_heavy", "is_metric": True,
                    "is_comparison": is_comparison, "has_ticker": has_ticker}
        if is_comparison:
            return {"strategy": "balanced", "is_metric": False,
                    "is_comparison": True, "has_ticker": has_ticker}
        return {"strategy": "semantic", "is_metric": False,
                "is_comparison": False, "has_ticker": has_ticker}

    @staticmethod
    def _extract_tickers(query: str) -> list[str]:
        """Extract known ticker symbols from query."""
        known = list(_TICKER_ID_MAP.keys()) or ALL_TICKERS
        found = []
        for t in known:
            if t.lower() in query.lower():
                found.append(t)
        return found

    def _normalize_by_type(self, results: list[dict], score_key: str) -> None:
        """Normalize scores to [0, 1] using per-type max to preserve score spread."""
        by_type: dict[str, list[dict]] = {}
        for r in results:
            by_type.setdefault(r.get("doc_type", ""), []).append(r)
        for entries in by_type.values():
            max_val = max(abs(r.get(score_key, 0.0)) for r in entries) or 1.0
            for r in entries:
                raw = r.get(score_key, 0.0)
                r[score_key] = max(0.0, raw) / max_val

    def _normalize_global(self, results: list[dict], score_key: str) -> None:
        """Normalize scores to [0, 1] using global max across all types."""
        max_val = max(abs(r.get(score_key, 0.0)) for r in results) or 1.0
        for r in results:
            raw = r.get(score_key, 0.0)
            r[score_key] = max(0.0, raw) / max_val

    def _load(self):
        self._chroma = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        self._bm25 = BM25Index()
        self._bm25.load()

    # ── RRF fusion ─────────────────────────────────────────────────────────────
    @staticmethod
    def _rrf(rankings: list[list[int]], scores: list[list[float]],
             k: int = RRF_K) -> list[tuple[int, float]]:
        d: dict[int, dict] = {}
        for ranking, s_list in zip(rankings, scores):
            for rank, idx in enumerate(ranking):
                d.setdefault(idx, {"rrf": 0.0})["rrf"] += 1.0 / (k + rank + 1)
        return sorted(d, key=lambda x: d[x]["rrf"], reverse=True)

    # ── Retrieve ────────────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K_FINAL,
        approach: str = DEFAULT_RETRIEVAL_APPROACH,
        rewrite: bool = ENABLE_QUERY_REWRITING,
    ) -> list[dict]:
        t0 = time.time()
        valid_approaches = {"bm25", "vector", "hybrid", "hybrid_rerank"}
        if approach not in valid_approaches:
            raise ValueError(f"Unknown retrieval approach '{approach}'. Expected one of {sorted(valid_approaches)}")

        rewrite_info = rewrite_query(query) if rewrite else None
        search_query = rewrite_info.rewritten_query if rewrite_info else query
        self.last_rewritten_query = search_query
        self.last_retrieval_metadata = {
            "approach": approach,
            "rewrite_enabled": rewrite,
            "rewrite_additions": rewrite_info.additions if rewrite_info else [],
        }

        query_emb = self.embedder.embed([search_query])[0]

        chroma_results = self._chroma_search(query_emb, n=RETRIEVAL_TOP_K_HYBRID)
        bm25_results = self._bm25_search(search_query, n=RETRIEVAL_TOP_K_HYBRID)

        if not chroma_results and not bm25_results:
            log.warning("No results from ChromaDB or BM25")
            return []

        if approach == "vector":
            self._normalize_global(chroma_results, "_score")
            self._normalize_global(chroma_results, "_chroma_score")
            candidates = chroma_results
            if self.ticker_filter:
                candidates = self._apply_ticker_filter(candidates)
            return candidates[:top_k]

        if approach == "bm25":
            self._normalize_global(bm25_results, "_score")
            self._normalize_global(bm25_results, "_bm25_score")
            candidates = bm25_results
            if self.ticker_filter:
                candidates = self._apply_ticker_filter(candidates)
            return candidates[:top_k]

        # Normalize based on query type
        info = self._classify_query(search_query)
        if info["strategy"] == "keyword_heavy":
            # BM25 naturally favors financial docs for metric queries; use global norm
            self._normalize_global(chroma_results, "_score")
            self._normalize_global(chroma_results, "_chroma_score")
            self._normalize_global(bm25_results, "_score")
            self._normalize_global(bm25_results, "_bm25_score")
        else:
            # Global normalization across all doc types so best match wins
            self._normalize_global(chroma_results, "_score")
            self._normalize_global(chroma_results, "_chroma_score")
            self._normalize_global(bm25_results, "_score")
            self._normalize_global(bm25_results, "_bm25_score")

        # Merge: dedup by doc_id
        candidates: list[dict] = []
        seen_ids: set[str] = set()

        for c in chroma_results:
            doc_id = c.get("doc_id")
            if doc_id is None or doc_id not in seen_ids:
                candidates.append(c)
                if doc_id is not None:
                    seen_ids.add(doc_id)
        for b in bm25_results:
            doc_id = b.get("doc_id")
            if doc_id is None or doc_id not in seen_ids:
                candidates.append(b)
                if doc_id is not None:
                    seen_ids.add(doc_id)
            else:
                for cand in candidates:
                    if cand.get("doc_id") == doc_id:
                        cand["_bm25_score"] = b.get("_bm25_score", cand.get("_bm25_score", 0.0))
                        cand.setdefault("_chroma_score", cand.get("_score", 0.0))
                        break

        # Weighted blend for balanced/semantic queries; keyword_heavy handled separately
        if info["strategy"] == "balanced":
            BM25_WEIGHT, VECTOR_WEIGHT, TYPE_BOOST = 0.25, 0.55, 0.15
        else:  # semantic
            # BM25 weights inflated after global normalization vs semantic (0-1):
            # BM25 normalizes everything to max=1 across full corpus, not per query.
            # Keep BM25 low so semantic distance actually differentiates.
            BM25_WEIGHT, VECTOR_WEIGHT, TYPE_BOOST = 0.1, 0.75, 0.15

        for c in candidates:
            if info["strategy"] == "keyword_heavy":
                bm25 = c.get("_bm25_score", 0.0)
                c["_score"] = bm25
            else:
                bm25_score = c.get("_bm25_score", 0.0)
                chroma_score = c.get("_chroma_score", c.get("_score", 0.0))
                if info["strategy"] == "balanced":
                    base = (VECTOR_WEIGHT * chroma_score + BM25_WEIGHT * bm25_score) / (VECTOR_WEIGHT + BM25_WEIGHT)
                else:
                    base = VECTOR_WEIGHT * chroma_score + BM25_WEIGHT * bm25_score
                boost = TYPE_BOOST if c.get("doc_type") in ("financial", "earnings_call") else 0.0
                c["_score"] = base + boost

        candidates.sort(key=lambda x: x["_score"], reverse=True)

        # Apply ticker filter from top candidates (after blend)
        if self.ticker_filter:
            candidates = self._apply_ticker_filter(candidates)

        if approach == "hybrid_rerank":
            rerank_pool = candidates[: max(top_k, min(len(candidates), RETRIEVAL_TOP_K_HYBRID))]
            reranked = self._rerank(search_query, rerank_pool)
            remainder_ids = {d.get("doc_id") for d in reranked}
            candidates = reranked + [d for d in candidates if d.get("doc_id") not in remainder_ids]

        log.info("Retrieval done in %.1fs — %d docs returned",
                 time.time() - t0, len(candidates[:top_k]))
        return candidates[:top_k]

    def _apply_ticker_filter(self, candidates: list[dict]) -> list[dict]:
        tid = _TICKER_ID_MAP.get(self.ticker_filter or "")
        if not tid:
            return candidates
        filtered = [d for d in candidates if d.get("ticker_id") == tid]
        return filtered or candidates

    # ── ChromaDB ───────────────────────────────────────────────────────────────
    def _chroma_search(self, query_emb: list, n: int) -> list[dict]:
        try:
            col = self._chroma.get_collection(COLLECTION_NAME)
            resp = col.query(query_embeddings=[query_emb], n_results=n,
                             include=["documents", "metadatas", "distances"])
            results = []
            for doc, meta, dist in zip(
                resp["documents"][0], resp["metadatas"][0], resp["distances"][0],
            ):
                results.append({
                    "chunk_text": doc or "",
                    "ticker_id": meta.get("ticker_id"),
                    "doc_id": meta.get("doc_id"),
                    "doc_type": meta.get("doc_type"),
                    "doc_date": meta.get("doc_date"),
                    "title": meta.get("title", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    # Distances are euclidean (0 = identical); lower is better → higher score
                    "_score": 1.0 / (1.0 + float(dist) if dist is not None else 1.0),
                    "_chroma_score": 1.0 / (1.0 + float(dist) if dist is not None else 1.0),
                })
            return results
        except Exception as exc:
            log.warning("ChromaDB search failed: %s", exc)
            return []

    def _bm25_search(self, query: str, n: int) -> list[dict]:
        if self._bm25 is None or self._bm25._index is None:
            return []
        try:
            results = self._bm25.search(query, top_k=n)
            chunks = []
            for chunk, score in results:
                if score <= 0:
                    continue
                # Chunk structure: {id, chunk_text, metadata: {ticker_id, doc_type, ...}}
                meta = chunk.get("metadata", {}) or {}
                flat = dict(chunk)
                flat["_bm25_score"] = float(score)
                flat["_score"] = float(score)  # _score for consistency
                flat.setdefault("ticker_id", meta.get("ticker_id"))
                flat.setdefault("doc_type", meta.get("doc_type"))
                flat.setdefault("title", meta.get("title", ""))
                flat.setdefault("doc_date", meta.get("doc_date"))
                chunks.append(flat)
            return chunks
        except Exception as exc:
            log.warning("BM25 search failed: %s", exc)
            return []

    # ── Cross-encoder rerank ───────────────────────────────────────────────────
    def _rerank(self, query: str, docs: list[dict]) -> list[dict]:
        global _XC_MODEL
        try:
            if _XC_MODEL is None:
                _XC_MODEL = CrossEncoder(RERANKER_MODEL)
            self._xc_model = _XC_MODEL
            texts = [d.get("chunk_text", "") for d in docs]
            scores = self._xc_model.predict(list(zip([query] * len(texts), texts)))
            scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [dict(d, _rerank_score=float(s)) for d, s in scored]
        except Exception as exc:
            log.warning("Cross-encoder failed (skipping): %s", exc)
            return docs
