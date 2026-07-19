"""Orchestrate full ingestion: yfinance → PostgreSQL → chunk → embed → ChromaDB + BM25."""

from __future__ import annotations

import logging
import time
import argparse
from datetime import datetime

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_db_url
from src.ingest.yfinance_ingest import run_full_ingestion, backfill_financial_docs, backfill_earnings_docs
from src.ingest.earnings_transcript_ingest import ingest_transcripts
from src.retrieval.embed import Embedder
from src.vectorstore.chromadb_client import (
    get_or_create_collection, upsert_chunks, count_collection, reset_collection,
    COLLECTION_NAME)
from src.retrieval.bm25_index import BM25Index
from src.ingest.document_chunker import chunk_all_documents, Chunk
from sqlalchemy import create_engine

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self):
        self.engine = create_engine(get_db_url())

    def run(self, step: str = "all") -> dict:
        """Run ingestion pipeline.

        step:
            all         — full pipeline (yfinance + transcripts + embed + BM25)
            fetch       — yfinance data only → PostgreSQL
            transcripts — static earnings transcripts → PostgreSQL documents
            backfill    — rewrite financial_metrics rows as documents
            embed       — chunk + embed → ChromaDB
            bm25        — build BM25 index
            stats       — print table counts
        """
        results = {}
        start = time.time()

        if step in ("all", "fetch"):
            log.info("=== STEP: Fetch yfinance data ===")
            results["fetch"] = run_full_ingestion()

        if step in ("all", "transcripts"):
            log.info("=== STEP: Ingest static earnings transcripts ===")
            results["transcripts"] = ingest_transcripts()

        if step == "backfill":
            log.info("=== STEP: Backfill financial docs ===")
            results["backfill"] = {"docs": backfill_financial_docs()}

        if step == "earnings":
            log.info("=== STEP: Backfill earnings docs ===")
            results["earnings"] = {"docs": backfill_earnings_docs()}

        if step in ("all", "embed"):
            log.info("=== STEP: Embed chunks to ChromaDB ===")
            results["embed"] = self._embed()

        if step in ("all", "bm25"):
            log.info("=== STEP: Build BM25 index ===")
            results["bm25"] = self._bm25()

        if step == "stats":
            results["stats"] = self._stats()

        log.info("Pipeline complete in %.1fs", time.time() - start)
        return results

    def _embed(self) -> dict:
        chunks = chunk_all_documents(self.engine)
        if not chunks:
            log.warning("No documents to embed. Run fetch step first.")
            return {"chunks": 0, "embedded": 0}

        log.info("Embedding %d chunks...", len(chunks))
        embedder = Embedder()
        texts = [c.chunk_text for c in chunks]

        batch_size = 64
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(embedder.embed(batch))

        chroma_chunks = []
        for c, emb in zip(chunks, all_embeddings):
            chroma_chunks.append({
                "id": f"chunk-{c.doc_id}-{c.chunk_index}",
                "embedding": emb,
                "chunk_text": c.chunk_text,
                "metadata": {
                    "ticker_id": c.ticker_id,
                    "doc_id": c.doc_id,
                    "doc_type": c.doc_type,
                    "doc_date": c.doc_date or "",
                    "title": c.title or "",
                    "chunk_index": c.chunk_index,
                },
            })

        # Reset collection on full pipeline to avoid stale docs
        reset_collection(COLLECTION_NAME)
        n = upsert_chunks(chroma_chunks)
        return {"chunks": len(chunks), "embedded": n}

    def _bm25(self) -> dict:
        chunks = chunk_all_documents(self.engine)
        if not chunks:
            return {"chunks": 0}

        bm25_chunks = [
            {
                "id": f"chunk-{c.doc_id}-{c.chunk_index}",
                "chunk_text": c.chunk_text,
                "metadata": {
                    "ticker_id": c.ticker_id,
                    "doc_id": c.doc_id,
                    "doc_type": c.doc_type,
                    "doc_date": c.doc_date or "",
                    "title": c.title or "",
                },
            }
            for c in chunks
        ]
        idx = BM25Index()
        idx.build(bm25_chunks)
        idx.save()
        return {"chunks": len(bm25_chunks)}

    def _stats(self) -> dict:
        from sqlalchemy import text
        stats = {}
        with self.engine.connect() as conn:
            for table, col in [
                ("tickers", "id"),
                ("prices", "id"),
                ("financial_metrics", "id"),
                ("documents", "id"),
            ]:
                r = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                stats[table] = r
        stats["chroma"] = count_collection()
        return stats


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run ingestion pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "fetch", "transcripts", "backfill", "earnings", "embed", "bm25", "stats"],
        default="all",
        help="Which step to run (default: all)",
    )
    parser.add_argument("--reset", action="store_true", help="Reset ChromaDB collection before embed")
    args = parser.parse_args()

    if args.reset and args.step in ("all", "embed"):
        reset_collection(COLLECTION_NAME)

    log.info("Starting pipeline @ %s | step=%s", datetime.now().isoformat(), args.step)
    pipeline = Pipeline()
    results = pipeline.run(step=args.step)

    if args.step == "stats":
        for name, count in results["stats"].items():
            log.info("  %-20s: %s", name, count)
    else:
        log.info("Results: %s", results)


if __name__ == "__main__":
    main()