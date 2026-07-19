"""Build ChromaDB vector index and BM25 index from PostgreSQL documents.

Usage:
    python -m scripts.build_indexes --reset   # wipe existing indexes first
"""

import argparse
import logging
import sys
import time

from sqlalchemy import create_engine

from src.config import get_db_url, COLLECTION_NAME, DATA_DIR
from src.ingest.document_chunker import chunk_all_documents
from src.retrieval.embed import embed_texts, Embedder
from src.vectorstore.chromadb_client import (get_or_create_collection, upsert_chunks,
                                             count_collection, reset_collection)
from src.retrieval.bm25_index import BM25Index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def build_chroma(chunks: list, embedder: Embedder, collection=None) -> int:
    log.info("Embedding %d chunks...", len(chunks))
    texts = [c.chunk_text for c in chunks]
    chunk_ids = [f"chunk-{c.doc_id}-{c.chunk_index}" for c in chunks]

    # Batch embed to avoid memory spikes
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

    return upsert_chunks(chroma_chunks, collection)


def build_bm25(chunks: list) -> None:
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
    log.info("BM25 index saved.")


def main(reset: bool = False):
    DATA_DIR.mkdir(exist_ok=True)

    engine = create_engine(get_db_url())

    if reset:
        log.info("Resetting existing collection...")
        reset_collection(COLLECTION_NAME)

    # Step 1: Chunk documents
    log.info("Loading documents from PostgreSQL...")
    chunks = chunk_all_documents(engine)
    if not chunks:
        log.warning("No documents found. Run yfinance_ingest first.")
        return

    log.info("Got %d chunks from %d documents", len(chunks), len(set(c.doc_id for c in chunks)))

    # Step 2: Embed and upsert to ChromaDB
    t0 = time.time()
    embedder = Embedder()
    collection = get_or_create_collection()
    n_vec = build_chroma(chunks, embedder, collection)
    log.info("ChromaDB: %d chunks indexed in %.1fs", n_vec, time.time() - t0)

    # Step 3: Build BM25
    t0 = time.time()
    build_bm25(chunks)
    log.info("BM25 built in %.1fs", time.time() - t0)

    log.info("Build complete. ChromaDB count: %d", count_collection())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB + BM25 indexes")
    parser.add_argument("--reset", action="store_true", help="Reset existing collection first")
    args = parser.parse_args()
    main(reset=args.reset)