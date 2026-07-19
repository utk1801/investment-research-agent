"""ChromaDB client — HTTP mode to connect to the Docker ChromaDB service."""

from __future__ import annotations

import logging
from typing import Sequence

import chromadb
from chromadb.config import Settings

from src.config import CHROMADB_URL, COLLECTION_NAME, EMBEDDING_DIM, CHROMA_DIR

log = logging.getLogger(__name__)

# Use persistent client — writes to CHROMA_DIR which is mounted as a volume
_client: chromadb.PersistentClient | None = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
    log.info("ChromaDB client connected. Persist dir: %s", CHROMA_DIR)
    return _client


def get_or_create_collection(name: str = COLLECTION_NAME):
    client = get_client()
    try:
        return client.get_collection(name)
    except Exception:
        log.info("Creating ChromaDB collection: %s", name)
        return client.create_collection(
            name=name,
            metadata={"dimension": EMBEDDING_DIM},
        )


def upsert_chunks(chunks: list[dict], collection=None) -> int:
    if not chunks:
        return 0
    col = collection or get_or_create_collection()

    ids = [c["id"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    documents = [c["chunk_text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    log.info("Upserted %d chunks → ChromaDB collection '%s'", len(chunks), col.name)
    return len(chunks)


def count_collection(name: str = COLLECTION_NAME) -> int:
    try:
        col = get_client().get_collection(name)
        return col.count()
    except Exception:
        return 0


def reset_collection(name: str = COLLECTION_NAME) -> None:
    client = get_client()
    try:
        client.delete_collection(name)
        log.info("Deleted collection: %s", name)
    except Exception:
        log.warning("Collection '%s' not found", name)
    get_or_create_collection(name)