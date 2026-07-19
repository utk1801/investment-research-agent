"""Chunk documents into ~512-token pieces for vector storage."""

from __future__ import annotations

import logging
import tiktoken
from dataclasses import dataclass

from src.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

log = logging.getLogger(__name__)

# Use cl100k_base (GPT-4 tokenizer) for token counting
_enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    ticker_id: int
    doc_id: int
    doc_type: str
    doc_date: str | None
    title: str
    chunk_index: int
    chunk_text: str


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Split text into token-based chunks with overlap."""
    tokens = _enc.encode(text)
    if len(tokens) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = _enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap
        if start >= len(tokens) - overlap:
            break
    return chunks


def chunk_document(title: str, content: str, ticker_id: int, doc_id: int,
                   doc_type: str, doc_date: str | None,
                   ticker_symbol: str = "", ticker_name: str = "") -> list[Chunk]:
    """Chunk a single document. Full company name is prepended for financial docs."""
    chunks: list[Chunk] = []

    if doc_type == "news":
        full_text = f"{title}\n\n{content}".strip()
        if count_tokens(full_text) <= CHUNK_SIZE_TOKENS:
            return [Chunk(
                ticker_id=ticker_id, doc_id=doc_id, doc_type=doc_type,
                doc_date=doc_date, title=title, chunk_index=0, chunk_text=full_text,
            )]
        parts = chunk_text(full_text)
    else:
        # Financial: content is already richly formatted from _upsert_financial_doc
        full_text = content.strip()
        parts = chunk_text(full_text)

    for i, text in enumerate(parts):
        chunks.append(Chunk(
            ticker_id=ticker_id,
            doc_id=doc_id,
            doc_type=doc_type,
            doc_date=doc_date,
            title=title,
            chunk_index=i,
            chunk_text=text,
        ))
    return chunks


def chunk_all_documents(engine) -> list[Chunk]:
    """Load all documents from DB and return chunks."""
    from sqlalchemy import text

    # Load ticker metadata for name lookup
    ticker_info: dict[int, dict] = {}
    with engine.begin() as conn:
        ticker_rows = conn.execute(text("SELECT id, symbol, name FROM tickers")).fetchall()
        for tid, sym, name in ticker_rows:
            ticker_info[tid] = {"symbol": sym, "name": name}

        rows = conn.execute(
            text("SELECT id, ticker_id, doc_type, title, content, doc_date FROM documents")
        ).fetchall()

    chunks: list[Chunk] = []
    for row in rows:
        doc_id, ticker_id, doc_type, title, content, doc_date = row
        doc_date_str = doc_date.isoformat() if doc_date else None
        title_str = title or ""
        content_str = content or ""
        ticker = ticker_info.get(ticker_id, {})
        ticker_symbol = ticker.get("symbol", "")
        ticker_name = ticker.get("name", ticker_symbol)
        for chunk in chunk_document(
            title_str, content_str, ticker_id, doc_id, doc_type, doc_date_str,
            ticker_symbol=ticker_symbol, ticker_name=ticker_name
        ):
            chunks.append(chunk)

    log.info("Chunked %d docs → %d chunks", len(rows), len(chunks))
    return chunks