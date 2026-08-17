"""Load earnings call transcripts from static JSON files and write to PostgreSQL."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from sqlalchemy import create_engine, text

from src.config import TRANSCRIPT_DIR, ALL_TICKERS, get_db_url

log = logging.getLogger(__name__)


def _engine():
    return create_engine(get_db_url())


def _ensure_documents_schema(engine) -> None:
    """Keep document constraints compatible with repeatable Airflow ingestion."""
    statements = [
        """
        ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_doc_type_check
        """,
        """
        ALTER TABLE documents
            ADD CONSTRAINT documents_doc_type_check
            CHECK (doc_type IN ('earnings_call', 'news', 'financial'))
        """,
        """
        DELETE FROM documents d
        USING (
            SELECT id,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker_id, doc_type, source, title,
                                    COALESCE(doc_date, DATE '0001-01-01')
                       ORDER BY id
                   ) AS rn
            FROM documents
            WHERE doc_type = 'earnings_call'
              AND source = 'static'
        ) dup
        WHERE d.id = dup.id
          AND dup.rn > 1
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_static_transcript_unique
            ON documents (
                ticker_id,
                doc_type,
                source,
                title,
                (COALESCE(doc_date, DATE '0001-01-01'))
            )
            WHERE doc_type = 'earnings_call'
              AND source = 'static'
        """,
    ]
    with engine.begin() as conn:
        for sql in statements:
            conn.execute(text(sql))


def _get_ticker_id(engine, symbol: str) -> int | None:
    with engine.begin() as conn:
        result = conn.execute(
            text("SELECT id FROM tickers WHERE symbol = :sym"),
            {"sym": symbol},
        ).fetchone()
        return result[0] if result else None


def _get_date(doc_date_str: str) -> date | None:
    if not doc_date_str:
        return None
    try:
        from datetime import datetime
        return datetime.strptime(doc_date_str, "%Y-%m-%d").date()
    except Exception:
        return None


def _assemble_content(data: dict) -> str:
    """Flatten a transcript JSON into a single readable string for chunking."""
    s = data.get("sections", {})

    prepared = s.get("prepared_remarks", "")
    financials = s.get("financial_highlights", "")
    qanda = s.get("qanda_session", "")
    outlook = s.get("forward_outlook", "")

    return (
        f"{data.get('title', '')}\n\n"
        f"{prepared}\n\n"
        f"FINANCIAL HIGHLIGHTS\n{financials}\n\n"
        f"ANALYST Q&A\n{qanda}\n\n"
        f"FORWARD OUTLOOK\n{outlook}"
    )


def ingest_transcripts() -> dict:
    """Load all transcripts from TRANSCRIPT_DIR and write to documents table."""
    engine = _engine()
    _ensure_documents_schema(engine)
    transcript_dir = Path(TRANSCRIPT_DIR)

    if not transcript_dir.exists():
        log.error("Transcript directory not found: %s", transcript_dir)
        return {"loaded": 0, "skipped": 0, "errors": 0}

    files = list(transcript_dir.glob("*.json"))
    if not files:
        log.warning("No transcript files found in %s", transcript_dir)
        return {"loaded": 0, "skipped": 0, "errors": 0}

    log.info("Found %d transcript files in %s", len(files), transcript_dir)

    counts = {"upserted": 0, "skipped": 0, "errors": 0}

    for fpath in sorted(files):
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            log.warning("Failed to parse %s: %s", fpath.name, exc)
            counts["errors"] += 1
            continue

        ticker = data.get("ticker", "").strip().upper()
        if not ticker or ticker not in ALL_TICKERS:
            log.warning("Unknown ticker '%s' in %s — skipping", ticker, fpath.name)
            counts["skipped"] += 1
            continue

        ticker_id = _get_ticker_id(engine, ticker)
        if not ticker_id:
            log.warning("Ticker %s not found in DB — skipping", ticker)
            counts["skipped"] += 1
            continue

        title = data.get("title", f"{ticker} Earnings Call").strip()
        content = _assemble_content(data)
        doc_date = _get_date(data.get("call_date", ""))

        with engine.begin() as conn:
            # Only write transcripts — keep existing news/financial docs
            conn.execute(
                text("""
                    INSERT INTO documents (ticker_id, doc_type, title, content, doc_date, source)
                    VALUES (:ticker_id, 'earnings_call', :title, :content, :doc_date, 'static')
                    ON CONFLICT (
                        ticker_id,
                        doc_type,
                        source,
                        title,
                        (COALESCE(doc_date, DATE '0001-01-01'))
                    )
                    WHERE doc_type = 'earnings_call'
                      AND source = 'static'
                    DO UPDATE SET
                        content = EXCLUDED.content
                """),
                {
                    "ticker_id": ticker_id,
                    "title": title,
                    "content": content,
                    "doc_date": doc_date,
                },
            )
        counts["upserted"] += 1
        log.info("  Upserted %s — %s", ticker, data.get("quarter", ""))

    log.info("Transcript ingest done: upserted=%d skipped=%d errors=%d",
             counts["upserted"], counts["skipped"], counts["errors"])
    return counts


if __name__ == "__main__":
    import sys, logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                         stream=sys.stdout)

    counts = ingest_transcripts()

    # Stats
    engine = _engine()
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM documents WHERE doc_type = 'earnings_call'")).scalar()
        log.info("Ingest result: %s", counts)
        log.info("Documents table now has %d earnings_call rows", total)
