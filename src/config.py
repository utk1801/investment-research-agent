"""Global configuration — env vars, ticker list, constants."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── Load .env ─────────────────────────────────────────────────────────────
_dotenv = Path(__file__).parent.parent / ".env"
if _dotenv.exists():
    load_dotenv(_dotenv, override=True)


# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"


# ── Environment ────────────────────────────────────────────────────────────
APP_ENV = os.getenv("APP_ENV", "local")


# ── PostgreSQL ─────────────────────────────────────────────────────────────
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "invest_agent")
POSTGRES_USER = os.getenv("POSTGRES_USER", "invest_agent")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "invest123")


def get_db_url() -> str:
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )


# ── ChromaDB ────────────────────────────────────────────────────────────────
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = os.getenv("CHROMADB_PORT", "8000")

CHROMADB_URL = f"http://{CHROMADB_HOST}:{CHROMADB_PORT}"


# ── LLM ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))


# ── Embeddings & Retrieval ─────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-128"
)

RETRIEVAL_TOP_K_HYBRID = 20   # candidates before reranking
RETRIEVAL_TOP_K_FINAL = 5     # chunks passed to LLM
RRF_K = 60                    # RRF fusion parameter

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 50

COLLECTION_NAME = "investment_docs"


# ── Ticker List ──────────────────────────────────────────────────────────────
TICKERS: dict[str, list[str]] = {
    "Technology":   ["AAPL", "MSFT", "NVDA"],
    "Finance":      ["JPM",  "GS"],
    "Healthcare":   ["UNH",  "LLY"],
    "Consumer":     ["AMZN", "WMT"],
    "Energy":       ["XOM"],
    "Industrials":  ["CAT"],
    "Real Estate":  ["AMT"],
}

ALL_TICKERS: list[str] = [
    t for tickers in TICKERS.values() for t in tickers
]

TICKER_SECTORS: dict[str, str] = {
    t: sector for sector, tickers in TICKERS.items() for t in tickers
}


# ── Ingestion ────────────────────────────────────────────────────────────────
YFINANCE_SLEEP_SECS = 0.3        # delay between tickers
HISTORY_PERIOD = "1y"            # how far back to pull prices
FINANCIAL_QUARTERS = 8            # how many quarters of financials

# ── Static Transcripts ────────────────────────────────────────────────────────
TRANSCRIPT_DIR = DATA_DIR / "transcripts"


# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")