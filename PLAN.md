# Investment Research Agent — Implementation Plan

## Context

Build an end-to-end RAG + agentic application for investment research. The app answers natural-language questions about stocks using live + historical financial data, embedded document retrieval, hybrid search, and LLM synthesis. Covers ~50 tickers across S&P sectors with multi-source data ingestion.

**Project root:** `/Users/utkarshgarg/Desktop/Projects/investment-research-agent/`

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| Interface | Streamlit | Fast to build, integrates natively with pandas/plotly |
| Vector DB | Postgres + pgvector | One DB for all data, SQL joins work, easy backup |
| LLM | OpenAI GPT-4o | Strong financial reasoning. Configurable to Ollama for local. |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Fast, 384-dim, good for financial text |
| Orchestration | Python scripts + cron | Simpler than Airflow for this scale |
| Container | Docker Compose | App + Postgres + pgvector in one command |
| Monitoring | Custom Streamlit dashboard | Feedback collection + charts, no extra infra |
| Reranker | cross-encoder (cross-encoder/ms-marco-MiniLM-L-6) | Lightweight, can run CPU-side |

---

## Directory Structure

```
investment-research-agent/
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
├── scripts/
│   └── init_db.sql             # pgvector extension + schema
├── src/
│   ├── __init__.py
│   ├── config.py               # Env vars, constants, ticker list
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── yfinance_ingest.py  # Ticker fundamentals, financials, news
│   │   ├── fmp_ingest.py        # Income stmt, balance sheet, key metrics
│   │   ├── finnhub_ingest.py    # Analyst recs, price targets, earnings
│   │   └── document_chunker.py # Chunk SEC filings, earnings, news
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── run_ingestion.py     # Orchestrates full ingestion, handles retries
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── db.py                # Postgres/pgvector connection + CRUD
│   │   └── schema.sql           # Table DDL (committed for reproducibility)
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embed.py             # Text → embedding via sentence-transformers
│   │   ├── bm25_index.py        # BM25 keyword index (rank_bm25 library)
│   │   ├── retriever.py         # Hybrid search: vector + BM25 score fusion
│   │   └── reranker.py          # Cross-encoder rerank of top-k results
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── prompts.py           # 3+ prompt variants for evaluation
│   │   ├── client.py            # OpenAI API calls, fallback to mock
│   │   └── evaluator.py         # RAGAS-style eval: faithfulness, answer relevance, retrieval precision
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── feedback.py          # Store user feedback, thumbs up/down, ratings
│   │   └── dashboard.py         # Streamlit monitoring page (5+ charts)
│   ├── query_rewriting/
│   │   ├── __init__.py
│   │   └── rewriter.py          # LLM-based query decomposition + expansion
│   └── ui/
│       ├── __init__.py
│       ├── app.py               # Main Streamlit entry point
│       └── pages/
│           ├── research.py      # Main Q&A page
│           ├── dashboard.py     # Monitoring page (shortcut to monitoring/dashboard)
│           └── ingestion.py     # Manual ingestion trigger page
├── tests/
│   ├── test_retrieval.py
│   ├── test_ingestion.py
│   └── test_llm.py
└── notebooks/
    └── evaluation.ipynb         # Compare prompt variants, retrieval strategies
```

---

## Database Schema

### Tables

```sql
-- 1. Tickers
CREATE TABLE tickers (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    name TEXT,
    sector TEXT,
    market_cap BIGINT,
    inserted_at TIMESTAMP DEFAULT NOW()
);

-- 2. Financial snapshots (re-ingested quarterly per ticker)
CREATE TABLE financial_metrics (
    id SERIAL PRIMARY KEY,
    ticker_id INTEGER REFERENCES tickers(id),
    period DATE,
    revenue BIGINT,
    net_income BIGINT,
    free_cash_flow BIGINT,
    total_debt BIGINT,
    equity BIGINT,
    pe_ratio FLOAT,
    pb_ratio FLOAT,
    dividend_yield FLOAT,
    revenue_growth FLOAT,
    inserted_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(ticker_id, period)
);

-- 3. Daily price data
CREATE TABLE prices (
    id SERIAL PRIMARY KEY,
    ticker_id INTEGER REFERENCES tickers(id),
    date DATE NOT NULL,
    open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume BIGINT,
    UNIQUE(ticker_id, date)
);

-- 4. Embedded documents (news, filings, earnings)
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    ticker_id INTEGER REFERENCES tickers(id),
    doc_type TEXT,             -- 'news' | '10k' | '10q' | 'earnings_call'
    source TEXT,               -- 'finnhub' | 'sec_edgar' | 'yfinance'
    title TEXT,
    content TEXT,
    doc_date DATE,
    inserted_at TIMESTAMP DEFAULT NOW()
);

-- 5. Chunked text for embedding (child of documents)
CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    ticker_id INTEGER REFERENCES tickers(id),
    chunk_index INTEGER,
    chunk_text TEXT,
    embedding vector(384),     -- all-MiniLM-L6-v2 output
    doc_date DATE,
    doc_type TEXT
);
CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- 6. User queries and feedback
CREATE TABLE query_log (
    id SERIAL PRIMARY KEY,
    query_text TEXT,
    ticker_filter TEXT,
    rewritten_query TEXT,
    response_text TEXT,
    retrieval_time_ms INTEGER,
    total_time_ms INTEGER,
    model_used TEXT,
    prompt_variant TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 7. User feedback on responses
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES query_log(id),
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    thumbs_up BOOLEAN,
    thumbs_down BOOLEAN,
    comment TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Implementation Steps (Ordered)

### Phase 0 — Foundation (Day 1)
1. Create virtualenv + pyproject.toml, install deps
2. Write `.env.example` with required env vars
3. Write `docker-compose.yml` (app + Postgres + pgvector)
4. Run `scripts/init_db.sql`, verify pgvector extension
5. Write `src/config.py` reading env vars

### Phase 1 — Ingestion (Day 1-2)
6. Implement `scripts/seed_tickers.py` — insert 50 tickers into DB
7. `yfinance_ingest.py` — pull fast_info, income_stmt, cashflow for all 50
8. `fmp_ingest.py` — pull key_metrics, income_stmt, ratios (FMP free tier)
9. `finnhub_ingest.py` — pull analyst recs, price targets, earnings dates
10. `document_chunker.py` — chunk raw docs into ~512-token pieces, emit chunks
11. `pipeline/run_ingestion.py` — orchestrate all 3 ingestors, handle rate limits, retry with backoff

### Phase 2 — Embedding Pipeline (Day 2)
12. `src/retrieval/embed.py` — encode all document_chunks, store vectors in pgvector
13. `src/retrieval/bm25_index.py` — build BM25 index over all chunk_text
14. Write `scripts/build_indexes.py` to run both embedding + BM25 build

### Phase 3 — Retrieval (Day 2-3)
15. `src/retrieval/retriever.py` — hybrid search (vector sim + BM25 rrFusion), return top-k chunks with scores
16. `src/retrieval/reranker.py` — cross-encoder score top-k docs, return final ranking
17. `src/query_rewriting/rewriter.py` — rewrite user query: expand ticker symbols, decompose multi-part questions

### Phase 4 — LLM Layer (Day 3-4)
18. `src/llm/client.py` — OpenAI API client with retries, token counting, mock fallback for tests
19. `src/llm/prompts.py` — 3 prompt templates (basic, detailed, conservative/citations)
20. `src/llm/evaluator.py` — RAGAS-equivalent: faithfulness, answer relevance, context precision, context recall

### Phase 5 — Interface (Day 4)
21. `src/ui/pages/research.py` — main chat UI with ticker selector, query input, response display, source citations
22. `src/ui/pages/ingestion.py` — view ingestion status, trigger re-ingest manually
23. `src/ui/app.py` — sidebar with nav, settings (model select, prompt variant select)

### Phase 6 — Monitoring (Day 5)
24. `src/monitoring/feedback.py` — store feedback per query
25. `src/monitoring/dashboard.py` — 5+ charts: query volume/day, avg latency (retrieval vs total), thumbs up/down ratio, rating distribution, prompt variant comparison

### Phase 7 — Evaluation (Day 5-6)
26. `notebooks/evaluation.ipynb` — compare 3 prompt variants on 20 sample queries, measure RAGAS scores, plot comparison
27. Retrieval eval: compare hybrid vs vector-only vs BM25-only on hit-rate@5

### Phase 8 — Containerization + Final (Day 6-7)
28. `Dockerfile` — Python app, install deps, run Streamlit
29. Verify `docker-compose up` brings up app + DB
30. `README.md` — setup instructions, env vars, run commands, architecture diagram
31. Smoke test: full ingestion runs, query returns answer, feedback logs

---

## Ingestion Pipeline Details

### 50 Tickers by Sector

```python
TICKERS = {
    "Technology": ["NVDA", "AAPL", "MSFT", "GOOGL", "AMAT", "CRM", "ORCL", "INTC"],
    "Finance":   ["JPM", "GS", "MS", "BAC", "WFC", "C", "AXP", "V"],
    "Healthcare":["UNH", "LLY", "JNJ", "ABBV", "PFE", "TMO", "ABT", "DHR"],
    "Consumer":  ["AMZN", "WMT", "COST", "HD", "NKE", "SBUX", "MCD", "TGT"],
    "Energy":    ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO"],
    "Industrials": ["CAT", "DE", "BA", "HON", "UPS", "RTX", "LMT", "GE"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "SPG", "O", "WELL", "DLR"],
}
```

### Rate Limit Strategy

```python
# yfinance: no limit, but add 0.3s sleep between tickers
# FMP free tier: 250 req/day → ~5 req/min safe; 1.2s sleep between calls
# Finnhub free tier: 60 req/min → 1 req/sec; 1.1s sleep between calls

# Per-run schedule:
# Weekly: full financials (income_stmt, balance_sheet, cashflow) for all 50
# Daily: prices, news, fast_info
# Hourly: check for new analyst recommendations
```

### Document Chunking Strategy

```
10-K filings:       chunk at section headers (1, 2, 7A, 9) → ~8-12 chunks per filing
Earnings transcripts: chunk by Q&A turns → ~6-10 chunks per call
News articles:       whole article as one chunk (short) or 2 chunks if > 500 words
Chunk size:         512 tokens (≈ ~300-400 words), no overlap for docs, 50-token overlap for filings
Metadata:           ticker_id, doc_type, doc_date, source
```

---

## Retrieval Pipeline (Query Flow)

```
User query
    │
    ▼
┌─────────────────────────────────────────────┐
│  1. Query Rewriter (LLM call)               │
│     - Extract ticker symbols                │
│     - Decompose multi-part questions         │
│     - Expand abbreviations                   │
│     Example:                                 │
│       "How did NVDA do compared to AAPL     │
│        last quarter?" →                      │
│       ["NVDA earnings last quarter",        │
│        "AAPL earnings last quarter",         │
│        "NVDA vs AAPL comparison"]            │
└─────────────────────┬───────────────────────┘
                      │ (rewritten queries)
                      ▼
┌─────────────────────────────────────────────┐
│  2. Parallel Retrieval (per rewritten query)│
│                                             │
│     Vector search: pgvector top-10          │
│         cosine similarity                   │
│     BM25 search: rank_bm25 top-10           │
│     ─────────────────────────────────       │
│     RRF fusion (k=60): combine both ranks   │
│     → top-k = 20 candidates                 │
└─────────────────────┬───────────────────────┘
                      │ (ranked candidates)
                      ▼
┌─────────────────────────────────────────────┐
│  3. Reranker (cross-encoder)                │
│     Score top-20 by relevance to query      │
│     Output: top-5 chunks with rerank scores │
└─────────────────────┬───────────────────────┘
                      │ (final context chunks)
                      ▼
┌─────────────────────────────────────────────┐
│  4. LLM Generate                           │
│     Prompt template: system + context +    │
│     user query → synthesize answer          │
│     Cite sources inline                     │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
                 Final Response
              (with source citations)
```

---

## Prompt Templates (3 Variants for Evaluation)

### Variant A — Basic (baseline)
```
You are a financial analyst. Answer the question based on the provided context.
If the context is insufficient, say you don't know.
Context: {context}
Question: {question}
Answer:
```

### Variant B — Structured + Citations
```
You are a senior financial analyst. Use the provided context to answer the question.
For each statement in your answer, cite the source using [Source N] notation.
If the context doesn't fully answer the question, acknowledge the gap.
---
Context:
{context}
---
Question: {question}
---
Answer (structured format):
[Findings]
[Supporting Data]
[Limitations / Gaps]
Sources:
```

### Variant C — Conservative / No Fabrication
```
You are a precise financial analyst. Answer using ONLY the provided context.
Do NOT make up numbers, ratios, or facts. If a fact is not in the context, say "not available in the provided data."
If the context has no relevant information, respond: "I don't have information about that in the available data."
---
Context:
{context}
---
Question: {question}
---
Answer:
```

---

## Monitoring Dashboard Spec (5+ Charts)

| Chart | Type | Data Source | Metric |
|---|---|---|---|
| Queries per day | Bar chart | `query_log.created_at` grouped by day | volume trend |
| Retrieval vs total latency | Line chart | `retrieval_time_ms`, `total_time_ms` | SLA monitoring |
| Thumbs up / down ratio | Gauge or pie | `feedback.thumbs_up` aggregate | user satisfaction |
| Rating distribution | Histogram | `feedback.rating` 1-5 | quality overview |
| Prompt variant comparison | Multi-line | avg score by `prompt_variant` | which prompt wins |

---

## Evaluation Framework

### Retrieval Eval
- **Metric: Hit Rate @ K** — is the chunk containing the answer in top-K?
- Run 20 golden queries manually created, compare hybrid vs vector-only vs BM25-only
- Golden queries: e.g. "What was NVDA's revenue growth in Q4 2025?", "Which bank had the highest dividend yield?"
- Report Precision@5, Recall@5 alongside hit rate

### LLM Eval
- **RAGAS metrics** (install `ragas`): faithfulness, answer_relevancy, context_precision, context_recall
- Use 20 golden query / answer pairs (human-written baseline)
- Compare all 3 prompt variants on the same retrieval output

### A/B Testing in Production
- Randomly assign prompt variant per query
- Store `prompt_variant` in `query_log`
- Dashboard shows confidence interval per variant

---

## Key Files — What to Write and When

| File | Lines (est.) | Priority |
|---|---|---|
| `docker-compose.yml` | 60 | 🔴 Must-do first |
| `pyproject.toml` | 30 | 🔴 Must-do first |
| `.env.example` | 15 | 🔴 Must-do first |
| `scripts/init_db.sql` | 40 | 🔴 Must-do first |
| `src/config.py` | 50 | 🔴 Must-do first |
| `src/ingest/yfinance_ingest.py` | 80 | 🟡 Day 1 |
| `src/pipeline/run_ingestion.py` | 60 | 🟡 Day 1 |
| `src/vectorstore/db.py` | 60 | 🟡 Day 2 |
| `src/retrieval/embed.py` | 50 | 🟡 Day 2 |
| `src/retrieval/bm25_index.py` | 50 | 🟡 Day 2 |
| `src/retrieval/retriever.py` | 80 | 🟠 Day 3 |
| `src/query_rewriting/rewriter.py` | 50 | 🟠 Day 3 |
| `src/llm/prompts.py` | 60 | 🟠 Day 3 |
| `src/llm/client.py` | 50 | 🟠 Day 3 |
| `src/llm/evaluator.py` | 60 | 🟡 Day 4 |
| `src/ui/pages/research.py` | 150 | 🟠 Day 4 |
| `src/ui/pages/dashboard.py` | 80 | 🟡 Day 5 |
| `src/ui/pages/ingestion.py` | 60 | 🟡 Day 5 |
| `src/monitoring/feedback.py` | 40 | 🟡 Day 5 |
| `notebooks/evaluation.ipynb` | — | 🟢 Day 6 |
| `Dockerfile` | 20 | 🔴 Final |
| `README.md` | 80 | 🔴 Final |
| `tests/` | — | 🟢 Throughout |

---

## Verification & Testing

1. **Ingestion smoke test:** run `python -m src.pipeline.run_ingestion` — all 50 tickers ingest, no crashes
2. **Vector store test:** query "revenue growth NVIDIA" — correct chunks returned
3. **End-to-end test:** ask "What was AAPL's P/E ratio last quarter?" — answer matches ingested data
4. **Feedback test:** click thumbs down → check `feedback` table has row
5. **Docker test:** `docker compose up` → app at localhost:8501, Postgres at localhost:5432
6. **Eval notebook:** run cells → compare 3 prompt variants, see RAGAS scores printed

---

## Alternative Choices (if scope needs to shift)

| Decision | Default | Alternative |
|---|---|---|
| Vector DB | pgvector | Qdrant (better async perf) |
| LLM | OpenAI GPT-4o | Ollama (local, free) |
| Orchestration | Python + cron | Kestra (visual DAG, heavier) |
| Embedding | sentence-transformers | OpenAI text-embedding-3-small |
| Reranker | cross-encoder | Cohere rerank API |
| Chat | Streamlit | Gradio (simpler) |
| Monitoring | custom dashboard | Evidently AI (pre-built) |