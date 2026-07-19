# Investment Research Agent — Project Plan v2

## Overview

End-to-end RAG application for investment research. Answers natural-language questions about stocks using live financial data, embedded document retrieval, hybrid search (BM25 + vector), re-ranking, and LLM synthesis. Covers 15 tickers across 7 S&P sectors. Designed for local single-user testing with production-grade observability.

**Project root:** `/Users/utkarshgarg/Desktop/Projects/investment-research-agent/`

---

## Grading Rubric

| Criterion | Target | Implementation |
|---|---|---|
| Problem Description | 2 | This doc (well-described) |
| Retrieval Flow | 2 | ChromaDB vector store + SQLite structured data |
| Retrieval Evaluation | 2 | Hit-rate@K over 3 approaches (vector, BM25, hybrid) |
| LLM Evaluation | 2 | RAGAS metrics (faithfulness, relevance, precision, recall) |
| Interface | 2 | Streamlit web app |
| Ingestion Pipeline | 2 | One-shot + watchtower cron (fully automated) |
| Monitoring | 2 | Grafana dashboard (5+ charts) + user feedback capture |
| Containerization | 2 | docker-compose (app + PostgreSQL + ChromaDB + Grafana) |
| Reproducibility | 2 | Complete setup instructions in README |
| Best Practice: Hybrid Search | +1 | BM25 keyword + ChromaDB vector RRF fusion |
| Best Practice: Re-ranking | +1 | Cross-encoder rerank of top-20 hybrid candidates → top-5 |
| Best Practice: Query Rewriting | +1 | LLM-based query decomposition + ticker extraction |

**Max score: 21/18 (target 18+)**.

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| Interface | Streamlit | Fast, native pandas/plotly, web-based |
| Structured DB | PostgreSQL | Persistent, SQL joins, Grafana native |
| Vector DB | ChromaDB (in-process) | Zero infra, fast, Python-native, no pgvector折腾 |
| LLM | OpenAI GPT-4o | Strong financial reasoning. Configurable to Ollama for local. |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, 384-dim) | Fast, good for financial text |
| Reranker | cross-encoder (cross-encoder/ms-marco-MiniLM-L-6) | Lightweight CPU-side re-ranking |
| BM25 | rank_bm25 | Keyword search complement to vector |
| Monitoring | Grafana + Prometheus | 5+ charts, live dashboards, no extra cost |
| Orchestration | Watchtower (auto-restart) + cron schedule | Minimal ops overhead |
| Container | Docker Compose | App + DB + ChromaDB + Grafana in one command |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                       │
│   Query → Rewrite → Retrieve → Rerank → LLM → Answer │
└───────────────────────┬─────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            │     Hybrid Search      │
            │  (BM25 + ChromaDB)     │
            │       RRF k=60         │
            └───────┬────────────────┘
                    │
      ┌─────────────┴──────────────┐
      │     ChromaDB (vectors)      │
      │   sentence-transformers     │
      └─────────────────────────────┘
      ┌─────────────────────────────┐
      │   PostgreSQL (struct data)   │
      │  prices, financials, tickers  │
      └─────────────────────────────┘
      ┌─────────────────────────────┐
      │   Grafana + Prometheus        │
      │   query latency, retrieval,   │
      │   feedback, rating charts     │
      └─────────────────────────────┘
```

---

## Directory Structure

```
investment-research-agent/
├── .env.example
├── docker-compose.yml          # Streamlit + PostgreSQL + ChromaDB + Grafana
├── Dockerfile
├── pyproject.toml
├── README.md                    # Setup, screenshots, examples
├── scripts/
│   ├── init_db.sql              # PostgreSQL schema + init data
│   └── build_indexes.py         # Embed + BM25 build script
├── src/
│   ├── __init__.py
│   ├── config.py                # Env vars, ticker list
│   ├── ingest/
│   │   ├── __init__.py
│   │   └── yfinance_ingest.py  # Fundamentals, financials, prices, news
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── run_ingestion.py     # Orchestrate all ingestors, handle retries
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── chromadb_client.py   # ChromaDB collection CRUD + similarity search
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embed.py             # Text → embedding
│   │   ├── bm25_index.py        # BM25 keyword index
│   │   ├── retriever.py         # Hybrid search + reranking pipeline
│   │   └── reranker.py          # Cross-encoder reranking
│   ├── query_rewriting/
│   │   ├── __init__.py
│   │   └── rewriter.py          # LLM query decomposition + ticker extraction
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── prompts.py           # 3 prompt variants with evaluation configs
│   │   ├── client.py            # OpenAI API client with mock fallback
│   │   └── evaluator.py         # RAGAS-style eval metrics
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── feedback.py          # Store user feedback in PostgreSQL
│   │   └── metrics.py           # Export Prometheus metrics to PostgreSQL
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── eval_runner.py       # Retrieval + LLM eval script (hit-rate, RAGAS)
│   └── ui/
│       ├── __init__.py
│       ├── app.py               # Streamlit main entry + sidebar
│       └── pages/
│           ├── research.py       # Main Q&A chat page
│           ├── dashboard.py      # Redirect to Grafana
│           └── ingestion.py      # Manual ingestion + status page
├── notebooks/
│   └── evaluation.ipynb         # Compare 3 retrieval approaches + prompt variants
└── data/
    └── chroma_db/               # ChromaDB persisted collection (gitignored)
```

---

## Database Schema

### PostgreSQL (structured data)

```sql
-- 1. Tickers
CREATE TABLE tickers (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    name TEXT,
    sector TEXT,
    inserted_at TIMESTAMP DEFAULT NOW()
);

-- 2. Financial snapshots (re-ingested quarterly)
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

-- 4. Raw documents (earnings calls + news from yfinance)
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    ticker_id INTEGER REFERENCES tickers(id),
    doc_type TEXT CHECK (doc_type IN ('earnings_call', 'news')),
    title TEXT,
    content TEXT,
    doc_date DATE,
    inserted_at TIMESTAMP DEFAULT NOW()
);

-- 5. User queries (for latency + monitoring)
CREATE TABLE query_log (
    id SERIAL PRIMARY KEY,
    query_text TEXT,
    ticker_filter TEXT,
    rewritten_query TEXT,
    retrieval_chunks INTEGER,
    retrieval_time_ms INTEGER,
    total_time_ms INTEGER,
    model_used TEXT,
    prompt_variant TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 6. User feedback on responses
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES query_log(id),
    thumbs_up BOOLEAN,
    thumbs_down BOOLEAN,
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    comment TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### ChromaDB (vector store)

ChromaDB client persists to `./data/chroma_db/` via Docker volume.

```
Collection: investment_docs

Document schema per chunk:
{
  "ticker_id": int,
  "doc_id": int,          -- FK to PostgreSQL documents.id
  "doc_type": str,         -- 'earnings_call' | 'news'
  "doc_date": date,
  "title": str,
  "chunk_index": int,
  "chunk_text": str,      -- ~512 tokens
  "embedding": vector(384)
}
```

---

## Ticker List (15 tickers)

```python
TICKERS = {
    "Technology":   ["AAPL", "MSFT", "NVDA"],
    "Finance":      ["JPM", "GS"],
    "Healthcare":   ["UNH", "LLY"],
    "Consumer":     ["AMZN", "WMT"],
    "Energy":       ["XOM"],
    "Industrials":  ["CAT"],
    "Real Estate":  ["AMT"],
}
# 11 tickers — add 4 more if room: GOOGL, GS, JNJ, MPC
```

---

## Ingestion Pipeline

### Data Sources (yfinance only)

```
yfinance per ticker:
  - fast_info:       market cap, pe_ratio, dividend_yield, 52w range
  - income_stmt:     quarterly revenue, net income (8 quarters)
  - cashflow:        free cash flow (8 quarters)
  - balance_sheet:   total_debt, total_equity (8 quarters)
  - history(period=1y): daily OHLCV prices
  - news:            recent news items (title + summary)
  - earnings_dates:  upcoming earnings dates
```

### Chunking Strategy

```
Earnings calls: chunk by Q&A turns → ~8-10 chunks per call, 50-token overlap
News articles:  whole article as 1 chunk if short; 2 chunks if > 500 words
Chunk size:     512 tokens (~300-400 words), no overlap for news, 50-token for calls
```

### Hybrid Search

```
Per query:
  1. ChromaDB vector search → top-20 (cosine similarity)
  2. BM25 keyword search → top-20
  3. RRF fusion (k=60) → top-20 ranked candidates
  4. Cross-encoder rerank → top-5 final chunks
```

### Rate Limit Strategy

```
yfinance: no limit
  → 0.3s sleep between tickers
  → 15 tickers × 0.3s = ~5s ingestion (fast)

Schedule (via cron inside container):
  Daily:   prices + news + fast_info
  Weekly:  financials (income_stmt, cashflow, balance_sheet)
  Hourly:  check for new news / earnings dates
```

---

## Retrieval Pipeline (Query Flow)

```
User query
    │
    ▼
┌────────────────────────────────────────────────────┐
│  1. Query Rewriter (LLM call)                       │
│     - Extract / normalize ticker symbols             │
│     - Decompose multi-part questions                │
│     Example:                                         │
│       "How did AAPL do vs MSFT last quarter?" →     │
│       ["AAPL earnings last quarter",                 │
│        "MSFT earnings last quarter",                 │
│        "AAPL vs MSFT comparison"]                    │
└──────────────────────┬─────────────────────────────┘
                       │ (rewritten queries)
                       ▼
┌────────────────────────────────────────────────────┐
│  2. Parallel Hybrid Retrieval (per rewritten query) │
│                                                    │
│     ChromaDB vector: cosine similarity top-20      │
│     BM25 keyword:    top-20                        │
│     RRF fusion (k=60): merge both rank lists       │
│     → top-20 candidates per rewritten query         │
└──────────────────────┬─────────────────────────────┘
                       │ (ranked candidates)
                       ▼
┌────────────────────────────────────────────────────┐
│  3. Cross-encoder Reranker                         │
│     Score top-20 by relevance to original query    │
│     Output: top-5 chunks with scores               │
└──────────────────────┬─────────────────────────────┘
                       │ (final context)
                       ▼
┌────────────────────────────────────────────────────┐
│  4. LLM Generate                                  │
│     Prompt template: system + context + user query  │
│     Cite sources inline [Source N]                 │
│     Return structured answer                       │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
                  Final Response
             (with source citations + ticker)
```

---

## Prompt Templates (3 Variants)

### Variant A — Basic
```
You are a financial analyst. Answer based on the provided context.
If context is insufficient, say you don't know.

Context: {context}
Question: {question}
Answer:
```

### Variant B — Structured + Citations
```
You are a senior financial analyst. Use the provided context to answer.
Cite each statement using [Source N] notation.
Acknowledge gaps if context is incomplete.

---
Context:
{context}
---
Question: {question}
---
Answer (structured):
[Findings]
[Supporting Data]
[Limitations / Gaps]
Sources:
```

### Variant C — Conservative / No Fabrication
```
You are a precise financial analyst. Answer using ONLY the provided context.
Do NOT make up numbers, ratios, or facts. If not in context, say "not available."

---
Context:
{context}
---
Question: {question}
---
Answer:
```

---

## Monitoring & Feedback

### Grafana Dashboard (5+ charts)

All data sourced directly from PostgreSQL via Grafana's built-in PostgreSQL datasourse plugin.

| # | Chart | Type | Query / Source |
|---|---|---|---|
| 1 | Queries per day | Bar chart | `query_log.created_at` grouped by day |
| 2 | Retrieval vs total latency | Line chart | `retrieval_time_ms` vs `total_time_ms` avg/day |
| 3 | Thumbs up / down ratio | Pie / gauge | `feedback.thumbs_up` vs `thumbs_down` sum |
| 4 | Rating distribution | Histogram | `feedback.rating` 1-5 distribution |
| 5 | Prompt variant usage | Pie chart | `query_log.prompt_variant` count |
| 6 | Retrieval chunk count distribution | Histogram | `query_log.retrieval_chunks` |
| 7 | Total feedback count | Stat tile | `feedback.id` count |

### Grafana Setup

- Grafana listens on `http://localhost:3000`
- PostgreSQL added as a datasource: `postgresql://invest_agent:invest123@postgres:5432/invest_agent`
- Dashboard JSON imported via provisioning (`/etc/grafana/provisioning/dashboards/`)
- Prometheus datasource not required — Grafana queries PostgreSQL directly

### User Feedback Flow

```
UI: thumbs up/down + 1-5 star rating + optional comment → stored in feedback table
Dashboard: updates live via PostgreSQL datasource refresh
```

---

## Evaluation

### Retrieval Evaluation (Hit-Rate@K)

Run 15 golden queries across 3 approaches:
```
1. Vector-only (ChromaDB similarity)
2. BM25-only (keyword search)
3. Hybrid (RRF fusion) ← default
4. Hybrid + Reranked (default + cross-encoder)

Metrics: Hit-Rate@5, Hit-Rate@10, MRR@10
```

### LLM Evaluation (RAGAS-style)

Install `ragas` library. Run 3 prompt variants on 15 golden queries.

Metrics:
- **Faithfulness**: does the answer follow from the context?
- **Answer Relevancy**: is the answer relevant to the question?
- **Context Precision**: is the relevant context ranked high?
- **Context Recall**: does the context contain the answer?

### Golden Query Set (15 queries)

```
1. "What was AAPL's revenue last quarter?"                    → AAPL financials
2. "How did MSFT's free cash flow change over the last year?" → MSFT cashflow
3. "Which bank had the highest dividend yield?"              → JPM, GS
4. "What is NVDA's current P/E ratio?"                       → NVDA metrics
5. "How did UNH perform in their latest earnings call?"      → UNH earnings
6. "Compare AAPL and MSFT revenue growth"                    → AAPL vs MSFT
7. "What sectors had the best performance this year?"       → multi-ticker
8. "Give me the latest news about WMT"                       → WMT news
9. "What is the 52-week range for XOM?"                      → XOM fast_info
10. "How much debt does CAT carry?"                           → CAT balance sheet
11. "What did LLY report for net income?"                     → LLY financials
12. "Summarize AMZN's latest earnings call highlights"         → AMZN earnings
13. "What was the trading volume for JPM yesterday?"         → JPM prices
14. "Which tech stock had the best return this month?"        → multi-ticker
15. "What does the analyst say about NVDA's price target?"   → NVDA news
```

### A/B Prompt Testing

- Prompt variant assigned randomly per query (1:1:1 distribution)
- `prompt_variant` stored in `query_log`
- Grafana dashboard filters by variant to compare ratings

---

## Docker Compose Services

```yaml
services:
  postgres:
    image: postgres:16
    environment: [POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB]
    volumes: [pgdata:/var/lib/postgresql/data]
    ports: ["5432:5432"]

  chromadb:
    image: chromadb/chroma:latest
    ports: ["8000:8000"]
    volumes: [chroma_data:/chroma/chroma_db]

  app:
    build: .
    depends_on: [postgres, chromadb]
    env_file: [.env]
    ports: ["8501:8501"]

  grafana:
    image: grafana/grafana:latest
    depends_on: [postgres]
    ports: ["3000:3000"]
    volumes: [./grafana/provisioning:/etc/grafana/provisioning]
    environment: [GF_SECURITY_ADMIN_PASSWORD, GF_SERVER_ROOT_URL]
```

---

## Implementation Steps

### Phase 0 — Foundation (Day 1)
1. venv + pyproject.toml
2. `.env.example`
3. `docker-compose.yml` (all 4 services)
4. `Dockerfile`
5. `scripts/init_db.sql` — run on postgres start
6. Grafana provisioning files (`/grafana/provisioning/datasources/`, `/dashboards/`)
7. `src/config.py`

### Phase 1 — Ingestion (Day 1-2)
8. `scripts/seed_tickers.py` — insert 15 tickers
9. `src/ingest/yfinance_ingest.py` — pull fundamentals, financials, prices, news
10. `src/ingest/document_chunker.py` — chunk docs into ~512-token pieces
11. `src/pipeline/run_ingestion.py` — orchestrate, handle retries + backoff

### Phase 2 — Embedding Pipeline (Day 2)
12. `src/vectorstore/chromadb_client.py` — ChromaDB collection CRUD
13. `src/retrieval/embed.py` — encode all chunks, upsert to ChromaDB
14. `src/retrieval/bm25_index.py` — build BM25 over chunk_text
15. `scripts/build_indexes.py` — run embedding + BM25 build

### Phase 3 — Retrieval (Day 2-3)
16. `src/retrieval/retriever.py` — hybrid search (RRF) + reranking pipeline
17. `src/retrieval/reranker.py` — cross-encoder rerank
18. `src/query_rewriting/rewriter.py` — LLM query decomposition

### Phase 4 — LLM Layer (Day 3-4)
19. `src/llm/client.py` — OpenAI API client with retries + mock fallback
20. `src/llm/prompts.py` — 3 prompt templates
21. `src/llm/evaluator.py` — RAGAS-style metrics

### Phase 5 — Interface (Day 4)
22. `src/ui/pages/research.py` — chat UI with ticker selector, response + citations
23. `src/ui/pages/ingestion.py` — ingestion status + manual trigger
24. `src/ui/pages/dashboard.py` — redirect link to Grafana
25. `src/ui/app.py` — sidebar nav + settings (model, prompt variant select)

### Phase 6 — Monitoring (Day 5)
26. `src/monitoring/feedback.py` — store thumbs/rating/comment
27. `src/monitoring/metrics.py` — query_log insert on each request
28. Grafana dashboard JSON provisioning
29. Import Grafana dashboard, configure PostgreSQL datasource

### Phase 7 — Evaluation (Day 5-6)
30. `notebooks/evaluation.ipynb` — hit-rate by approach, RAGAS scores by prompt variant
31. `src/evaluation/eval_runner.py` — script to run evaluation headless

### Phase 8 — Containerization + Final (Day 6-7)
32. Smoke test full docker-compose stack
33. `README.md` — screenshots, setup instructions, examples, architecture diagram

---

## DB Size Estimate (15 tickers, 1-year prices)

| Table | Est. Size |
|---|---|
| tickers (15 rows) | < 5 KB |
| financial_metrics (~15 × 8q × 12 cols) | ~200 KB |
| prices (15 × 252 days × 7 cols) | ~60 KB |
| documents (15 × 5yr × 105 docs) | ~100-200 MB |
| query_log + feedback | scales w/ usage |
| ChromaDB vectordb (10K chunks × 1.5 KB) | ~15 MB |
| **Total** | **~250-350 MB** |

---

## Key Files

| File | Est. Lines | Priority |
|---|---|---|
| `docker-compose.yml` | 55 | MUST-DO |
| `pyproject.toml` | 30 | MUST-DO |
| `.env.example` | 15 | MUST-DO |
| `Dockerfile` | 20 | MUST-DO |
| `scripts/init_db.sql` | 50 | MUST-DO |
| `grafana/provisioning/datasources/pg.yml` | 20 | MUST-DO |
| `grafana/provisioning/dashboards/dashboards.yml` | 10 | MUST-DO |
| `grafana/provisioning/dashboards/invest_agent.json` | 200 | MUST-DO |
| `src/config.py` | 50 | MUST-DO |
| `src/ingest/yfinance_ingest.py` | 80 | HIGH |
| `src/pipeline/run_ingestion.py` | 60 | HIGH |
| `src/vectorstore/chromadb_client.py` | 50 | HIGH |
| `src/retrieval/embed.py` | 50 | HIGH |
| `src/retrieval/bm25_index.py` | 50 | HIGH |
| `src/retrieval/retriever.py` | 80 | HIGH |
| `src/retrieval/reranker.py` | 40 | HIGH |
| `src/query_rewriting/rewriter.py` | 50 | HIGH |
| `src/llm/client.py` | 50 | HIGH |
| `src/llm/prompts.py` | 60 | HIGH |
| `src/llm/evaluator.py` | 60 | MED |
| `src/monitoring/feedback.py` | 30 | HIGH |
| `src/monitoring/metrics.py` | 40 | HIGH |
| `src/ui/pages/research.py` | 150 | HIGH |
| `src/ui/pages/dashboard.py` | 20 | MED |
| `src/ui/pages/ingestion.py` | 60 | MED |
| `src/ui/app.py` | 80 | HIGH |
| `scripts/build_indexes.py` | 40 | HIGH |
| `src/evaluation/eval_runner.py` | 60 | MED |
| `notebooks/evaluation.ipynb` | — | MED |
| `README.md` | 120 | MUST-DO |

---

## Verification Checklist

```
Grading Criterion       → Expected Score   → How Verified
─────────────────────────────────────────────────────────────────
Problem Description     → 2                → This plan (well-described)
Retrieval Flow         → 2                → ChromaDB + PostgreSQL dual store
Retrieval Evaluation   → 2                → Hit-rate@K on 3/4 approaches
LLM Evaluation        → 2                → RAGAS 4 metrics, 3 prompt variants
Interface              → 2                → Streamlit web app
Ingestion Pipeline    → 2                → Fully automated via cron
Monitoring             → 2                → Grafana (7 charts) + feedback capture
Containerization      → 2                → docker-compose (4 services)
Reproducibility       → 2                → README + setup instructions
─────────────────────────────────────────────────────────────────
Hybrid Search          → +1               → BM25 + ChromaDB RRF fusion
Re-ranking             → +1               → Cross-encoder top-5 rerank
Query Rewriting        → +1               → LLM-based decomposition
─────────────────────────────────────────────────────────────────
Total                  → 21 / 18 min      → All criteria met
```