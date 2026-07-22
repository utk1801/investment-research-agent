# Investment Research Agent

A Streamlit web app that answers financial questions by retrieving relevant earnings call content and synthesizing answers with an LLM. Built with hybrid retrieval (BM25 + vector search), PostgreSQL, ChromaDB, and OpenAI.

---

## The Problem

If you want to research what NVIDIA's CEO said about AI demand, or how JPMorgan's management views credit risk, you have to:

1. Find and open earnings call transcripts
2. Scroll through hundreds of pages
3. Extract relevant quotes manually
4. Synthesize findings yourself

This takes 30–60 minutes per question. The agent automates the retrieval and synthesis step, returning a grounded answer in seconds — with citations pointing back to the source documents.

---

## Architecture

```
 Earnings Call JSON files
          │
          ▼
  ┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
  │  Transcript     │     │  PostgreSQL  │     │  ChromaDB   │
  │  Ingestor       │────▶│  documents   │────▶│  + BM25     │
  └─────────────────┘     └──────────────┘     └──────────────┘
                                                         │
                                                         ▼
User question ──▶ Hybrid Retriever ◀────────────────────┘
                      │
                      ▼
               Top-k chunks
                      │
                      ▼
               LLM (OpenAI GPT-4o)
                      │
                      ▼
              Answer + Sources + Tokens
                      │
                      ▼
            PostgreSQL (query_log + feedback)
                      │
                      ▼
                  Grafana
```

**Stack:**
- **Streamlit** — web UI
- **PostgreSQL** — document store + query/feedback logs
- **ChromaDB** — vector embeddings (sentence-transformers: `all-MiniLM-L6-v2`)
- **BM25** — keyword search (rank_bm25)
- **OpenAI** — answer synthesis (`gpt-4o`)
- **Grafana** — metrics dashboards

---

## Data

### Earnings Call Transcripts (Static)

JSON files in `data/transcripts/`, one per ticker per quarter:

```json
{
  "title": "Apple Q1 2025 Earnings Call Transcript",
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "quarter": "Q1 2025",
  "call_date": "2025-02-01",
  "sections": {
    "prepared_remarks": "Apple reporting record Q1 revenue of $124.3 billion...",
    "financial_highlights": "EPS $2.41, gross margin 46.2 percent...",
    "qanda_session": "Analyst: Tim, on Apple Intelligence...\nTim Cook: Samik...",
    "forward_outlook": "Tim Cook: Looking ahead to Q2, we are optimistic..."
  }
}
```

Sections include CEO/CFO prepared remarks, financial highlights, analyst Q&A, and forward outlook — rich narrative text for retrieval. Currently covers: AAPL, MSFT, JPM, NVDA, AMZN, GS (10 transcript files).

### Tickers (12 companies)

| Symbol | Name | Sector |
|--------|------|--------|
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corp. | Technology |
| NVDA | NVIDIA Corp. | Technology |
| JPM | JPMorgan Chase & Co. | Finance |
| GS | Goldman Sachs Group | Finance |
| UNH | UnitedHealth Group | Healthcare |
| LLY | Eli Lilly | Healthcare |
| AMZN | Amazon.com Inc. | Consumer |
| WMT | Walmart Inc. | Consumer |
| XOM | Exxon Mobil Corp. | Energy |
| CAT | Caterpillar Inc. | Industrials |
| AMT | American Tower Corp. | Real Estate |

---

## Setup

### 1. Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (or pip/poetry)
- Docker & Docker Compose
- OpenAI API key

### 2. Environment

```bash
cp .env.example .env
# Add your OpenAI key:
# OPENAI_API_KEY=sk-...
```

### 3. Start services

```bash
docker compose up -d --build
```

This starts:
- **PostgreSQL** on `localhost:5432`
- **ChromaDB** on `localhost:8000`
- **Streamlit app** on `localhost:8501`
- **Grafana** on `localhost:3000` (admin / admin123)

### 4. Ingest transcripts → ChromaDB → Build index

```bash
uv run python -m src.pipeline.run_ingestion --step transcripts
uv run python -m src.pipeline.run_ingestion --step embed --reset
uv run python -m src.pipeline.run_ingestion --step bm25
```

Or run all in one step (fetches yfinance data + transcripts + rebuilds index):

```bash
uv run python -m src.pipeline.run_ingestion --step all
```

Check counts:

```bash
uv run python -m src.pipeline.run_ingestion --step stats
```

---

## Usage

### Research Page (localhost:8501)

Ask questions about CEO commentary, financial guidance, AI strategy, credit risk, or M&A outlook across the tracked companies.

**Example questions:**

```
What did Jensen Huang say about AI infrastructure demand?
How does Jamie Dimon view commercial real estate credit risk?
What is Satya Nadella's outlook on Microsoft's AI monetization?
What is Microsoft guidance for Azure growth?
Did Tim Cook mention anything about Apple Intelligence in China?
How does Goldman Sachs CEO view the 2026 deal pipeline?
```

**Features:**
- Chat history — past queries and answers persist in session
- Source citations — expand "📚 Sources" to see chunk excerpts and their original documents
- Ticker filter — sidebar selector limits retrieval to a single company
- Token & cost tracking — each answer shows token count and estimated USD cost
- Debug panel — "🔧 DEBUG: Retrieval chunks" shows raw retrieved documents

### Model & Prompt Variants

Configure in the Research page sidebar:
- **Model**: `gpt-4o` (default), `gpt-4o-mini`, `gpt-3.5-turbo`
- **Prompt variant**: A, B, or C
  - **A** — standard grounded answer
  - **B** — structured `[Findings] → [Data] → [Limitations]` with `[Source N]` citations
  - **C** — conservative; says "not available" instead of inferring

### Ingestion Page (localhost:8501/ingestion)

- View document counts per ticker
- **Run Ingestion** — pull latest data from yfinance (prices, financials, news) into PostgreSQL
- **Build Index** — re-embed all documents into ChromaDB and rebuild BM25
- **Backfill** — export quarterly financial metrics as searchable documents

### Dashboard Page (localhost:8501/dashboard)

Live stats: total queries, feedback counts, average response times.

---

## Monitoring

Grafana dashboard at `localhost:3000` (admin / admin123) shows:

- Queries per day (time series)
- Total LLM cost and token count (7-day rolling)
- Avg retrieval time vs total time (lower is faster)
- Thumbs Up / Down ratio (pie chart)
- Rating distribution (if using star ratings)
- Cumulative cost by model over time

**Key PostgreSQL queries for Grafana:**

```sql
-- Queries per day
SELECT DATE(created_at) AS time, COUNT(*) AS queries
FROM query_log WHERE $__timeFilter(created_at)
GROUP BY DATE(created_at) ORDER BY time

-- Thumbs ratio
SELECT 'thumbs_up' AS label, COUNT(*) AS value
FROM feedback WHERE thumbs_up = true
UNION ALL
SELECT 'thumbs_down', COUNT(*) FROM feedback WHERE thumbs_down = true

-- Cost over time
SELECT DATE(created_at) AS time,
  SUM(total_cost_usd) AS cost_usd,
  SUM(total_tokens) AS tokens
FROM query_log
WHERE $__timeFilter(created_at)
GROUP BY DATE(created_at) ORDER BY time
```

---

## Project Structure

```
src/
├── config.py                 # All env vars, paths, ticker list, constants
├── pipeline/
│   └── run_ingestion.py     # --step fetch|transcripts|embed|bm25|stats
├── ingest/
│   ├── earnings_transcript_ingest.py   # JSON → documents table
│   ├── yfinance_ingest.py               # yfinance → prices/financials/news
│   └── document_chunker.py              # ~512-token chunks for vector store
├── retrieval/
│   ├── retriever.py          # Hybrid retriever (BM25 + ChromaDB, weighted blend)
│   ├── embed.py              # Sentence-transformer embedder
│   └── bm25_index.py         # Rank BM25 index
├── llm/
│   └── client.py             # OpenAI client + prompt variants (A/B/C)
├── monitoring/
│   └── metrics.py            # log_query(), log_feedback() → PostgreSQL
├── query_rewriting/          # (optional) query expansion hooks
└── ui/
    ├── app.py                # Streamlit navigation shell
    └── pages/
        ├── research.py       # Main chat interface + feedback
        ├── ingestion.py      # Ingestion controls + ticker stats table
        └── dashboard.py      # Live metrics from query_log

data/
├── chroma_db/                # ChromaDB persistent storage
├── bm25_index.pkl            # BM25 ranked index file
└── transcripts/              # Static JSON transcript files (*.json)

grafana/provisioning/
├── dashboards/invest_agent.json   # Full dashboard definition
└── datasources/pg.yml            # PostgreSQL datasource config

scripts/
└── init_db.sql              # Schema: tickers, financial_metrics, prices,
                             # documents, query_log, feedback
```

---

## Adding More Transcripts

Drop a new JSON file into `data/transcripts/` following the schema above:

```
data/transcripts/
├── AAPL_Q1_2025.json
├── AAPL_Q2_2025.json
├── MSFT_Q2_FY2025.json
...
└── NEW_TICKER_Q1_2025.json   # ← add here
```

Then re-ingest and rebuild the index:

```bash
uv run python -m src.pipeline.run_ingestion --step transcripts
uv run python -m src.pipeline.run_ingestion --step embed --reset
uv run python -m src.pipeline.run_ingestion --step bm25
```

**Required JSON fields:**
```json
{
  "title": "string",
  "ticker": "SYMBOL",       // must match a ticker in tickers table
  "company_name": "string",
  "quarter": "Q1 2025",
  "call_date": "YYYY-MM-DD",
  "sections": {
    "prepared_remarks": "string",
    "financial_highlights": "string",
    "qanda_session": "string",
    "forward_outlook": "string"
  }
}
```

The ticker symbol in the JSON must exist in `scripts/init_db.sql` (or must be added to the `INSERT INTO tickers` seed list there).

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required. OpenAI API key |
| `LLM_MODEL` | `gpt-4o` | LLM model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `CHROMADB_HOST` | `localhost` | ChromaDB host |
| `CHROMADB_PORT` | `8000` | ChromaDB port |
| `APP_ENV` | `local` | `local` or `production` |

---

## Monitoring Dashboard:
Screenshot for Grafana running locally, showing metrics for our Investment RAG agent:
<img width="1441" height="771" alt="image" src="https://github.com/user-attachments/assets/e420a9e5-e030-4e45-9db6-a7c323aa3f5c" />
