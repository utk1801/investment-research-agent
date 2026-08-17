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
- **Cross-encoder reranking** — optional reranker for final hybrid candidates
- **RAG evaluation** — retrieval and generation benchmarks persisted to PostgreSQL
- **OpenAI** — answer synthesis (`gpt-4o`)
- **Apache Airflow** — scheduled ingestion and index rebuild orchestration
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

Sections include CEO/CFO prepared remarks, financial highlights, analyst Q&A, and forward outlook — rich narrative text for retrieval.

The configured ticker universe currently includes AAPL, MSFT, NVDA, JPM, GS, UNH, LLY, AMZN, WMT, XOM, CAT, and AMT. Static transcript files exist for the covered companies in `data/transcripts/`; a transcript for a new ticker is only ingested after that ticker is added to both app config and Postgres.

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
## Screenshots for Streamlit UI
RAG Agent's response when info is present in its context:
<img width="1000" alt="rag context" src="https://github.com/user-attachments/assets/75ed42c0-26d7-4f84-8255-1a8cb1c9c87a" />

RAG Agent's response when context doesn't contain the required info:
<img width="1000" alt="no-context-rag" src="https://github.com/user-attachments/assets/090453af-b7dc-4476-ab2d-ae332a926c63" />

### Other Screenshots
<table>
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/bb1e879d-9801-4131-9aff-1a68d3845e8e" width="600"><br>App UI</td>
    <td align="center"><img src="https://github.com/user-attachments/assets/3291ddb2-7232-47a2-9703-3a9c9e7a0c76" width="600"><br>Grafana</td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/9e69f894-800e-414f-aea5-c25ed261f1ff" width="600"><br>Airflow</td>
    <td align="center"><img src="https://github.com/user-attachments/assets/11a44cdc-e0ce-45dd-9188-f02eb41ef694" width="600"><br>Evaluation</td>
  </tr>
</table>

---
<img width="1461" height="788" alt="grafana-screenshot" src="" />

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
- **Airflow** on `localhost:8080` (airflow / airflow)

For an existing stack where the app image is already running, start only Airflow
without rebuilding the app container:

```bash
docker compose build airflow-init airflow-webserver airflow-scheduler
docker compose up -d --no-build airflow-init airflow-webserver airflow-scheduler
```

### 4. Automated ingestion with Airflow

Airflow runs the `investment_research_ingestion` DAG daily at 6am Pacific time
and can also be triggered manually from the Airflow UI.

Open Airflow:

- URL: `http://localhost:8080`
- Username: `airflow`
- Password: `airflow`

The DAG orchestrates:

1. `fetch_market_data`
2. `ingest_transcripts`
3. `backfill_financial_docs`
4. `rebuild_embeddings`
5. `rebuild_bm25`
6. `collect_stats`

Each Airflow task runs the existing ingestion command inside the `invest_app`
container, so the scheduled job uses the same code and dependencies as the
Streamlit application.

To run the flow manually:

1. Open `http://localhost:8080`
2. Log in with `airflow` / `airflow`
3. Open the `investment_research_ingestion` DAG
4. Click the trigger/play button
5. Check task logs from the DAG run graph if a step fails

The full DAG should be used after adding a new transcript because it loads the
JSON, regenerates financial docs, rebuilds Chroma embeddings, and rebuilds BM25.
If you only need one step, open the DAG graph and run or clear that task from the
Airflow UI.

To restart only Airflow services:

```bash
docker compose up -d --no-build airflow-init airflow-webserver airflow-scheduler
```

### 5. Manual fallback: ingest transcripts → ChromaDB → Build index

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
- **Retrieval**: `hybrid` (default), `hybrid_rerank`, `vector`, or `bm25`
- **Query rewriting**: expands ticker/company and financial concept terms before retrieval
- **Prompt variant**: A, B, or C
  - **A** — standard grounded answer
  - **B** — structured `[Findings] → [Data] → [Limitations]` with `[Source N]` citations
  - **C** — conservative; says "not available" instead of inferring

### Evaluation Page (localhost:8501/evaluation)

Run repeatable RAG benchmarks from the UI:

- **Retrieval evaluation** compares BM25, vector search, hybrid search, and hybrid+rerank using hit rate, precision@k, MRR, and nDCG@k.
- **Generation evaluation** compares prompt variants A, B, and C using fact coverage, groundedness, citation score, latency, tokens, and estimated cost.
- Results are stored in PostgreSQL tables: `evaluation_runs`, `retrieval_evaluation_results`, and `generation_evaluation_results`.

CLI equivalent:

```bash
uv run python -m src.evaluation.runner --retrieval-only
uv run python -m src.evaluation.runner
```

### Ingestion Page (localhost:8501/ingestion)

- View document counts per ticker
- **Run Ingestion** — pull latest data from yfinance (prices, financials, news) into PostgreSQL
- **Build Index** — re-embed all documents into ChromaDB and rebuild BM25
- **Backfill** — export quarterly financial metrics as searchable documents
- **Airflow link** — open scheduled orchestration for automated ingestion runs

`Backfill` does not fetch new market data. It converts existing rows in
`financial_metrics` into `documents` rows with `doc_type = financial`, then the
index must be rebuilt before those generated docs are searchable. The Airflow
DAG handles this rebuild automatically after the backfill step.

### Dashboard Page (localhost:8501/dashboard)

Live stats: total queries, feedback counts, average response times.

---

## Monitoring

Grafana dashboard at `localhost:3000` (admin / admin123) is provisioned automatically by Docker.
The `invest_agent_psql` datasource is recreated on startup from `grafana/provisioning/datasources/pg.yml`,
so you do not need to manually add or test the Postgres connection in the Grafana UI.

The dashboard shows:

- Queries over time
- Total LLM cost and token count for the selected time range
- Avg retrieval time vs total time (lower is faster)
- Thumbs Up / Down ratio (pie chart)
- Prompt variant usage (pie chart)
- Retrieval chunk count distribution
- Recent queries table

**Key PostgreSQL queries for Grafana:**

```sql
-- Queries over time
SELECT
  $__timeGroupAlias(created_at, '1m'),
  COUNT(*) AS queries
FROM query_log
WHERE $__timeFilter(created_at)
GROUP BY 1
ORDER BY 1

-- Thumbs ratio
SELECT 'thumbs_up' AS label, COUNT(*) AS value
FROM feedback
WHERE thumbs_up = true AND $__timeFilter(created_at)
UNION ALL
SELECT 'thumbs_down', COUNT(*)
FROM feedback
WHERE thumbs_down = true AND $__timeFilter(created_at)

-- Cost over time
SELECT
  $__timeGroupAlias(created_at, '1m'),
  SUM(total_cost_usd) AS cost_usd,
  SUM(total_tokens) AS tokens
FROM query_log
WHERE $__timeFilter(created_at)
GROUP BY 1
ORDER BY 1
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
├── query_rewriting/
│   └── rewriter.py           # Rule-based ticker/company/concept query expansion
├── evaluation/
│   ├── dataset.py            # Labeled RAG evaluation questions
│   ├── scoring.py            # Retrieval + generation metrics
│   ├── storage.py            # Evaluation tables + schema guards
│   └── runner.py             # CLI/UI evaluation runner
├── llm/
│   └── client.py             # OpenAI client + prompt variants (A/B/C)
├── monitoring/
│   └── metrics.py            # log_query(), log_feedback() → PostgreSQL
└── ui/
    ├── app.py                # Streamlit navigation shell
    └── pages/
        ├── research.py       # Main chat interface + feedback
        ├── evaluation.py     # RAG retrieval/generation evaluation runner
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

airflow/
└── dags/
    └── investment_ingestion_dag.py  # Scheduled ingestion DAG

Dockerfile.airflow           # Lightweight Airflow image with Docker CLI
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

If the ticker already exists in `src/config.py` and the `tickers` table, trigger
the `investment_research_ingestion` DAG in Airflow. The DAG will ingest the JSON
and rebuild both indexes.

Manual equivalent:

```bash
uv run python -m src.pipeline.run_ingestion --step transcripts
uv run python -m src.pipeline.run_ingestion --step embed --reset
uv run python -m src.pipeline.run_ingestion --step bm25
```

Transcript ingestion is idempotent: rerunning the Airflow DAG updates the
matching static transcript instead of inserting duplicate document rows.

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

The ticker symbol in the JSON must exist in:

- `src/config.py` under `TICKERS`
- the Postgres `tickers` table
- `scripts/init_db.sql` for fresh database setup reproducibility

For example, to add TSLA:

1. Add `TSLA` to the `TICKERS` dictionary in `src/config.py`.
2. Add `('TSLA', 'Tesla Inc.', 'Consumer')` to the seed list in `scripts/init_db.sql`.
3. Add TSLA to the already-running database:

```bash
docker exec invest_postgres psql -U invest_agent -d invest_agent -c "INSERT INTO tickers (symbol, name, sector) VALUES ('TSLA', 'Tesla Inc.', 'Consumer') ON CONFLICT (symbol) DO NOTHING;"
```

4. Restart the app container so the running Python process sees the updated ticker config:

```bash
docker compose restart app
```

5. Trigger `investment_research_ingestion` in Airflow.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required. OpenAI API key |
| `LLM_MODEL` | `gpt-4o` | LLM model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Optional cross-encoder reranker |
| `DEFAULT_RETRIEVAL_APPROACH` | `hybrid` | `bm25`, `vector`, `hybrid`, or `hybrid_rerank` |
| `ENABLE_QUERY_REWRITING` | `true` | Expand ticker/company/concept terms before retrieval |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `CHROMADB_HOST` | `localhost` | ChromaDB host |
| `CHROMADB_PORT` | `8000` | ChromaDB port |
| `APP_ENV` | `local` | `local` or `production` |
| `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` | — | Set by Docker Compose for Airflow metadata |
