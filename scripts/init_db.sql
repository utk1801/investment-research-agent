-- ── Investment Research Agent — PostgreSQL Schema ──────────────────────────

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── 1. Tickers ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tickers (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) UNIQUE NOT NULL,
    name TEXT,
    sector TEXT,
    inserted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ── 2. Financial Snapshots ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS financial_metrics (
    id SERIAL PRIMARY KEY,
    ticker_id INTEGER NOT NULL REFERENCES tickers(id) ON DELETE CASCADE,
    period DATE NOT NULL,
    revenue BIGINT,
    net_income BIGINT,
    free_cash_flow BIGINT,
    total_debt BIGINT,
    total_equity BIGINT,
    pe_ratio FLOAT,
    pb_ratio FLOAT,
    dividend_yield FLOAT,
    revenue_growth FLOAT,
    market_cap BIGINT,
    inserted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(ticker_id, period)
);

-- ── 3. Daily Price Data ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS prices (
    id SERIAL PRIMARY KEY,
    ticker_id INTEGER NOT NULL REFERENCES tickers(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    open_price FLOAT,
    high_price FLOAT,
    low_price FLOAT,
    close_price FLOAT,
    volume BIGINT,
    UNIQUE(ticker_id, date)
);

-- ── 4. Raw Documents ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    ticker_id INTEGER NOT NULL REFERENCES tickers(id) ON DELETE CASCADE,
    doc_type VARCHAR(20) NOT NULL CHECK (doc_type IN ('earnings_call', 'news')),
    title TEXT,
    content TEXT,
    doc_date DATE,
    source VARCHAR(20) DEFAULT 'yfinance',
    inserted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ── 5. Query Log ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_log (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    ticker_filter VARCHAR(10),
    rewritten_query TEXT,
    retrieval_chunk_count INTEGER,
    retrieval_time_ms INTEGER,
    total_time_ms INTEGER,
    model_used VARCHAR(50),
    prompt_variant VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ── 6. User Feedback ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    query_id INTEGER NOT NULL REFERENCES query_log(id) ON DELETE CASCADE,
    thumbs_up BOOLEAN,
    thumbs_down BOOLEAN,
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    comment TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ── Indexes ──────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_financial_metrics_ticker_period
    ON financial_metrics(ticker_id, period DESC);
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
    ON prices(ticker_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_documents_ticker_type
    ON documents(ticker_id, doc_type, doc_date DESC);
CREATE INDEX IF NOT EXISTS idx_query_log_created
    ON query_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_query
    ON feedback(query_id);

-- ── Seed Tickers ────────────────────────────────────────────────────────────
INSERT INTO tickers (symbol, name, sector) VALUES
    ('AAPL',  'Apple Inc.',              'Technology'),
    ('MSFT',  'Microsoft Corp.',         'Technology'),
    ('NVDA',  'NVIDIA Corp.',            'Technology'),
    ('JPM',   'JPMorgan Chase & Co.',    'Finance'),
    ('GS',    'Goldman Sachs Group',      'Finance'),
    ('UNH',   'UnitedHealth Group Inc.', 'Healthcare'),
    ('LLY',   'Eli Lilly and Co.',       'Healthcare'),
    ('AMZN',  'Amazon.com Inc.',         'Consumer'),
    ('WMT',   'Walmart Inc.',            'Consumer'),
    ('XOM',   'Exxon Mobil Corp.',       'Energy'),
    ('CAT',   'Caterpillar Inc.',        'Industrials'),
    ('AMT',   'American Tower Corp.',    'Real Estate')
ON CONFLICT (symbol) DO NOTHING;