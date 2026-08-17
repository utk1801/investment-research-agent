"""PostgreSQL persistence for RAG evaluation runs."""

from __future__ import annotations

from sqlalchemy import Engine, text


def ensure_runtime_schema(engine: Engine) -> None:
    """Create evaluation tables and add query-log columns used by the app."""
    statements = [
        """
        ALTER TABLE query_log
            ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER DEFAULT 0,
            ADD COLUMN IF NOT EXISTS completion_tokens INTEGER DEFAULT 0,
            ADD COLUMN IF NOT EXISTS total_tokens INTEGER DEFAULT 0,
            ADD COLUMN IF NOT EXISTS input_cost_usd NUMERIC(12, 6) DEFAULT 0,
            ADD COLUMN IF NOT EXISTS output_cost_usd NUMERIC(12, 6) DEFAULT 0,
            ADD COLUMN IF NOT EXISTS total_cost_usd NUMERIC(12, 6) DEFAULT 0
        """,
        """
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id SERIAL PRIMARY KEY,
            run_name TEXT NOT NULL,
            retrieval_approach VARCHAR(50),
            prompt_variants TEXT[],
            top_k INTEGER NOT NULL,
            notes TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS retrieval_evaluation_results (
            id SERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
            case_id TEXT NOT NULL,
            question TEXT NOT NULL,
            approach VARCHAR(50) NOT NULL,
            hit_rate FLOAT,
            precision_at_k FLOAT,
            mrr FLOAT,
            ndcg_at_k FLOAT,
            relevant_count INTEGER,
            top_score FLOAT,
            latency_ms INTEGER,
            retrieved_doc_ids JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS generation_evaluation_results (
            id SERIAL PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
            case_id TEXT NOT NULL,
            question TEXT NOT NULL,
            retrieval_approach VARCHAR(50) NOT NULL,
            prompt_variant VARCHAR(10) NOT NULL,
            fact_coverage FLOAT,
            groundedness FLOAT,
            citation_score FLOAT,
            overall_score FLOAT,
            latency_ms INTEGER,
            total_tokens INTEGER,
            total_cost_usd NUMERIC(12, 6),
            answer_preview TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_retrieval_eval_run
            ON retrieval_evaluation_results(run_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_generation_eval_run
            ON generation_evaluation_results(run_id)
        """,
    ]
    with engine.connect() as conn:
        for sql in statements:
            conn.execute(text(sql))
        conn.commit()
