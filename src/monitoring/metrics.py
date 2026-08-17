"""Monitoring: metrics logging to PostgreSQL + Grafana dashboard JSON."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from src.config import get_db_url
from src.evaluation.storage import ensure_runtime_schema


@dataclass
class QueryMetrics:
    """Metrics captured per query for Grafana dashboards."""
    query_text: str
    ticker_filter: Optional[str]
    rewritten_query: Optional[str]
    retrieval_chunk_count: int
    retrieval_time_ms: int
    total_time_ms: int
    model_used: str
    prompt_variant: str
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Cost (USD)
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    total_cost_usd: float = 0.0


class MetricsLogger:
    """Logs query metrics and user feedback to PostgreSQL."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or get_db_url()
        self._engine = create_engine(self.db_url, poolclass=NullPool)
        ensure_runtime_schema(self._engine)

    def log_query(self, metrics: QueryMetrics) -> int:
        """Insert a query_log row. Returns the query_id."""
        sql = text("""
            INSERT INTO query_log
                (query_text, ticker_filter, rewritten_query,
                 retrieval_chunk_count, retrieval_time_ms,
                 total_time_ms, model_used, prompt_variant,
                 prompt_tokens, completion_tokens, total_tokens,
                 input_cost_usd, output_cost_usd, total_cost_usd)
            VALUES
                (:query_text, :ticker_filter, :rewritten_query,
                 :retrieval_chunk_count, :retrieval_time_ms,
                 :total_time_ms, :model_used, :prompt_variant,
                 :prompt_tokens, :completion_tokens, :total_tokens,
                 :input_cost_usd, :output_cost_usd, :total_cost_usd)
            RETURNING id
        """)
        with self._engine.connect() as conn:
            result = conn.execute(sql, {
                "query_text": metrics.query_text,
                "ticker_filter": metrics.ticker_filter,
                "rewritten_query": metrics.rewritten_query,
                "retrieval_chunk_count": metrics.retrieval_chunk_count,
                "retrieval_time_ms": metrics.retrieval_time_ms,
                "total_time_ms": metrics.total_time_ms,
                "model_used": metrics.model_used,
                "prompt_variant": metrics.prompt_variant,
                "prompt_tokens": metrics.prompt_tokens,
                "completion_tokens": metrics.completion_tokens,
                "total_tokens": metrics.total_tokens,
                "input_cost_usd": metrics.input_cost_usd,
                "output_cost_usd": metrics.output_cost_usd,
                "total_cost_usd": metrics.total_cost_usd,
            })
            conn.commit()
            return result.scalar_one()

    def log_feedback(
        self,
        query_id: int,
        thumbs_up: Optional[bool] = None,
        thumbs_down: Optional[bool] = None,
        rating: Optional[int] = None,
        comment: Optional[str] = None,
    ) -> None:
        """Insert a feedback row, or UPDATE comment on an existing row for query_id."""
        with self._engine.connect() as conn:
            if comment:
                # Find the existing thumbs row for this query and append the comment
                result = conn.execute(text("""
                    UPDATE feedback
                    SET comment = :comment
                    WHERE query_id = :query_id
                      AND comment IS NULL
                """), {"comment": comment, "query_id": query_id})
                conn.commit()
                logging.getLogger(__name__).warning(
                    "UPDATE feedback comment rows_matched=%s, query_id=%s, comment='%s'",
                    result.rowcount, query_id, comment[:50]
                )
            else:
                conn.execute(text("""
                    INSERT INTO feedback
                        (query_id, thumbs_up, thumbs_down, rating, comment)
                    VALUES
                        (:query_id, :thumbs_up, :thumbs_down, :rating, :comment)
                """), {
                    "query_id": query_id,
                    "thumbs_up": thumbs_up,
                    "thumbs_down": thumbs_down,
                    "rating": rating,
                    "comment": None,
                })
                conn.commit()

    def get_query_stats(self, days: int = 7) -> dict:
        """Return aggregate stats for the last N days."""
        sql = text("""
            SELECT
                COUNT(*) as total_queries,
                AVG(total_time_ms)::int as avg_total_ms,
                AVG(retrieval_time_ms)::int as avg_retrieval_ms,
                COUNT(CASE WHEN prompt_variant = 'A' THEN 1 END) as variant_a,
                COUNT(CASE WHEN prompt_variant = 'B' THEN 1 END) as variant_b,
                COUNT(CASE WHEN prompt_variant = 'C' THEN 1 END) as variant_c
            FROM query_log
            WHERE created_at >= NOW() - INTERVAL ':days days'
        """)
        with self._engine.connect() as conn:
            result = conn.execute(sql, {"days": days})
            row = result.fetchone()
            if row:
                return {
                    "total_queries": row[0],
                    "avg_total_ms": row[1] or 0,
                    "avg_retrieval_ms": row[2] or 0,
                    "variant_a": row[3] or 0,
                    "variant_b": row[4] or 0,
                    "variant_c": row[5] or 0,
                }
            return {}

    def close(self):
        self._engine.dispose()


# ── Simple function API ──────────────────────────────────────────────────────
_metrics_logger: MetricsLogger | None = None


def _logger() -> MetricsLogger:
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = MetricsLogger()
    return _metrics_logger


def log_query(
    query_text: str,
    ticker_filter: str | None,
    retrieval_chunk_count: int,
    retrieval_time_ms: int,
    model: str,
    prompt_variant: str,
    rewritten_query: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    input_cost_usd: float = 0.0,
    output_cost_usd: float = 0.0,
    total_cost_usd: float = 0.0,
) -> int | None:
    """Insert a row into query_log. Returns the inserted query_id."""
    try:
        metrics = QueryMetrics(
            query_text=query_text,
            ticker_filter=ticker_filter,
            rewritten_query=rewritten_query,
            retrieval_chunk_count=retrieval_chunk_count,
            retrieval_time_ms=retrieval_time_ms,
            total_time_ms=retrieval_time_ms,
            model_used=model,
            prompt_variant=prompt_variant,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            input_cost_usd=input_cost_usd,
            output_cost_usd=output_cost_usd,
            total_cost_usd=total_cost_usd,
        )
        return _logger().log_query(metrics)
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to log query: %s", exc)
        return None
