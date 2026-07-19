"""Monitoring: user feedback capture."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from src.config import get_db_url


class FeedbackLogger:
    """Stores user feedback on LLM responses."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or get_db_url()
        self._engine = create_engine(self.db_url, poolclass=NullPool)

    def log_feedback(
        self,
        query_id: int,
        thumbs_up: Optional[bool] = None,
        thumbs_down: Optional[bool] = None,
        rating: Optional[int] = None,
        comment: Optional[str] = None,
    ) -> bool:
        """
        Store user feedback for a query.

        Args:
            query_id: FK to query_log.id
            thumbs_up: True if user liked the response
            thumbs_down: True if user disliked the response
            rating: 1-5 star rating
            comment: Free-text optional comment
        """
        if thumbs_up and thumbs_down:
            thumbs_up = False  # can't be both

        sql = text("""
            INSERT INTO feedback
                (query_id, thumbs_up, thumbs_down, rating, comment)
            VALUES
                (:query_id, :thumbs_up, :thumbs_down, :rating, :comment)
        """)
        with self._engine.connect() as conn:
            conn.execute(sql, {
                "query_id": query_id,
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "rating": rating,
                "comment": comment,
            })
            conn.commit()
        return True

    def close(self):
        self._engine.dispose()