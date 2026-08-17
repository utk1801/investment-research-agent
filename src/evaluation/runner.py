"""Run retrieval and generation evaluations for the RAG application."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime
from statistics import mean

from sqlalchemy import create_engine, text

from src.config import DEFAULT_RETRIEVAL_APPROACH, RETRIEVAL_TOP_K_FINAL, get_db_url
from src.evaluation.dataset import DEFAULT_EVAL_CASES, EvalCase
from src.evaluation.scoring import generation_metrics, retrieval_metrics
from src.evaluation.storage import ensure_runtime_schema
from src.llm.client import LLMClient
from src.retrieval.retriever import HybridRetriever

log = logging.getLogger(__name__)


class EvaluationRunner:
    def __init__(self, db_url: str | None = None):
        self.engine = create_engine(db_url or get_db_url())
        ensure_runtime_schema(self.engine)

    def run(
        self,
        cases: list[EvalCase] | None = None,
        retrieval_approaches: list[str] | None = None,
        prompt_variants: list[str] | None = None,
        top_k: int = RETRIEVAL_TOP_K_FINAL,
        include_generation: bool = True,
        model: str | None = None,
        run_name: str | None = None,
    ) -> dict:
        cases = cases or DEFAULT_EVAL_CASES
        retrieval_approaches = retrieval_approaches or ["bm25", "vector", "hybrid", "hybrid_rerank"]
        prompt_variants = prompt_variants or ["A", "B", "C"]
        run_name = run_name or f"rag-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        run_id = self._create_run(run_name, DEFAULT_RETRIEVAL_APPROACH, prompt_variants, top_k)
        retrieval_rows = self._run_retrieval(run_id, cases, retrieval_approaches, top_k)

        generation_rows: list[dict] = []
        best_retrieval = self._best_retrieval_approach(retrieval_rows) or DEFAULT_RETRIEVAL_APPROACH
        if include_generation:
            generation_rows = self._run_generation(
                run_id, cases, best_retrieval, prompt_variants, top_k, model
            )

        return {
            "run_id": run_id,
            "run_name": run_name,
            "best_retrieval_approach": best_retrieval,
            "best_prompt_variant": self._best_prompt_variant(generation_rows),
            "retrieval_summary": self._summarize(retrieval_rows, "approach"),
            "generation_summary": self._summarize(generation_rows, "prompt_variant"),
        }

    def _create_run(
        self,
        run_name: str,
        retrieval_approach: str,
        prompt_variants: list[str],
        top_k: int,
    ) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO evaluation_runs
                    (run_name, retrieval_approach, prompt_variants, top_k, notes)
                VALUES
                    (:run_name, :retrieval_approach, :prompt_variants, :top_k, :notes)
                RETURNING id
            """), {
                "run_name": run_name,
                "retrieval_approach": retrieval_approach,
                "prompt_variants": prompt_variants,
                "top_k": top_k,
                "notes": "Retrieval variants plus optional generation prompt benchmark.",
            })
            conn.commit()
            return result.scalar_one()

    def _run_retrieval(
        self,
        run_id: int,
        cases: list[EvalCase],
        approaches: list[str],
        top_k: int,
    ) -> list[dict]:
        rows: list[dict] = []
        for case in cases:
            retriever = HybridRetriever(ticker_filter=case.ticker_filter)
            for approach in approaches:
                start = time.time()
                docs = retriever.retrieve(
                    case.question,
                    top_k=top_k,
                    approach=approach,
                    rewrite=True,
                )
                latency_ms = int((time.time() - start) * 1000)
                metrics = retrieval_metrics(case, docs, k=top_k)
                row = {
                    "run_id": run_id,
                    "case_id": case.id,
                    "question": case.question,
                    "approach": approach,
                    "latency_ms": latency_ms,
                    "retrieved_doc_ids": [
                        {
                            "doc_id": d.get("doc_id"),
                            "title": d.get("title"),
                            "doc_type": d.get("doc_type"),
                            "score": d.get("_score"),
                        }
                        for d in docs
                    ],
                    **metrics,
                }
                self._insert_retrieval_row(row)
                rows.append(row)
        return rows

    def _run_generation(
        self,
        run_id: int,
        cases: list[EvalCase],
        retrieval_approach: str,
        prompt_variants: list[str],
        top_k: int,
        model: str | None,
    ) -> list[dict]:
        rows: list[dict] = []
        llm = LLMClient(model=model)
        for case in cases:
            retriever = HybridRetriever(ticker_filter=case.ticker_filter)
            docs = retriever.retrieve(
                case.question,
                top_k=top_k,
                approach=retrieval_approach,
                rewrite=True,
            )
            for variant in prompt_variants:
                start = time.time()
                response = llm.generate(case.question, docs, variant=variant)
                latency_ms = int((time.time() - start) * 1000)
                metrics = generation_metrics(case, response.answer, docs)
                row = {
                    "run_id": run_id,
                    "case_id": case.id,
                    "question": case.question,
                    "retrieval_approach": retrieval_approach,
                    "prompt_variant": variant,
                    "latency_ms": latency_ms,
                    "total_tokens": response.total_tokens,
                    "total_cost_usd": response.total_cost_usd,
                    "answer_preview": response.answer[:500],
                    **metrics,
                }
                self._insert_generation_row(row)
                rows.append(row)
        return rows

    def _insert_retrieval_row(self, row: dict) -> None:
        payload = dict(row)
        payload["retrieved_doc_ids"] = json.dumps(payload["retrieved_doc_ids"])
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO retrieval_evaluation_results
                    (run_id, case_id, question, approach, hit_rate, precision_at_k,
                     mrr, ndcg_at_k, relevant_count, top_score, latency_ms,
                     retrieved_doc_ids)
                VALUES
                    (:run_id, :case_id, :question, :approach, :hit_rate,
                     :precision_at_k, :mrr, :ndcg_at_k, :relevant_count,
                     :top_score, :latency_ms, CAST(:retrieved_doc_ids AS JSONB))
            """), payload)
            conn.commit()

    def _insert_generation_row(self, row: dict) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO generation_evaluation_results
                    (run_id, case_id, question, retrieval_approach, prompt_variant,
                     fact_coverage, groundedness, citation_score, overall_score,
                     latency_ms, total_tokens, total_cost_usd, answer_preview)
                VALUES
                    (:run_id, :case_id, :question, :retrieval_approach,
                     :prompt_variant, :fact_coverage, :groundedness,
                     :citation_score, :overall_score, :latency_ms,
                     :total_tokens, :total_cost_usd, :answer_preview)
            """), row)
            conn.commit()

    @staticmethod
    def _summarize(rows: list[dict], group_key: str) -> list[dict]:
        if not rows:
            return []
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            grouped.setdefault(row[group_key], []).append(row)
        summary = []
        for group, group_rows in grouped.items():
            metric_names = [
                key for key, value in group_rows[0].items()
                if isinstance(value, (int, float)) and key not in {"run_id", "latency_ms", "total_tokens"}
            ]
            item = {group_key: group, "cases": len(group_rows)}
            for metric in metric_names:
                item[metric] = mean(float(row[metric] or 0) for row in group_rows)
            item["avg_latency_ms"] = mean(float(row.get("latency_ms") or 0) for row in group_rows)
            summary.append(item)
        return sorted(summary, key=lambda r: r.get("overall_score", r.get("mrr", 0)), reverse=True)

    @staticmethod
    def _best_retrieval_approach(rows: list[dict]) -> str | None:
        summary = EvaluationRunner._summarize(rows, "approach")
        if not summary:
            return None
        return max(summary, key=lambda r: (r.get("mrr", 0), r.get("ndcg_at_k", 0)))["approach"]

    @staticmethod
    def _best_prompt_variant(rows: list[dict]) -> str | None:
        summary = EvaluationRunner._summarize(rows, "prompt_variant")
        if not summary:
            return None
        return max(summary, key=lambda r: r.get("overall_score", 0))["prompt_variant"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run RAG retrieval and generation evaluations")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM generation scoring")
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K_FINAL)
    parser.add_argument("--model", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    result = EvaluationRunner().run(
        include_generation=not args.retrieval_only,
        top_k=args.top_k,
        model=args.model,
        run_name=args.run_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
