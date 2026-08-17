"""RAG evaluation page — retrieval and generation benchmark controls."""

from __future__ import annotations

import streamlit as st
from sqlalchemy import create_engine, text

from src.config import get_db_url
from src.evaluation.dataset import DEFAULT_EVAL_CASES
from src.evaluation.runner import EvaluationRunner
from src.evaluation.storage import ensure_runtime_schema
from src.ui.app import render_sidebar


def _engine():
    engine = create_engine(get_db_url())
    ensure_runtime_schema(engine)
    return engine


def _latest_runs(limit: int = 10) -> list[dict]:
    engine = _engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, run_name, retrieval_approach, top_k, created_at
            FROM evaluation_runs
            ORDER BY created_at DESC
            LIMIT :limit
        """), {"limit": limit}).mappings().all()
    return [dict(row) for row in rows]


def _retrieval_summary(run_id: int) -> list[dict]:
    engine = _engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT approach,
                   COUNT(*) AS cases,
                   AVG(hit_rate) AS hit_rate,
                   AVG(precision_at_k) AS precision_at_k,
                   AVG(mrr) AS mrr,
                   AVG(ndcg_at_k) AS ndcg_at_k,
                   AVG(latency_ms)::int AS avg_latency_ms
            FROM retrieval_evaluation_results
            WHERE run_id = :run_id
            GROUP BY approach
            ORDER BY mrr DESC, ndcg_at_k DESC
        """), {"run_id": run_id}).mappings().all()
    return [dict(row) for row in rows]


def _generation_summary(run_id: int) -> list[dict]:
    engine = _engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT prompt_variant,
                   COUNT(*) AS cases,
                   AVG(fact_coverage) AS fact_coverage,
                   AVG(groundedness) AS groundedness,
                   AVG(citation_score) AS citation_score,
                   AVG(overall_score) AS overall_score,
                   AVG(latency_ms)::int AS avg_latency_ms,
                   SUM(total_tokens) AS total_tokens,
                   SUM(total_cost_usd) AS total_cost_usd
            FROM generation_evaluation_results
            WHERE run_id = :run_id
            GROUP BY prompt_variant
            ORDER BY overall_score DESC
        """), {"run_id": run_id}).mappings().all()
    return [dict(row) for row in rows]


def render_evaluation():
    prefs = st.session_state.get("prefs", {})
    model = prefs.get("model", "gpt-4o")

    st.title("🧪 RAG evaluation")
    st.caption("Benchmark retrieval approaches and prompt variants against a labeled earnings-call evaluation set.")

    with st.container(border=True):
        st.subheader("Evaluation set")
        st.write(f"{len(DEFAULT_EVAL_CASES)} questions covering AI demand, Azure growth, Apple Intelligence, credit risk, deal pipeline, and metric-style retrieval.")
        st.dataframe(
            [
                {
                    "Case": case.id,
                    "Question": case.question,
                    "Ticker filter": case.ticker_filter or "All",
                    "Expected docs": ", ".join(case.expected_doc_types),
                }
                for case in DEFAULT_EVAL_CASES
            ],
            hide_index=True,
        )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Retrieval evaluation")
            st.write("Scores BM25, vector, hybrid, and hybrid+rerank with hit rate, precision@k, MRR, and nDCG@k.")
            if st.button("Run retrieval evaluation", icon=":material/search:"):
                with st.spinner("Running retrieval benchmark..."):
                    result = EvaluationRunner().run(include_generation=False)
                st.success(f"Run {result['run_id']} complete. Best retrieval: {result['best_retrieval_approach']}")
                st.rerun()

    with col2:
        with st.container(border=True):
            st.subheader("Generation evaluation")
            st.write("Runs prompt variants A, B, and C using the best retrieval approach, then scores coverage, groundedness, citations, tokens, and cost.")
            if st.button("Run full RAG evaluation", icon=":material/psychology:", type="primary"):
                with st.spinner("Running retrieval and LLM benchmark..."):
                    result = EvaluationRunner().run(include_generation=True, model=model)
                st.success(
                    f"Run {result['run_id']} complete. Best retrieval: "
                    f"{result['best_retrieval_approach']}; best prompt: {result['best_prompt_variant']}"
                )
                st.rerun()

    st.divider()
    st.subheader("Recent runs")
    runs = _latest_runs()
    if not runs:
        st.info("No evaluation runs yet.")
        return

    st.dataframe(runs, hide_index=True)
    selected_run = st.selectbox("Inspect run", options=[run["id"] for run in runs])

    retrieval = _retrieval_summary(selected_run)
    if retrieval:
        st.markdown("**Retrieval summary**")
        st.dataframe(
            retrieval,
            hide_index=True,
            column_config={
                "hit_rate": st.column_config.NumberColumn(format="%.2f"),
                "precision_at_k": st.column_config.NumberColumn(format="%.2f"),
                "mrr": st.column_config.NumberColumn(format="%.2f"),
                "ndcg_at_k": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    generation = _generation_summary(selected_run)
    if generation:
        st.markdown("**Generation summary**")
        st.dataframe(
            generation,
            hide_index=True,
            column_config={
                "fact_coverage": st.column_config.NumberColumn(format="%.2f"),
                "groundedness": st.column_config.NumberColumn(format="%.2f"),
                "citation_score": st.column_config.NumberColumn(format="%.2f"),
                "overall_score": st.column_config.NumberColumn(format="%.2f"),
                "total_cost_usd": st.column_config.NumberColumn(format="$%.6f"),
            },
        )


if __name__ == "__main__":
    prefs = render_sidebar()
    st.session_state["prefs"] = prefs
    render_evaluation()
