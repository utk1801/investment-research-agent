"""Airflow DAG for scheduled investment research data ingestion."""

from __future__ import annotations

import os
from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum


PYTHON = "python"
LOCAL_TZ = pendulum.timezone("America/Los_Angeles")
PROJECT_CONTAINER = os.getenv("AIRFLOW_INGESTION_CONTAINER", "invest_app")

default_args = {
    "owner": "investment-research-agent",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def pipeline_task(task_id: str, step: str, extra_args: str = "") -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=(
            f"docker exec {PROJECT_CONTAINER} "
            f"{PYTHON} -m src.pipeline.run_ingestion --step {step} {extra_args}"
        ).strip(),
    )


with DAG(
    dag_id="investment_research_ingestion",
    description="Fetch data, ingest transcripts, rebuild embeddings, and rebuild BM25.",
    default_args=default_args,
    start_date=pendulum.datetime(2026, 1, 1, tz=LOCAL_TZ),
    schedule="0 6 * * *",
    catchup=False,
    is_paused_upon_creation=False,
    max_active_runs=1,
    tags=["investment-research", "rag", "ingestion"],
) as dag:
    fetch_market_data = pipeline_task("fetch_market_data", "fetch")
    ingest_transcripts = pipeline_task("ingest_transcripts", "transcripts")
    backfill_financial_docs = pipeline_task("backfill_financial_docs", "backfill")
    rebuild_embeddings = pipeline_task("rebuild_embeddings", "embed", "--reset")
    rebuild_bm25 = pipeline_task("rebuild_bm25", "bm25")
    collect_stats = pipeline_task("collect_stats", "stats")

    [fetch_market_data, ingest_transcripts] >> backfill_financial_docs
    backfill_financial_docs >> rebuild_embeddings >> rebuild_bm25 >> collect_stats
