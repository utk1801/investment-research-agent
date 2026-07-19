"""Ingestion status page — table counts + manual trigger."""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import streamlit as st
from sqlalchemy import create_engine, text

from src.ui.app import render_sidebar, render_home
from src.config import get_db_url

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()




def _counts() -> dict:
    engine = create_engine(get_db_url())
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT 'tickers', COUNT(*) FROM tickers
            UNION ALL SELECT 'prices', COUNT(*) FROM prices
            UNION ALL SELECT 'financials', COUNT(*) FROM financial_metrics
            UNION ALL SELECT 'documents', COUNT(*) FROM documents
        """))
        return {"tickers": 0, "prices": 0, "financials": 0, "documents": 0, "chroma": 0} | dict(result.fetchall())


def _run_subprocess(step: str, reset: bool = False) -> tuple[int, str, str]:
    cmd = [sys.executable, "-m", "src.pipeline.run_ingestion"]
    if step:
        cmd.extend(["--step", step])
    if reset:
        cmd.append("--reset")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_PROJECT_ROOT))
    return result.returncode, result.stdout, result.stderr


def _trigger_ingestion():
    return _run_subprocess("fetch")


def _trigger_embed():
    return _run_subprocess("embed", reset=True)


def _trigger_backfill():
    return _run_subprocess("backfill")




def render_ingestion():
    prefs = st.session_state.get("prefs", {})
    st.title("📥 Ingestion Status")
    st.divider()

    counts = _counts()

    # Stat tiles
    col1, col2, col3, col4 = st.columns(4)
    for col, (name, count) in zip(
        [col1, col2, col3, col4],
        [("Tickers", counts["tickers"]),
         ("Price rows", counts["prices"]),
         ("Financial records", counts["financials"]),
         ("Documents", counts["documents"])],
    ):
        with col:
            st.metric(label=name, value=f"{count:,}" if count else "0", delta="✅" if count else "❌")

    st.divider()

    # Action buttons
    col_f, col_e, col_b = st.columns(3)
    with col_f:
        st.subheader("🔄 Refresh Data")
        st.caption("Pull latest yfinance data into PostgreSQL.")
        if st.button("Run Ingestion", type="primary", use_container_width=True):
            with st.spinner("Ingesting... (~30s)"):
                code, out, err = _trigger_ingestion()
            if code == 0:
                st.success("Ingestion complete!")
                st.rerun()
            else:
                st.error(f"Failed: {err[-500:]}")
                st.text(out[-500:])

    with col_e:
        st.subheader("📦 Rebuild Index")
        st.caption("Re-embed documents into ChromaDB + build BM25.")
        if st.button("Build Index", type="secondary", use_container_width=True):
            with st.spinner("Embedding... (~30s)"):
                code, out, err = _trigger_embed()
            if code == 0:
                st.success("Index built!")
                st.rerun()
            else:
                st.error(f"Failed: {err[-500:]}")
                st.text(out[-500:])

    with col_b:
        st.subheader("📊 Backfill Financial Docs")
        st.caption("Export financial_metrics rows to ChromaDB.")
        if st.button("Backfill", type="secondary", use_container_width=True):
            with st.spinner("Backfilling... (~30s)"):
                code, out, err = _trigger_backfill()
            if code == 0:
                st.success("Backfill done! Rebuild index next.")
            else:
                st.error(f"Failed: {err[-500:]}")
                st.text(out[-500:])

    st.divider()

    # Ticker detail
    st.subheader("📋 Ticker Detail")
    engine = create_engine(get_db_url())
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT t.symbol, t.sector, t.name,
                   (SELECT COUNT(*) FROM prices WHERE ticker_id = t.id) AS price_count,
                   (SELECT COUNT(*) FROM financial_metrics WHERE ticker_id = t.id) AS fin_count,
                   (SELECT COUNT(*) FROM documents WHERE ticker_id = t.id) AS doc_count
            FROM tickers t ORDER BY t.symbol
        """)).fetchall()

    st.dataframe(
        [{"Symbol": r[0], "Sector": r[1], "Name": r[2],
          "Prices": r[3], "Financials": r[4], "Documents": r[5]}
         for r in rows],
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    render_home()
    render_ingestion()