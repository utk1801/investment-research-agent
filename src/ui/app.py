"""Streamlit app — shared sidebar settings and home page."""

from __future__ import annotations

import streamlit as st

from src.config import TICKER_SECTORS, ALL_TICKERS, LLM_MODEL, TICKERS

st.set_page_config(
    page_title="Investment Research Agent",
    page_icon="📈",
    layout="wide",
)


def render_sidebar() -> dict:
    """Render sidebar and return user preferences as a dict."""
    with st.sidebar:
        st.title("📈 Research Agent")
        st.divider()

        st.caption("**Ticker Filter**")
        available_sectors = ["All"] + list(TICKERS.keys())
        default_sector = st.session_state.get("_sector", "All")
        sector_idx = available_sectors.index(default_sector) if default_sector in available_sectors else 0
        selected_sector = st.selectbox(
            "Sector",
            options=available_sectors,
            index=sector_idx,
            key="_sector",
        )

        if selected_sector == "All":
            ticker_opts = ALL_TICKERS
        else:
            ticker_opts = TICKERS.get(selected_sector, [])

        default_ticker = st.session_state.get("_ticker", "")
        ticker_idx = (ticker_opts.index(default_ticker) + 1) if default_ticker in ticker_opts else 0
        selected_ticker = st.selectbox(
            "Ticker",
            options=[""] + ticker_opts,
            index=ticker_idx,
            key="_ticker",
        )

        st.divider()
        st.caption("**Model Settings**")
        model = st.selectbox(
            "Model",
            options=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            index=["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"].index(LLM_MODEL)
            if LLM_MODEL in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
            else 0,
        )
        prompt_variant = st.selectbox(
            "Prompt Style",
            options=["A — Basic", "B — Structured + Citations", "C — Conservative"],
            index=1,
        )
        retrieval_approach = st.selectbox(
            "Retrieval",
            options=[
                "hybrid — Hybrid",
                "hybrid_rerank — Hybrid + rerank",
                "vector — Vector only",
                "bm25 — Keyword only",
            ],
            index=0,
        )
        rewrite_query = st.toggle("Rewrite queries", value=True)

    return {
        "ticker": selected_ticker,
        "model": model,
        "prompt_variant": prompt_variant.split(" — ")[0],
        "retrieval_approach": retrieval_approach.split(" — ")[0],
        "rewrite_query": rewrite_query,
    }


def render_home():
    st.title("Investment Research Agent")
    st.markdown(
        "Ask natural-language questions about stocks — live financial data, "
        "hybrid retrieval, and LLM synthesis."
    )
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📥 Ingestion")
        st.write("Pulls yfinance data: prices, financials, news for 12 tickers.")
        if st.button("Open Ingestion Panel", use_container_width=True, type="secondary"):
            st.switch_page("pages/ingestion.py")

    with col2:
        st.subheader("🔍 Research")
        st.write("Ask questions, get cited answers powered by hybrid search.")
        if st.button("Open Research Panel", use_container_width=True, type="primary"):
            st.switch_page("pages/research.py")

    with col3:
        st.subheader("📊 Dashboard")
        st.write("Query latency, retrieval stats, and user feedback in Grafana.")
        if st.button("Open Grafana", use_container_width=True, type="secondary"):
            st.switch_page("pages/dashboard.py")

    st.divider()


def render_home_page():
    prefs = render_sidebar()
    st.session_state["prefs"] = prefs
    render_home()


def main():
    page = st.navigation(
        [
            st.Page(render_home_page, title="Home", icon=":material/home:", default=True),
            st.Page("pages/research.py", title="Research", icon=":material/search:"),
            st.Page("pages/evaluation.py", title="Evaluation", icon=":material/science:"),
            st.Page("pages/ingestion.py", title="Ingestion", icon=":material/cloud_upload:"),
            st.Page("pages/dashboard.py", title="Dashboard", icon=":material/monitoring:"),
        ],
        position="top",
    )
    page.run()


if __name__ == "__main__":
    main()
