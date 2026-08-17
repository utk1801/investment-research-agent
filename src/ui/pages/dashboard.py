"""Dashboard page — redirect link to Grafana."""

from __future__ import annotations

import streamlit as st

from src.ui.app import render_sidebar


def render_dashboard():
    st.title("📊 Monitoring Dashboard")
    st.divider()

    st.markdown(
        "The monitoring dashboard is provisioned automatically by Docker and connects "
        "to PostgreSQL through the `invest_agent_psql` datasource.\n\n"
        "Charts: query volume, latency, thumbs up/down ratio, prompt variant usage, "
        "retrieval chunk counts, cost, tokens, and recent queries."
    )

    st.info("Grafana login: `admin` / `admin123`")

    st.link_button("Open Grafana", "http://localhost:3000", icon=":material/open_in_new:")

    st.divider()

    # Quick stats from PostgreSQL
    st.subheader("📈 Live Stats")
    try:
        from sqlalchemy import create_engine, text
        from src.config import get_db_url
        engine = create_engine(get_db_url())
        with engine.connect() as conn:
            qlog = conn.execute(text("SELECT COUNT(*) FROM query_log")).scalar()
            feedback = conn.execute(text("SELECT COUNT(*) FROM feedback")).scalar()
            thumbs_up = conn.execute(text("SELECT COUNT(*) FROM feedback WHERE thumbs_up = TRUE")).scalar()
            thumbs_down = conn.execute(text("SELECT COUNT(*) FROM feedback WHERE thumbs_down = TRUE")).scalar()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Queries logged", f"{qlog or 0}")
        col2.metric("Feedback items", f"{feedback or 0}")
        col3.metric("👍 Thumbs up", f"{thumbs_up or 0}")
        col4.metric("👎 Thumbs down", f"{thumbs_down or 0}")
    except Exception as exc:
        st.info(f"DB not reachable: {exc}")


if __name__ == "__main__":
    prefs = render_sidebar()
    st.session_state["prefs"] = prefs
    render_dashboard()
