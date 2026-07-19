"""Dashboard page — redirect link to Grafana."""

from __future__ import annotations

import streamlit as st

from src.ui.app import render_sidebar, render_home


def render_dashboard():
    st.title("📊 Monitoring Dashboard")
    st.divider()

    st.markdown(
        "The monitoring dashboard is powered by **Grafana** connected directly to "
        "the PostgreSQL database.\n\n"
        "Charts: query volume, latency, thumbs up/down ratio, rating distribution, "
        "prompt variant usage, retrieval chunk counts."
    )

    st.warning(
        "⚠️ **Grafana is running but dashboard not yet imported.**\n\n"
        "To import the dashboard:\n"
        "1. Open [http://localhost:3000](http://localhost:3000) (admin / admin123)\n"
        "2. Go to **Dashboards → Import**\n"
        "3. Upload or paste the JSON from `grafana/provisioning/dashboards/invest_agent.json`"
    )

    st.link_button("🚀 Open Grafana", "http://localhost:3000", use_container_width=True)

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
    render_home()
    render_dashboard()