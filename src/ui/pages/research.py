"""Research page — query input, retrieval + LLM, source citations."""

from __future__ import annotations

import logging
import time

import streamlit as st

from src.ui.app import render_sidebar


def render_research():
    prefs = st.session_state.get("prefs", {})
    selected_ticker = prefs.get("ticker", "")
    model = prefs.get("model", "gpt-4o")
    prompt_variant = prefs.get("prompt_variant", "A")
    retrieval_approach = prefs.get("retrieval_approach", "hybrid_rerank")
    rewrite_enabled = prefs.get("rewrite_query", True)

    st.title("🔍 Research")
    st.caption(f"Model: **{model}** · Prompt: **{prompt_variant}**" +
               f" · Retrieval: **{retrieval_approach}**" +
               (" · Query rewrite: **on**" if rewrite_enabled else " · Query rewrite: **off**") +
               (f" · Ticker filter: **{selected_ticker}**" if selected_ticker else ""))

    # History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📚 Sources"):
                        for s in msg["sources"]:
                            st.markdown(f"**[{s['title']}](/{s['url']})** — {s['date'] or 'no date'}")
                            st.caption(s["snippet"][:300] + ("..." if len(s["snippet"]) > 300 else ""))
                _render_feedback_ui(key=f"hist_{msg['ts']}", query_id=msg.get("query_id"))

    # Prompt input
    prompt = st.chat_input("Ask about AAPL CEO strategy, JPM credit outlook, NVDA AI demand...")

    if prompt:
        st.chat_message("user").write(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        t0 = time.time()
        with st.spinner("Running retrieval + LLM synthesis..."):
            response = _run_query(
                prompt, selected_ticker, model, prompt_variant,
                retrieval_approach, rewrite_enabled,
            )

        elapsed = time.time() - t0
        with st.chat_message("assistant"):
            st.markdown(f"**Answer** *(in {elapsed:.1f}s)*\n\n{response['answer']}")

            # Token / cost breakdown
            tokens = response.get("tokens", 0)
            cost = response.get("cost_usd", 0.0)
            pt = response.get("prompt_tokens", 0)
            ct = response.get("completion_tokens", 0)
            if tokens > 0:
                st.caption(f"💰 {tokens:,} tokens ({pt:,}in / {ct:,}out) ≈ **${cost:.6f}** · Retrieved **{response['chunks']}** chunks")
            else:
                st.caption(f"Retrieved **{response['chunks']}** chunks")
            if response.get("rewritten_query") and response["rewritten_query"] != prompt:
                st.caption(f"Rewritten query: {response['rewritten_query']}")

            if response.get("sources"):
                with st.expander("📚 Sources"):
                    for s in response["sources"]:
                        st.markdown(f"**{s['title']}** — {s['date'] or 'no date'}")
                        st.caption(s["snippet"][:300] + ("..." if len(s["snippet"]) > 300 else ""))

            _render_feedback_ui(key="current", query_id=response.get("query_id"))

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response.get("sources", []),
            "query_id": response.get("query_id"),
            "ts": time.time(),
        })

def _run_query(
    prompt: str,
    ticker: str,
    model: str,
    prompt_variant: str,
    retrieval_approach: str,
    rewrite_enabled: bool,
) -> dict:
    """Run the full query pipeline. Returns {answer, chunks, sources}."""
    from src.retrieval.retriever import HybridRetriever
    from src.llm.client import LLMClient
    from src.monitoring.metrics import log_query
    from src.config import RETRIEVAL_TOP_K_HYBRID

    try:
        t0 = time.time()
        retriever = HybridRetriever(ticker_filter=ticker or None)
        # Retrieve more than top_k to account for chunk dedup — keep best per doc
        all_chunks = retriever.retrieve(
            prompt,
            top_k=RETRIEVAL_TOP_K_HYBRID,
            approach=retrieval_approach,
            rewrite=rewrite_enabled,
        )
        seen_doc_ids, chunks = set(), []
        for c in all_chunks:
            doc_id = c.get("doc_id")
            if doc_id is None or doc_id not in seen_doc_ids:
                chunks.append(c)
                if doc_id is not None:
                    seen_doc_ids.add(doc_id)
            if len(chunks) >= 5:
                break

        llm = LLMClient(model=model)
        resp = llm.generate(prompt, chunks, variant=prompt_variant)
        elapsed_ms = int((time.time() - t0) * 1000)
        query_id = log_query(
            query_text=prompt,
            ticker_filter=ticker,
            retrieval_chunk_count=len(chunks),
            retrieval_time_ms=elapsed_ms,
            model=model,
            prompt_variant=prompt_variant,
            rewritten_query=retriever.last_rewritten_query,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            total_tokens=resp.total_tokens,
            input_cost_usd=resp.input_cost_usd,
            output_cost_usd=resp.output_cost_usd,
            total_cost_usd=resp.total_cost_usd,
        )
        sources = [
            {
                "title": c.get("title") or c.get("doc_type", "document"),
                "date": c.get("doc_date"),
                "snippet": c.get("chunk_text", "")[:300],
                "url": "#",
            }
            for c in chunks
        ]
        return {
            "answer": resp.answer,
            "chunks": len(chunks),
            "sources": sources,
            "query_id": query_id,
            "tokens": resp.total_tokens,
            "cost_usd": resp.total_cost_usd,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "rewritten_query": retriever.last_rewritten_query,
            "retrieval_approach": retrieval_approach,
        }
    except Exception as exc:
        return {
            "answer": f"⚠️ Error: {exc}\n\nMake sure ingestion ran and ChromaDB is healthy.",
            "chunks": 0,
            "sources": [],
            "query_id": None,
        }


# ── Feedback helpers ─────────────────────────────────────────────────────────


def _submit_feedback(query_id: int, thumbs_up: bool) -> None:
    try:
        from src.monitoring.metrics import MetricsLogger
        fl = MetricsLogger()
        fl.log_feedback(query_id, thumbs_up=thumbs_up, thumbs_down=not thumbs_up)
        fl.close()
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to log feedback: %s", exc)


def _render_feedback_ui(key: str, query_id: int | None) -> None:
    """Render thumbs + optional comment.

    Session state keyed by query_id so it stays attached to the correct message
    even after it moves from 'current' to 'hist_{ts}' in the chat.
    """
    if query_id is None:
        return

    feedback_key = f"fb_q{query_id}"   # stable — never changes across reruns
    comment_key = f"comment_q{query_id}"

    if comment_key not in st.session_state:
        st.session_state[comment_key] = ""

    if feedback_key not in st.session_state:
        st.session_state[feedback_key] = {"sent": False, "thumbs": None, "done": False}

    state = st.session_state[feedback_key]

    if not state["sent"]:
        c1, c2 = st.columns(2)
        c1.button(
            "👍", key=f"up_{feedback_key}", use_container_width=True,
            on_click=_on_thumbs,
            args=(query_id, True, feedback_key),
        )
        c2.button(
            "👎", key=f"down_{feedback_key}", use_container_width=True,
            on_click=_on_thumbs,
            args=(query_id, False, feedback_key),
        )
    else:
        st.success("✅ Saved")
        if not state["done"] and not state.get("has_commented") and not state["thumbs"]:
            st.text_area("What could be improved?", key=comment_key,
                         placeholder="Optional — helps identify what went wrong")
            c, spacer = st.columns([1, 4])
            c.button("Send", key=f"send_{feedback_key}", use_container_width=True,
                     on_click=_on_comment, args=(feedback_key,))


def _on_thumbs(query_id: int, thumbs_up: bool, feedback_key: str) -> None:
    _submit_feedback(query_id, thumbs_up)
    st.session_state[feedback_key] = {
        "sent": True,
        "thumbs": thumbs_up,
        "done": False,
        "query_id": query_id,
        "has_commented": False,
    }


def _on_comment(feedback_key: str) -> None:
    log = logging.getLogger(__name__)
    fb = st.session_state.get(feedback_key, {})
    query_id = fb.get("query_id")
    comment_key = f"comment_q{query_id}"
    comment_raw = st.session_state.get(comment_key)
    comment = (comment_raw or "") or ""
    log.warning(f"_on_comment: query_id={query_id}, comment_key={comment_key}, raw={repr(comment_raw)}")
    if comment.strip() and query_id:
        try:
            import sys
            sys.modules.pop("src.monitoring.metrics", None)
            from src.monitoring.metrics import MetricsLogger
            fl = MetricsLogger()
            log.warning(f"_on_comment: writing comment to DB: '{comment.strip()}' for query_id={query_id}")
            fl.log_feedback(query_id, comment=comment.strip())
            fl.close()
        except Exception as exc:
            logging.getLogger(__name__).warning("Failed to log comment: %s", exc)
    log.warning(f"_on_comment: SKIPPED — comment='{comment}', query_id={query_id}")
    st.session_state[feedback_key]["done"] = True
    st.session_state[feedback_key]["has_commented"] = True
    return


if __name__ == "__main__":
    prefs = render_sidebar()
    st.session_state["prefs"] = prefs
    render_research()
