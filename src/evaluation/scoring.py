"""Metric helpers for retrieval and generation evaluation."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from src.evaluation.dataset import EvalCase


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "based", "be", "but", "by", "for",
    "from", "has", "have", "how", "in", "is", "it", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "what", "which", "with", "about",
    "did", "does", "say", "said",
}

_TICKER_ALIASES = {
    "AAPL": ["aapl", "apple"],
    "MSFT": ["msft", "microsoft"],
    "NVDA": ["nvda", "nvidia"],
    "JPM": ["jpm", "jpmorgan", "jpmorgan chase"],
    "GS": ["gs", "goldman", "goldman sachs"],
    "AMZN": ["amzn", "amazon", "aws"],
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokens(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9&.-]+", text.lower())
        if t not in _STOPWORDS and len(t) > 2
    ]


def relevance_score(case: EvalCase, doc: dict) -> float:
    """Return graded relevance in [0, 1] for one retrieved chunk."""
    text = _norm(f"{doc.get('title', '')} {doc.get('doc_type', '')} {doc.get('chunk_text', '')}")
    score = 0.0

    if case.expected_tickers:
        title = _norm(doc.get("title", ""))
        ticker_text = _norm(str(doc.get("ticker") or ""))
        aliases = []
        for ticker in case.expected_tickers:
            aliases.extend(_TICKER_ALIASES.get(ticker, [_norm(ticker)]))
        ticker_hit = any(alias in title or alias == ticker_text for alias in aliases)
        if ticker_hit:
            score += 0.4
    else:
        score += 0.4

    if case.expected_doc_types:
        if doc.get("doc_type") in case.expected_doc_types:
            score += 0.3
    else:
        score += 0.3

    if case.expected_source_terms:
        term_hits = sum(1 for term in case.expected_source_terms if _norm(term) in text)
        score += 0.3 * min(1.0, term_hits / max(1, min(2, len(case.expected_source_terms))))
    else:
        score += 0.3

    return min(1.0, score)


def retrieval_metrics(case: EvalCase, docs: Sequence[dict], k: int = 5) -> dict:
    scores = [relevance_score(case, doc) for doc in docs[:k]]
    binary = [1 if s >= 0.7 else 0 for s in scores]

    hit_rate = 1.0 if any(binary) else 0.0
    precision = sum(binary) / k if k else 0.0
    reciprocal_rank = 0.0
    for idx, is_relevant in enumerate(binary, 1):
        if is_relevant:
            reciprocal_rank = 1.0 / idx
            break

    dcg = sum(score / math.log2(idx + 2) for idx, score in enumerate(scores))
    ideal = sorted(scores, reverse=True)
    idcg = sum(score / math.log2(idx + 2) for idx, score in enumerate(ideal))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        "hit_rate": hit_rate,
        "precision_at_k": precision,
        "mrr": reciprocal_rank,
        "ndcg_at_k": ndcg,
        "relevant_count": sum(binary),
        "top_score": scores[0] if scores else 0.0,
    }


def fact_coverage(case: EvalCase, answer: str) -> float:
    if not case.expected_answer_terms:
        return 1.0
    answer_norm = _norm(answer)
    hits = sum(1 for term in case.expected_answer_terms if _norm(term) in answer_norm)
    return hits / len(case.expected_answer_terms)


def context_groundedness(answer: str, docs: Sequence[dict]) -> float:
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = set(_tokens(" ".join(doc.get("chunk_text", "") for doc in docs)))
    if not context_tokens:
        return 0.0
    counts = Counter(answer_tokens)
    supported = sum(count for tok, count in counts.items() if tok in context_tokens)
    return supported / sum(counts.values())


def citation_score(answer: str) -> float:
    citations = re.findall(r"\[(?:source\s*)?\d+\]", answer, flags=re.IGNORECASE)
    return min(1.0, len(citations) / 2)


def generation_metrics(case: EvalCase, answer: str, docs: Sequence[dict]) -> dict:
    coverage = fact_coverage(case, answer)
    grounded = context_groundedness(answer, docs)
    citations = citation_score(answer)
    overall = (0.45 * coverage) + (0.4 * grounded) + (0.15 * citations)
    return {
        "fact_coverage": coverage,
        "groundedness": grounded,
        "citation_score": citations,
        "overall_score": overall,
    }
