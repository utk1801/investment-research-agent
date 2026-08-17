"""Rule-based query rewriting for investment RAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import ALL_TICKERS


_COMPANY_ALIASES = {
    "AAPL": ["Apple", "Tim Cook", "Apple Intelligence", "iPhone"],
    "MSFT": ["Microsoft", "Satya Nadella", "Azure", "Copilot"],
    "NVDA": ["NVIDIA", "Nvidia", "Jensen Huang", "GPU", "accelerated computing"],
    "JPM": ["JPMorgan", "JPMorgan Chase", "Jamie Dimon", "credit risk"],
    "GS": ["Goldman Sachs", "David Solomon", "deal pipeline", "investment banking"],
    "AMZN": ["Amazon", "Andy Jassy", "AWS", "retail"],
}

_CONCEPT_EXPANSIONS = {
    "ai": ["artificial intelligence", "AI infrastructure", "data center", "accelerated computing"],
    "azure": ["cloud growth", "Azure revenue", "AI services"],
    "credit risk": ["commercial real estate", "charge-offs", "loan losses", "credit quality"],
    "guidance": ["outlook", "forward outlook", "forecast", "next quarter"],
    "cash flow": ["free cash flow", "operating cash flow", "capital expenditures"],
    "deal": ["M&A", "investment banking", "capital markets", "advisory pipeline"],
    "margin": ["gross margin", "operating margin", "profitability"],
}


@dataclass(frozen=True)
class QueryRewrite:
    original_query: str
    rewritten_query: str
    additions: list[str]


def rewrite_query(query: str) -> QueryRewrite:
    """Expand terse finance questions with ticker, company, and concept aliases."""
    q_lower = query.lower()
    additions: list[str] = []

    for ticker in ALL_TICKERS:
        ticker_lower = ticker.lower()
        aliases = _COMPANY_ALIASES.get(ticker, [])
        if ticker_lower in q_lower or any(alias.lower() in q_lower for alias in aliases):
            additions.extend([ticker, *aliases])

    for trigger, expansions in _CONCEPT_EXPANSIONS.items():
        if trigger in q_lower:
            additions.extend(expansions)

    deduped: list[str] = []
    seen = set()
    for item in additions:
        key = item.lower()
        if key not in seen and key not in q_lower:
            deduped.append(item)
            seen.add(key)

    if not deduped:
        return QueryRewrite(query, query, [])

    rewritten = f"{query} {' '.join(deduped)}"
    return QueryRewrite(query, rewritten, deduped)
