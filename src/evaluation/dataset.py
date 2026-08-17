"""Small labeled evaluation set for the investment research RAG flow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_tickers: list[str]
    expected_doc_types: list[str]
    expected_source_terms: list[str]
    expected_answer_terms: list[str]
    ticker_filter: str | None = None
    notes: str = ""


DEFAULT_EVAL_CASES: list[EvalCase] = [
    EvalCase(
        id="nvda-ai-demand",
        question="What did Jensen Huang say about AI infrastructure demand?",
        expected_tickers=["NVDA"],
        expected_doc_types=["earnings_call"],
        expected_source_terms=["AI", "infrastructure", "demand", "data center"],
        expected_answer_terms=["AI", "infrastructure", "demand"],
        ticker_filter="NVDA",
    ),
    EvalCase(
        id="msft-azure-growth",
        question="What is Microsoft's guidance for Azure growth?",
        expected_tickers=["MSFT"],
        expected_doc_types=["earnings_call"],
        expected_source_terms=["Azure", "growth", "guidance", "outlook"],
        expected_answer_terms=["Azure", "growth"],
        ticker_filter="MSFT",
    ),
    EvalCase(
        id="aapl-apple-intelligence",
        question="Did Tim Cook mention Apple Intelligence in China?",
        expected_tickers=["AAPL"],
        expected_doc_types=["earnings_call"],
        expected_source_terms=["Apple Intelligence", "China", "Tim Cook"],
        expected_answer_terms=["Apple Intelligence", "China"],
        ticker_filter="AAPL",
    ),
    EvalCase(
        id="jpm-credit-risk",
        question="How does Jamie Dimon view commercial real estate credit risk?",
        expected_tickers=["JPM"],
        expected_doc_types=["earnings_call"],
        expected_source_terms=["commercial real estate", "credit", "risk", "Jamie Dimon"],
        expected_answer_terms=["commercial real estate", "credit", "risk"],
        ticker_filter="JPM",
    ),
    EvalCase(
        id="gs-deal-pipeline",
        question="How does Goldman Sachs view the deal pipeline?",
        expected_tickers=["GS"],
        expected_doc_types=["earnings_call"],
        expected_source_terms=["deal pipeline", "M&A", "investment banking", "David Solomon"],
        expected_answer_terms=["deal", "pipeline"],
        ticker_filter="GS",
    ),
    EvalCase(
        id="amzn-aws-ai",
        question="What did Amazon management say about AWS and AI demand?",
        expected_tickers=["AMZN"],
        expected_doc_types=["earnings_call"],
        expected_source_terms=["AWS", "AI", "demand", "Andy Jassy"],
        expected_answer_terms=["AWS", "AI", "demand"],
        ticker_filter="AMZN",
    ),
    EvalCase(
        id="cross-company-ai",
        question="Compare AI demand commentary from Microsoft and NVIDIA.",
        expected_tickers=["MSFT", "NVDA"],
        expected_doc_types=["earnings_call"],
        expected_source_terms=["AI", "Azure", "NVIDIA", "demand"],
        expected_answer_terms=["Microsoft", "NVIDIA", "AI"],
    ),
    EvalCase(
        id="cash-flow-metric",
        question="Which transcript discusses free cash flow and margins?",
        expected_tickers=["AAPL", "MSFT", "NVDA", "AMZN"],
        expected_doc_types=["earnings_call"],
        expected_source_terms=["free cash flow", "margin", "gross margin"],
        expected_answer_terms=["cash flow", "margin"],
    ),
]
