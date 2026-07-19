"""OpenAI LLM client with prompt templates + token cost tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from openai import OpenAI

from src.config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE

# Pricing per 1M tokens (USD), Jan 2025
_MODEL_PRICING = {
    "gpt-4o":           (2.50,  10.00),
    "gpt-4o-mini":      (0.15,   0.60),
    "gpt-3.5-turbo":    (0.50,   1.50),
}


@dataclass
class LLMResponse:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


def _calc_cost(model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, float, float]:
    input_p, output_p = _MODEL_PRICING.get(model, (0.0, 0.0))
    input_cost = prompt_tokens * input_p / 1_000_000
    output_cost = completion_tokens * output_p / 1_000_000
    return input_cost, output_cost, input_cost + output_cost


class LLMClient:
    def __init__(self, model: str | None = None, temperature: float | None = None):
        import os
        self.client = OpenAI(api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"))
        self.model = model or LLM_MODEL
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self._prompts = _PROMPTS

    def generate(self, question: str, chunks: Sequence[dict], variant: str = "A") -> LLMResponse:
        """Generate answer from LLM given retrieved chunks and prompt variant."""
        if not chunks:
            return LLMResponse(answer="No relevant documents found. Try rephrasing or widening the ticker filter.",
                              prompt_tokens=0, completion_tokens=0, total_tokens=0,
                              input_cost_usd=0.0, output_cost_usd=0.0, total_cost_usd=0.0)

        context = _build_context(chunks)
        system = self._prompts[variant]

        user_msg = f"{system}\n\n---\nContext:\n{context}\n---\nQuestion: {question}\n---\nAnswer:"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": user_msg}],
                temperature=self.temperature,
            )
            usage = response.usage
            prompt_t = usage.prompt_tokens
            completion_t = usage.completion_tokens
            total_t = usage.total_tokens
            input_cost, output_cost, total_cost = _calc_cost(self.model, prompt_t, completion_t)
            return LLMResponse(
                answer=response.choices[0].message.content.strip(),
                prompt_tokens=prompt_t,
                completion_tokens=completion_t,
                total_tokens=total_t,
                input_cost_usd=input_cost,
                output_cost_usd=output_cost,
                total_cost_usd=total_cost,
            )
        except Exception as exc:
            return LLMResponse(answer=f"LLM unavailable: {exc}",
                              prompt_tokens=0, completion_tokens=0, total_tokens=0,
                              input_cost_usd=0.0, output_cost_usd=0.0, total_cost_usd=0.0)


def _build_context(chunks: Sequence[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        title = c.get("title") or c.get("doc_type", "document")
        date = c.get("doc_date") or "unknown date"
        text = (c.get("chunk_text") or "")[:600]
        parts.append(f"[{i}] {title} ({date}):\n{text}")
    return "\n\n".join(parts)


_PROMPTS = {
    "A": (
        "You are a financial analyst. Answer based only on the provided context.\n"
        "If context is insufficient, say you don't have enough information."
    ),
    "B": (
        "You are a senior financial analyst. Use the provided context to answer.\n"
        "Cite each statement with [Source N] notation.\n"
        "Structure: [Findings] → [Supporting Data] → [Limitations/Gaps]"
    ),
    "C": (
        "You are a precise financial analyst. Answer using ONLY the provided context.\n"
        "Do NOT make up numbers, ratios, or facts. If not in context, say 'not available'."
    ),
}