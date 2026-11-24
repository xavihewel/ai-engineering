"""Batch embeddings for structural budget chunks."""

from __future__ import annotations

import time

import structlog
from openai import OpenAI, RateLimitError

from app.embedding_pipeline.schemas import Chunk, EmbeddedChunk

MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
# Input price in USD per million tokens; update when pricing changes.
PRICE_PER_MILLION_TOKENS_USD = 0.02
_RETRY_DELAYS = (1, 2, 4)
log = structlog.get_logger()


class OpenAIEmbedder:
    """Embed chunks in batches; stats describe the last successful batch run."""

    def __init__(self, client: OpenAI | None = None) -> None:
        # Disable SDK retries so the explicit backoff controls the retry count.
        self._client = client if client is not None else OpenAI(max_retries=0)
        self.stats: dict[str, int | float] = {}

    def embed_one(self, text: str) -> list[float]:
        """Embed one text using the model's default dimensions."""
        return self._create([text])[0]

    def embed_many(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed chunks in order and record aggregate input-token costs."""
        self.stats = {}
        embedded: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            started = time.perf_counter()
            vectors = self._create([chunk.text for chunk in batch])
            latency_ms = (time.perf_counter() - started) * 1000
            log.info(
                "embedding_batch_done",
                chunks=len(batch),
                tokens=sum(chunk.token_count for chunk in batch),
                latency_ms=round(latency_ms, 1),
                model=MODEL,
            )
            embedded.extend(
                EmbeddedChunk(**chunk.model_dump(), embedding=vector)
                for chunk, vector in zip(batch, vectors, strict=True)
            )
        total_tokens = sum(chunk.token_count for chunk in chunks)
        self.stats = {
            "total_budgets": len({c.metadata["budget_id"] for c in chunks if "budget_id" in c.metadata}),
            "total_chunks": len(embedded),
            "total_tokens": total_tokens,
            "estimated_cost_usd": total_tokens * PRICE_PER_MILLION_TOKENS_USD / 1_000_000,
        }
        return embedded

    def _create(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(len(_RETRY_DELAYS) + 1):
            try:
                response = self._client.embeddings.create(model=MODEL, input=texts)
                data = sorted(response.data, key=lambda item: item.index)
                if [item.index for item in data] != list(range(len(texts))):
                    raise ValueError("Embedding response indices do not match the input batch")
                return [item.embedding for item in data]
            except RateLimitError:
                if attempt == len(_RETRY_DELAYS):
                    raise
                delay = _RETRY_DELAYS[attempt]
                log.warning("embedding_rate_limited", retry_in_s=delay)
                time.sleep(delay)
        raise AssertionError("Unreachable")
