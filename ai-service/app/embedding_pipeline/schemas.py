"""Input and output models for batch embedding of historical budgets."""

from typing_extensions import TypedDict

from pydantic import BaseModel

from app.generation.rag.schemas import (
    Budget,
    BudgetComponent,
    Chunk,
    ClientMetadata,
    EmbeddedChunk,
)


class IngestStats(TypedDict):
    """Aggregate metrics for a batch of embedded budgets."""

    total_budgets: int
    total_chunks: int
    total_tokens: int
    estimated_cost_usd: float


class IngestRequest(BaseModel):
    """Budgets to split and embed."""

    budgets: list[Budget]


class IngestResponse(BaseModel):
    """Embedded chunks and aggregate batch metrics."""

    chunks: list[EmbeddedChunk]
    stats: IngestStats


__all__ = [
    "BudgetComponent",
    "Budget",
    "Chunk",
    "ClientMetadata",
    "EmbeddedChunk",
    "IngestRequest",
    "IngestResponse",
    "IngestStats",
]
