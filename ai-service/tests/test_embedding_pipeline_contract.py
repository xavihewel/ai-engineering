"""Behavioral tests for the batch embedding pipeline."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import tiktoken

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import Budget, Chunk, IngestResponse


@pytest.fixture
def budget():
    return Budget.model_validate({
        "budget_id": "BUD-1",
        "client_metadata": {"name": "Client", "sector": "finance", "country": "ES"},
        "project_summary": "Banking application",
        "main_technology": "python",
        "year": 2024,
        "total_estimated_hours": 8,
        "components": [{
            "component_id": "AUTH-1", "name": "Authentication", "description": "Secure login",
            "tech_stack": ["python"], "estimated_hours": 8, "complexity": "low",
        }],
    })


def test_structural_chunk_preserves_context_metadata_and_tokens(budget):
    chunks = JSONStructuralChunker().chunk([budget])
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "BUD-1::AUTH-1"
    assert "Banking application" in chunk.text and "Secure login" in chunk.text
    assert chunk.metadata == {
        "budget_id": "BUD-1", "component_id": "AUTH-1", "client_sector": "finance",
        "main_technology": "python", "year": 2024, "complexity": "low", "estimated_hours": 8,
    }
    assert chunk.token_count == len(tiktoken.encoding_for_model("text-embedding-3-small").encode(chunk.text))


def test_long_component_is_not_split(budget):
    budget.components[0].description = "Long description. " * 1000
    chunks = JSONStructuralChunker().chunk([budget])
    assert len(chunks) == 1
    assert budget.components[0].description in chunks[0].text


def test_embed_many_batches_preserves_order_and_reports_cost():
    client = Mock()
    client.embeddings.create.side_effect = lambda **kwargs: SimpleNamespace(data=[
        SimpleNamespace(index=i, embedding=[float(value)])
        for i, value in reversed(list(enumerate(kwargs["input"])))
    ])
    chunks = [Chunk(chunk_id=str(i), text=str(i), metadata={"budget_id": "BUD-1"}, token_count=10) for i in range(201)]
    embedder = OpenAIEmbedder(client)
    result = embedder.embed_many(chunks)
    assert [len(call.kwargs["input"]) for call in client.embeddings.create.call_args_list] == [100, 100, 1]
    assert all(call.kwargs["model"] == "text-embedding-3-small" and "dimensions" not in call.kwargs for call in client.embeddings.create.call_args_list)
    assert [chunk.embedding for chunk in result] == [[float(i)] for i in range(201)]
    assert embedder.stats == {"total_budgets": 1, "total_chunks": 201, "total_tokens": 2010, "estimated_cost_usd": 2010 * 0.02 / 1_000_000}
    IngestResponse(chunks=result, stats=embedder.stats)


def test_rate_limit_retries_and_other_errors_propagate(monkeypatch):
    class RateLimit(Exception):
        pass

    monkeypatch.setattr("app.embedding_pipeline.embedder.RateLimitError", RateLimit)
    sleeps = []
    monkeypatch.setattr("app.embedding_pipeline.embedder.time.sleep", sleeps.append)
    client = Mock()
    client.embeddings.create.side_effect = [RateLimit(), RateLimit(), RateLimit(), SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0])])]
    embedder = OpenAIEmbedder(client)
    assert embedder.embed_one("text") == [1.0]
    assert sleeps == [1, 2, 4]
    client.embeddings.create.side_effect = ValueError("failure")
    with pytest.raises(ValueError, match="failure"):
        embedder.embed_one("text")


def test_empty_batch_does_not_call_api():
    client = Mock()
    embedder = OpenAIEmbedder(client)
    assert embedder.embed_many([]) == []
    assert embedder.stats["estimated_cost_usd"] == 0
    client.embeddings.create.assert_not_called()
