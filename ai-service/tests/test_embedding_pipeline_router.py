"""HTTP contract for batch ingestion."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.embedding_pipeline import router as batch_router


def test_batch_ingest_and_swagger(monkeypatch):
    class Chunker:
        def chunk(self, budgets):
            assert len(budgets) == 1
            return []

    class Embedder:
        stats = {"total_budgets": 0, "total_chunks": 0, "total_tokens": 0, "estimated_cost_usd": 0.0}
        def embed_many(self, chunks):
            assert chunks == []
            return []

    monkeypatch.setattr(batch_router, "JSONStructuralChunker", Chunker)
    monkeypatch.setattr(batch_router, "OpenAIEmbedder", Embedder)
    app = FastAPI()
    app.include_router(batch_router.router, prefix="/embeddings")
    client = TestClient(app)
    sample = {"budget_id": "B", "client_metadata": {"name": "C", "sector": "finance", "country": "ES"}, "project_summary": "P", "main_technology": "python", "year": 2024, "total_estimated_hours": 1, "components": [{"component_id": "A", "name": "N", "description": "D", "estimated_hours": 1, "complexity": "low"}]}
    response = client.post('/embeddings/ingest', json={"budgets": [sample]})
    assert response.status_code == 200
    assert response.json()["stats"]["total_budgets"] == 1
    assert client.post('/embeddings/ingest', json={"budgets": [{}]}).status_code == 422
    assert '/embeddings/ingest' in client.get('/openapi.json').json()['paths']
    assert client.get('/docs').status_code == 200


def test_errors_are_generic(monkeypatch):
    def fail():
        raise RuntimeError('secret detail')
    monkeypatch.setattr(batch_router, 'JSONStructuralChunker', fail)
    app = FastAPI()
    app.include_router(batch_router.router, prefix='/embeddings')
    response = TestClient(app).post('/embeddings/ingest', json={'budgets': []})
    assert response.status_code == 500
    assert 'secret detail' not in response.text


def test_combined_router_dispatches_both_contracts(monkeypatch):
    from app.generation.rag.schemas import IngestResponse as DocumentResponse

    async def legacy_handler(request, service):
        assert service == 'service'
        return DocumentResponse(document_id=1, chunks_created=1, embedding_dimension=1536, ingestion_time_ms=1)

    def service_dependency():
        return 'service'

    monkeypatch.setattr(batch_router, 'ingest_batch', lambda request: batch_router.IngestResponse(
        chunks=[], stats={'total_budgets': len(request.budgets), 'total_chunks': 0, 'total_tokens': 0, 'estimated_cost_usd': 0.0}))
    app = FastAPI()
    app.include_router(batch_router.create_router(legacy_handler, service_dependency), prefix='/embeddings')
    client = TestClient(app)
    assert client.post('/embeddings/ingest', json={'budgets': []}).json()['stats']['total_budgets'] == 0
    budget = {'budget_id': 'B', 'client_metadata': {'name': 'C', 'sector': 'finance', 'country': 'ES'}, 'project_summary': 'P', 'main_technology': 'python', 'year': 2024, 'total_estimated_hours': 1, 'components': [{'component_id': 'A', 'name': 'N', 'description': 'D', 'estimated_hours': 1, 'complexity': 'low'}]}
    response = client.post('/embeddings/ingest', json={'source_path': 'file', 'document_type': 'budget', 'content': budget})
    assert response.status_code == 200
    assert response.json()['document_id'] == 1
    assert 'anyOf' in client.get('/openapi.json').json()['paths']['/embeddings/ingest']['post']['requestBody']['content']['application/json']['schema']


def test_sample_budget_through_http(monkeypatch):
    import json
    from pathlib import Path
    from app.embedding_pipeline.schemas import EmbeddedChunk

    class FakeEmbedder:
        def embed_many(self, chunks):
            self.stats = {'total_budgets': 1, 'total_chunks': len(chunks), 'total_tokens': sum(c.token_count for c in chunks), 'estimated_cost_usd': 0.0}
            return [EmbeddedChunk(**c.model_dump(), embedding=[0.0] * 1536) for c in chunks]

    monkeypatch.setattr(batch_router, 'OpenAIEmbedder', FakeEmbedder)
    app = FastAPI()
    app.include_router(batch_router.router, prefix='/embeddings')
    sample = json.loads((Path(__file__).parents[1] / 'data/budgets_sample.json').read_text())[:1]
    response = TestClient(app).post('/embeddings/ingest', json={'budgets': sample})
    assert response.status_code == 200
    assert len(response.json()['chunks']) == len(sample[0]['components'])
