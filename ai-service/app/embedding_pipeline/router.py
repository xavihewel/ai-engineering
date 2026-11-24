"""HTTP orchestration for batch embedding."""

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.embedding_pipeline.chunker import JSONStructuralChunker
from app.embedding_pipeline.embedder import OpenAIEmbedder
from app.embedding_pipeline.schemas import IngestRequest, IngestResponse
from app.generation.rag.schemas import IngestRequest as DocumentIngestRequest
from app.generation.rag.schemas import IngestResponse as DocumentIngestResponse

log = structlog.get_logger()


def ingest_batch(request: IngestRequest) -> IngestResponse:
    """Create embeddings and report aggregate metrics for the submitted budgets."""
    try:
        chunks = JSONStructuralChunker().chunk(request.budgets)
        embedder = OpenAIEmbedder()
        embedded = embedder.embed_many(chunks)
        return IngestResponse(
            chunks=embedded,
            stats={**embedder.stats, "total_budgets": len(request.budgets)},
        )
    except Exception as exc:
        log.exception("batch_embeddings_ingest_failed", error_type=type(exc).__name__)
        raise HTTPException(status_code=500, detail="Failed to generate embeddings.") from exc


def create_router(legacy_handler=None, legacy_service_dependency=None) -> APIRouter:
    """Build a router, optionally retaining the existing document-ingest contract."""
    router = APIRouter(tags=["embeddings"])
    if legacy_handler is None:
        router.post("/ingest", response_model=IngestResponse)(ingest_batch)
    else:
        @router.post("/ingest", response_model=IngestResponse | DocumentIngestResponse,
                     responses={409: {"description": "Document already ingested"}})
        async def ingest(request: IngestRequest | DocumentIngestRequest,
                         service=Depends(legacy_service_dependency)):
            if isinstance(request, IngestRequest):
                from fastapi.concurrency import run_in_threadpool
                return await run_in_threadpool(ingest_batch, request)
            return await legacy_handler(request, service)
    return router


router = create_router()
