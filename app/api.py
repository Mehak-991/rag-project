"""
FastAPI application with RAG endpoints.
Provides REST API for document ingestion, querying, and system management.
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import tempfile

from app.config import get_settings
from app.logger import logger, perf_logger, estimate_tokens
from app.ingest import get_ingestor
from app.retriever import get_retriever
from app.generator import get_generator
from app.vectorstore import get_vector_store

settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Cost-Efficient RAG Application with ChromaDB and Ollama"
)

# Mount static files for serving the HTML UI
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    category: Optional[str] = None
    file_type: Optional[str] = None
    metadata_filter: Optional[Dict[str, Any]] = None
    rerank: Optional[bool] = True


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    context_used: bool
    retrieval_latency: float
    generation_latency: float
    total_latency: float
    chunks_retrieved: int
    token_estimate: int


class IngestResponse(BaseModel):
    success: bool
    message: str
    file_name: str
    chunks_added: Optional[int] = None
    latency: Optional[float] = None
    file_hash: Optional[str] = None


class StatsResponse(BaseModel):
    collection_name: str
    document_count: int
    embedding_model: str
    persist_directory: str


class MetricsResponse(BaseModel):
    query_latency: float
    retrieval_latency: float
    generation_latency: float
    chunks_retrieved: int
    token_estimate: int


# Initialize components
ingestor = get_ingestor()
retriever = get_retriever()
generator = get_generator()
vector_store = get_vector_store()


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Check if Ollama model is available
    if not generator.check_model_available():
        logger.warning(f"Model {settings.ollama_model} not found in Ollama")
        logger.info(f"Pulling model {settings.ollama_model}...")
        if generator.pull_model():
            logger.info(f"Model {settings.ollama_model} pulled successfully")
        else:
            logger.error(f"Failed to pull model {settings.ollama_model}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check vector store
        stats = vector_store.get_collection_stats()
        
        # Check Ollama
        model_available = generator.check_model_available()
        
        return {
            "status": "healthy",
            "app_name": settings.app_name,
            "version": settings.app_version,
            "vector_store": "connected" if "error" not in stats else "disconnected",
            "ollama_model": settings.ollama_model,
            "model_available": model_available
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    category: str = Query(default="general"),
    chunk_size: Optional[int] = Query(default=None),
    chunk_overlap: Optional[int] = Query(default=None)
):
    """
    Ingest a document into the vector store.
    
    Supports: PDF, HTML, Markdown
    """
    start_time = time.time()
    
    # Validate file type
    supported_extensions = {'.pdf', '.html', '.htm', '.md', '.markdown'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {supported_extensions}"
        )
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = Path(temp_file.name)
    
    try:
        # Ingest the file
        result = ingestor.ingest_file(
            file_path=temp_path,
            category=category,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        return IngestResponse(**result)
        
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary file
        if temp_path.exists():
            temp_path.unlink()


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    Returns answer with citations from retrieved documents.
    """
    query_start = time.time()
    
    try:
        # Retrieve documents
        retrieval_result = retriever.retrieve_with_rerank(
            query=request.query,
            top_k=request.top_k,
            category=request.category,
            file_type=request.file_type,
            metadata_filter=request.metadata_filter,
            rerank=request.rerank
        )
        
        retrieval_end = time.time()
        retrieval_latency = retrieval_end - query_start
        
        # Generate answer
        generation_result = generator.generate_answer(
            query=request.query,
            context=retrieval_result['context'],
            documents=retrieval_result['documents']
        )
        
        generation_end = time.time()
        generation_latency = generation_end - retrieval_end
        total_latency = generation_end - query_start
        
        # Log metrics
        perf_logger.log_query(
            query=request.query,
            latency=total_latency,
            chunks_retrieved=retrieval_result['count']
        )
        
        return QueryResponse(
            answer=generation_result['answer'],
            citations=generation_result['citations'],
            context_used=generation_result['context_used'],
            retrieval_latency=retrieval_latency,
            generation_latency=generation_latency,
            total_latency=total_latency,
            chunks_retrieved=retrieval_result['count'],
            token_estimate=generation_result.get('token_estimate', 0)
        )
        
    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get collection statistics."""
    try:
        stats = vector_store.get_collection_stats()
        
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats['error'])
        
        return StatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get recent performance metrics.
    Returns placeholder metrics - in production, this would query a metrics store.
    """
    try:
        # In production, this would return actual metrics from a metrics store
        return MetricsResponse(
            query_latency=0.0,
            retrieval_latency=0.0,
            generation_latency=0.0,
            chunks_retrieved=0,
            token_estimate=0
        )
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collection")
async def reset_collection():
    """Reset the entire collection (use with caution)."""
    try:
        success = vector_store.reset_collection()
        
        if success:
            return {
                "success": True,
                "message": "Collection reset successfully"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to reset collection"
            )
            
    except Exception as e:
        logger.error(f"Error resetting collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint - serve the HTML UI."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Cost-Efficient RAG Application",
        "endpoints": {
            "health": "/health",
            "ingest": "/ingest",
            "query": "/query",
            "stats": "/stats",
            "metrics": "/metrics"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
