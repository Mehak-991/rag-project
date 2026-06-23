"""
Reset the vector store and re-ingest all documents.
"""

from app.vectorstore import get_vector_store
from app.ingest import get_ingestor
from pathlib import Path
from app.logger import logger

def main():
    """Reset collection and re-ingest documents."""
    vector_store = get_vector_store()
    ingestor = get_ingestor()
    
    # Reset collection
    logger.info("Resetting vector store collection...")
    if vector_store.reset_collection():
        logger.info("Collection reset successfully")
    else:
        logger.error("Failed to reset collection")
        return
    
    # Re-ingest documents
    data_dir = Path("./data")
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    logger.info(f"Starting ingestion from {data_dir}")
    results = ingestor.ingest_directory(data_dir, category="general")
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    logger.info(f"Ingestion complete: {successful} successful, {failed} failed")
    
    for result in results:
        if result['success']:
            logger.info(f"✓ {result['file_name']}: {result['chunks_added']} chunks")
        else:
            logger.error(f"✗ {result['file_name']}: {result['message']}")

if __name__ == "__main__":
    main()
