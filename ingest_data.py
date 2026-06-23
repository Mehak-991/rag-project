"""
Script to ingest all documents from the data directory into the RAG system.
"""

from pathlib import Path
from app.ingest import get_ingestor
from app.logger import logger

def main():
    """Ingest all documents from the data directory."""
    data_dir = Path("./data")
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    ingestor = get_ingestor()
    
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
