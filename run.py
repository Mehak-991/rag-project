"""
Main entry point for the RAG application.
Starts the FastAPI server with proper configuration.
"""

import uvicorn
from app.config import get_settings
from app.logger import logger

settings = get_settings()


def main():
    """Start the FastAPI application."""
    logger.info(f" Starting {settings.app_name} v{settings.app_version}")
    logger.info(f" Host: {settings.host}")
    logger.info(f" Port: {settings.port}")
    logger.info(f" Debug: {settings.debug}")
    logger.info(f" Server running at: http://localhost:{settings.port}")
    
    uvicorn.run(
        "app.api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
