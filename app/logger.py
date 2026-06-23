"""
Logging module for RAG application.
Provides structured logging with performance metrics tracking.
"""

import sys
import time
from loguru import logger
from pathlib import Path
from typing import Optional, Dict, Any
from functools import wraps
from app.config import get_settings

settings = get_settings()


class PerformanceLogger:
    """Logger for tracking performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
    
    def log_query(self, query: str, latency: float, chunks_retrieved: int):
        """Log query performance metrics."""
        logger.info(
            f"Query: {query[:100]}... | "
            f"Latency: {latency:.3f}s | "
            f"Chunks: {chunks_retrieved}"
        )
    
    def log_retrieval(self, latency: float, chunks_retrieved: int):
        """Log retrieval performance metrics."""
        logger.info(
            f"Retrieval | Latency: {latency:.3f}s | Chunks: {chunks_retrieved}"
        )
    
    def log_generation(self, latency: float, token_estimate: int):
        """Log generation performance metrics."""
        logger.info(
            f"Generation | Latency: {latency:.3f}s | Tokens: {token_estimate}"
        )
    
    def log_ingestion(self, file_name: str, chunks_added: int, latency: float):
        """Log ingestion performance metrics."""
        logger.info(
            f"Ingestion | File: {file_name} | "
            f"Chunks: {chunks_added} | Latency: {latency:.3f}s"
        )


def setup_logger():
    """Configure loguru logger with file and console handlers."""
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=settings.log_level,
        colorize=True
    )
    
    # File handler for all logs
    logger.add(
        log_dir / "app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=settings.log_level,
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        compression="zip"
    )
    
    # Error log file
    logger.add(
        log_dir / "error_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        compression="zip"
    )
    
    return logger


# Initialize logger
app_logger = setup_logger()
perf_logger = PerformanceLogger()


def log_execution_time(func):
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
        return result
    return wrapper


def estimate_tokens(text: str) -> int:
    """Estimate token count using simple character-based heuristic."""
    # Rough estimate: ~4 characters per token for English text
    return len(text) // 4
