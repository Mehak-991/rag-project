"""
Retrieval module for searching and ranking documents.
Handles similarity search, metadata filtering, and context reranking.
"""

import time
from typing import List, Dict, Any, Optional
from app.config import get_settings
from app.logger import logger, perf_logger
from app.vectorstore import get_vector_store

settings = get_settings()


class DocumentRetriever:
    """Retrieves relevant documents from the vector store."""
    
    def __init__(self):
        """Initialize retriever."""
        self.vector_store = get_vector_store()
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
        file_type: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            category: Filter by document category
            file_type: Filter by file type
            metadata_filter: Additional metadata filters
            
        Returns:
            List of retrieved documents with metadata
        """
        start_time = time.time()
        
        top_k = top_k or settings.default_top_k
        
        # Build metadata filter
        where = {}
        if category:
            where['category'] = category
        if file_type:
            where['file_type'] = file_type
        if metadata_filter:
            where.update(metadata_filter)
        
        # Remove empty filter
        where = where if where else None
        
        try:
            # Query vector store
            results = self.vector_store.query(
                query_text=query,
                n_results=top_k,
                where=where
            )
            
            # Format results
            documents = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    documents.append({
                        'id': results['ids'][0][i],
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': results['distances'][0][i] if 'distances' in results else None
                    })
            
            end_time = time.time()
            latency = end_time - start_time
            
            perf_logger.log_retrieval(latency, len(documents))
            
            logger.info(
                f"Retrieved {len(documents)} documents for query: {query[:50]}..."
            )
            
            return documents
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise
    
    def rerank_context(
        self,
        documents: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents based on relevance to query.
        Simple reranking based on score and keyword matching.
        
        Args:
            documents: List of retrieved documents
            query: Original query
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return documents
        
        # Normalize scores if available
        for doc in documents:
            if doc.get('score') is not None:
                # Convert distance to similarity (lower distance = higher similarity)
                doc['similarity'] = 1 - min(doc['score'], 1.0)
            else:
                doc['similarity'] = 0.5
        
        # Sort by similarity score
        reranked = sorted(
            documents,
            key=lambda x: x.get('similarity', 0),
            reverse=True
        )
        
        logger.info(f"Reranked {len(reranked)} documents")
        return reranked
    
    def format_context(
        self,
        documents: List[Dict[str, Any]],
        max_context_length: int = 4000
    ) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of retrieved documents
            max_context_length: Maximum context length in characters
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            source = doc['metadata'].get('file_name', 'Unknown')
            chunk_id = doc['id']
            text = doc['text']
            
            part = f"[Source: {source} | Chunk ID: {chunk_id}]\n{text}\n"
            
            if current_length + len(part) > max_context_length:
                break
            
            context_parts.append(part)
            current_length += len(part)
        
        context = "\n---\n".join(context_parts)
        logger.info(f"Formatted context with {len(documents)} chunks, length: {current_length}")
        
        return context
    
    def retrieve_with_rerank(
        self,
        query: str,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
        file_type: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        rerank: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve documents with optional reranking.
        
        Args:
            query: Search query
            top_k: Number of results
            category: Filter by category
            file_type: Filter by file type
            metadata_filter: Additional filters
            rerank: Whether to rerank results
            
        Returns:
            Dictionary with documents and metadata
        """
        # Retrieve documents
        documents = self.retrieve(
            query=query,
            top_k=top_k,
            category=category,
            file_type=file_type,
            metadata_filter=metadata_filter
        )
        
        # Rerank if requested
        if rerank:
            documents = self.rerank_context(documents, query)
        
        # Format context
        context = self.format_context(documents)
        
        return {
            'documents': documents,
            'context': context,
            'count': len(documents)
        }


# Global retriever instance
retriever = DocumentRetriever()


def get_retriever() -> DocumentRetriever:
    """Get the global retriever instance."""
    return retriever
