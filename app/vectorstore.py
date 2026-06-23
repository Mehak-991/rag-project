"""
Vector store module for managing ChromaDB operations.
Handles persistent storage, metadata filtering, and collection management.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import Optional, Dict, Any, List
from app.config import get_settings
from app.logger import logger

settings = get_settings()


class VectorStore:
    """Manages ChromaDB vector store operations."""
    
    def __init__(self):
        """Initialize ChromaDB client and collection."""
        self.persist_dir = Path(settings.chroma_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model_name,
            device=settings.embedding_device
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(
            f"VectorStore initialized | Collection: {settings.chroma_collection_name}"
        )
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=settings.chroma_collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection: {settings.chroma_collection_name}")
            return collection
        except Exception:
            # Create new collection if doesn't exist
            collection = self.client.create_collection(
                name=settings.chroma_collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "RAG document collection"}
            )
            logger.info(f"Created new collection: {settings.chroma_collection_name}")
            return collection
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document text chunks
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs
        """
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to collection")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions
            
        Returns:
            Dictionary containing query results
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            logger.info(f"Query returned {len(results['ids'][0])} results")
            return results
        except Exception as e:
            logger.error(f"Error querying collection: {str(e)}")
            raise
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by its ID."""
        try:
            results = self.collection.get(ids=[doc_id])
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'document': results['documents'][0],
                    'metadata': results['metadatas'][0]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting document by ID: {str(e)}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by its ID."""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            return {
                'collection_name': settings.chroma_collection_name,
                'document_count': count,
                'embedding_model': settings.embedding_model_name,
                'persist_directory': str(self.persist_dir)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {'error': str(e)}
    
    def check_duplicate(self, file_hash: str) -> bool:
        """
        Check if a document with the given file hash already exists.
        
        Args:
            file_hash: Hash of the file to check
            
        Returns:
            True if duplicate exists, False otherwise
        """
        try:
            results = self.collection.get(
                where={"file_hash": file_hash}
            )
            return len(results['ids']) > 0
        except Exception as e:
            logger.error(f"Error checking duplicate: {str(e)}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset/delete the entire collection."""
        try:
            self.client.delete_collection(name=settings.chroma_collection_name)
            self.collection = self._get_or_create_collection()
            logger.info(f"Reset collection: {settings.chroma_collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            return False


# Global vector store instance
vector_store = VectorStore()


def get_vector_store() -> VectorStore:
    """Get the global vector store instance."""
    return vector_store
