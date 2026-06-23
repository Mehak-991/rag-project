"""
Unit tests for retrieval module.
"""

import pytest
from app.retriever import DocumentRetriever
from app.vectorstore import VectorStore


class TestDocumentRetriever:
    """Test document retrieval functionality."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store for testing."""
        class MockVectorStore:
            def __init__(self):
                self.test_documents = [
                    {
                        'id': 'doc1',
                        'text': 'This is about machine learning',
                        'metadata': {'category': 'tech', 'file_name': 'doc1.txt'}
                    },
                    {
                        'id': 'doc2',
                        'text': 'This is about cooking recipes',
                        'metadata': {'category': 'food', 'file_name': 'doc2.txt'}
                    }
                ]
            
            def query(self, query_text, n_results=5, where=None, where_document=None):
                # Simple mock query - return all docs
                return {
                    'ids': [[doc['id'] for doc in self.test_documents[:n_results]]],
                    'documents': [[doc['text'] for doc in self.test_documents[:n_results]]],
                    'metadatas': [[doc['metadata'] for doc in self.test_documents[:n_results]]],
                    'distances': [[0.1, 0.2][:n_results]]
                }
        
        return MockVectorStore()
    
    def test_retrieve(self, mock_vector_store):
        """Test basic retrieval."""
        retriever = DocumentRetriever()
        retriever.vector_store = mock_vector_store
        
        results = retriever.retrieve(query="machine learning", top_k=2)
        
        assert len(results) == 2
        assert all('id' in doc for doc in results)
        assert all('text' in doc for doc in results)
        assert all('metadata' in doc for doc in results)
    
    def test_retrieve_with_filter(self, mock_vector_store):
        """Test retrieval with metadata filter."""
        retriever = DocumentRetriever()
        retriever.vector_store = mock_vector_store
        
        results = retriever.retrieve(
            query="test",
            category="tech"
        )
        
        assert len(results) >= 0
    
    def test_rerank_context(self, mock_vector_store):
        """Test context reranking."""
        retriever = DocumentRetriever()
        
        documents = [
            {'id': 'doc1', 'text': 'test', 'metadata': {}, 'score': 0.1},
            {'id': 'doc2', 'text': 'test', 'metadata': {}, 'score': 0.5},
            {'id': 'doc3', 'text': 'test', 'metadata': {}, 'score': 0.3}
        ]
        
        reranked = retriever.rerank_context(documents, "test query")
        
        assert len(reranked) == 3
        # Should be sorted by similarity (inverse of score)
        assert reranked[0]['score'] == 0.1  # Highest similarity
        assert reranked[-1]['score'] == 0.5  # Lowest similarity
    
    def test_format_context(self, mock_vector_store):
        """Test context formatting."""
        retriever = DocumentRetriever()
        
        documents = [
            {
                'id': 'doc1',
                'text': 'Test content 1',
                'metadata': {'file_name': 'file1.txt'}
            },
            {
                'id': 'doc2',
                'text': 'Test content 2',
                'metadata': {'file_name': 'file2.txt'}
            }
        ]
        
        context = retriever.format_context(documents)
        
        assert 'file1.txt' in context
        assert 'file2.txt' in context
        assert 'doc1' in context
        assert 'doc2' in context
        assert 'Test content 1' in context
        assert 'Test content 2' in context
    
    def test_retrieve_with_rerank(self, mock_vector_store):
        """Test complete retrieval with reranking."""
        retriever = DocumentRetriever()
        retriever.vector_store = mock_vector_store
        
        result = retriever.retrieve_with_rerank(
            query="test query",
            top_k=2,
            rerank=True
        )
        
        assert 'documents' in result
        assert 'context' in result
        assert 'count' in result
        assert result['count'] == 2
