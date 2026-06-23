"""
Unit tests for document ingestion module.
"""

import pytest
import tempfile
from pathlib import Path
from app.ingest import DocumentLoader, DocumentChunker, DocumentIngestor
from app.vectorstore import VectorStore


class TestDocumentLoader:
    """Test document loading functionality."""
    
    def test_load_markdown(self):
        """Test loading markdown files."""
        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Document\n\nThis is a test.")
            temp_path = Path(f.name)
        
        try:
            loader = DocumentLoader()
            text = loader.load_markdown(temp_path)
            assert "Test Document" in text
            assert "test" in text.lower()
        finally:
            temp_path.unlink()
    
    def test_load_html(self):
        """Test loading HTML files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write("<html><body><h1>Test</h1><p>Content</p></body></html>")
            temp_path = Path(f.name)
        
        try:
            loader = DocumentLoader()
            text = loader.load_html(temp_path)
            assert "Test" in text
            assert "Content" in text
        finally:
            temp_path.unlink()
    
    def test_unsupported_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)
        
        try:
            loader = DocumentLoader()
            with pytest.raises(ValueError):
                loader.load_document(temp_path)
        finally:
            temp_path.unlink()


class TestDocumentChunker:
    """Test document chunking functionality."""
    
    def test_chunk_document(self):
        """Test document chunking."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        text = "This is a test document. " * 20
        
        metadata = {
            'source': 'test.txt',
            'file_name': 'test.txt',
            'category': 'test'
        }
        
        chunks = chunker.chunk_document(text, metadata)
        
        assert len(chunks) > 1
        assert all('text' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
        assert all('chunk_index' in chunk['metadata'] for chunk in chunks)
    
    def test_default_chunk_size(self):
        """Test default chunk size."""
        chunker = DocumentChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200


class TestDocumentIngestor:
    """Test document ingestion pipeline."""
    
    def test_compute_file_hash(self):
        """Test file hash computation."""
        ingestor = DocumentIngestor()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)
        
        try:
            hash1 = ingestor.compute_file_hash(temp_path)
            hash2 = ingestor.compute_file_hash(temp_path)
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex length
        finally:
            temp_path.unlink()
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store for testing."""
        class MockVectorStore:
            def __init__(self):
                self.documents = {}
            
            def check_duplicate(self, file_hash):
                return file_hash in self.documents
            
            def add_documents(self, documents, metadatas, ids):
                for i, doc_id in enumerate(ids):
                    self.documents[doc_id] = {
                        'text': documents[i],
                        'metadata': metadatas[i]
                    }
        
        return MockVectorStore()
    
    def test_ingest_file_duplicate(self, mock_vector_store):
        """Test duplicate file detection."""
        ingestor = DocumentIngestor()
        ingestor.vector_store = mock_vector_store
        
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test\nContent")
            temp_path = Path(f.name)
        
        try:
            # First ingestion
            result1 = ingestor.ingest_file(temp_path, category='test')
            assert result1['success'] is True
            
            # Second ingestion (duplicate)
            result2 = ingestor.ingest_file(temp_path, category='test')
            assert result2['success'] is False
            assert 'Duplicate' in result2['message']
        finally:
            temp_path.unlink()
