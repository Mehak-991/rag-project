"""
Document ingestion module.
Handles loading, chunking, and processing of PDF, HTML, and Markdown files.
"""

import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pypdf
from bs4 import BeautifulSoup
import markdown
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import get_settings
from app.logger import logger, estimate_tokens, perf_logger
from app.vectorstore import get_vector_store

settings = get_settings()


class DocumentLoader:
    """Loads documents from various file formats."""
    
    @staticmethod
    def load_pdf(file_path: Path) -> str:
        """Load text from PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            logger.info(f"Loaded PDF: {file_path.name}")
            return text
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def load_html(file_path: Path) -> str:
        """Load text from HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator='\n', strip=True)
            logger.info(f"Loaded HTML: {file_path.name}")
            return text
        except Exception as e:
            logger.error(f"Error loading HTML {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def load_markdown(file_path: Path) -> str:
        """Load text from Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
            # Convert markdown to plain text
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            logger.info(f"Loaded Markdown: {file_path.name}")
            return text
        except Exception as e:
            logger.error(f"Error loading Markdown {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def load_document(file_path: Path) -> str:
        """Load document based on file extension."""
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            return DocumentLoader.load_pdf(file_path)
        elif file_ext in ['.html', '.htm']:
            return DocumentLoader.load_html(file_path)
        elif file_ext in ['.md', '.markdown']:
            return DocumentLoader.load_markdown(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")


class DocumentChunker:
    """Chunks documents into smaller pieces for embedding."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """Initialize chunker with configurable parameters."""
        self.chunk_size = chunk_size or settings.default_chunk_size
        self.chunk_overlap = chunk_overlap or settings.default_chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_document(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Chunk document and attach metadata to each chunk.
        
        Args:
            text: Document text
            metadata: Base metadata for the document
            
        Returns:
            List of chunks with metadata
        """
        chunks = self.text_splitter.split_text(text)
        
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'chunk_count': len(chunks)
            })
            chunked_docs.append({
                'text': chunk,
                'metadata': chunk_metadata
            })
        
        logger.info(f"Chunked document into {len(chunks)} chunks")
        return chunked_docs


class DocumentIngestor:
    """Manages document ingestion pipeline."""
    
    def __init__(self):
        """Initialize ingestor."""
        self.loader = DocumentLoader()
        self.chunker = DocumentChunker()
        self.vector_store = get_vector_store()
    
    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as file:
            for chunk in iter(lambda: file.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def ingest_file(
        self,
        file_path: Path,
        category: str = "general",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Ingest a single file into the vector store.
        
        Args:
            file_path: Path to the file
            category: Document category for metadata
            chunk_size: Optional chunk size override
            chunk_overlap: Optional chunk overlap override
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = datetime.now()
        
        try:
            # Compute file hash for duplicate detection
            file_hash = self.compute_file_hash(file_path)
            
            # Check for duplicates
            if self.vector_store.check_duplicate(file_hash):
                logger.warning(f"Duplicate file detected: {file_path.name}")
                return {
                    'success': False,
                    'message': 'Duplicate file already ingested',
                    'file_name': file_path.name,
                    'file_hash': file_hash
                }
            
            # Load document
            text = self.loader.load_document(file_path)
            
            # Update chunker if custom parameters provided
            if chunk_size or chunk_overlap:
                self.chunker = DocumentChunker(chunk_size, chunk_overlap)
            
            # Prepare base metadata
            base_metadata = {
                'source': str(file_path),
                'file_name': file_path.name,
                'file_hash': file_hash,
                'category': category,
                'ingestion_date': datetime.now().isoformat(),
                'file_type': file_path.suffix.lower()
            }
            
            # Chunk document
            chunked_docs = self.chunker.chunk_document(text, base_metadata)
            
            # Prepare for vector store
            documents = [doc['text'] for doc in chunked_docs]
            metadatas = [doc['metadata'] for doc in chunked_docs]
            ids = [
                f"{file_hash}_{doc['metadata']['chunk_index']}"
                for doc in chunked_docs
            ]
            
            # Add to vector store
            self.vector_store.add_documents(documents, metadatas, ids)
            
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()
            
            perf_logger.log_ingestion(
                file_path.name,
                len(chunked_docs),
                latency
            )
            
            return {
                'success': True,
                'message': 'File ingested successfully',
                'file_name': file_path.name,
                'file_hash': file_hash,
                'chunks_added': len(chunked_docs),
                'latency': latency,
                'chunk_size': self.chunker.chunk_size,
                'chunk_overlap': self.chunker.chunk_overlap
            }
            
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {str(e)}")
            return {
                'success': False,
                'message': str(e),
                'file_name': file_path.name
            }
    
    def ingest_directory(
        self,
        directory: Path,
        category: str = "general",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Ingest all supported files from a directory.
        
        Args:
            directory: Path to directory
            category: Document category
            chunk_size: Optional chunk size override
            chunk_overlap: Optional chunk overlap override
            
        Returns:
            List of ingestion results for each file
        """
        supported_extensions = {'.pdf', '.html', '.htm', '.md', '.markdown'}
        results = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                result = self.ingest_file(
                    file_path,
                    category,
                    chunk_size,
                    chunk_overlap
                )
                results.append(result)
        
        successful = sum(1 for r in results if r['success'])
        logger.info(
            f"Ingested {successful}/{len(results)} files from {directory}"
        )
        
        return results


# Global ingestor instance
ingestor = DocumentIngestor()


def get_ingestor() -> DocumentIngestor:
    """Get the global ingestor instance."""
    return ingestor
