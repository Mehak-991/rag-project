"""
Generator module for RAG answering.
Handles LLM interaction with Ollama and generates answers from retrieved context.
"""

import time
from typing import Dict, Any, Optional
from ollama import Client
from app.config import get_settings
from app.logger import logger, perf_logger, estimate_tokens

settings = get_settings()


class RAGGenerator:
    """Generates answers using retrieved context and Ollama LLM."""
    
    def __init__(self):
        """Initialize generator with Ollama client."""
        self.client = Client(host=settings.ollama_base_url)
        self.model = settings.ollama_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        
        logger.info(f"RAGGenerator initialized with model: {self.model}")
    
    def generate_answer(
        self,
        query: str,
        context: str,
        documents: list
    ) -> Dict[str, Any]:
        """
        Generate answer using retrieved context.
        
        Args:
            query: User question
            context: Retrieved context string
            documents: List of retrieved documents for citation
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        try:
            # Generate response using Ollama
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                }
            )
            
            answer = response['response']
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Estimate tokens
            token_estimate = estimate_tokens(answer)
            
            perf_logger.log_generation(latency, token_estimate)
            
            # Prepare citations
            citations = self._prepare_citations(documents)
            
            logger.info(f"Generated answer in {latency:.3f}s")
            
            return {
                'answer': answer,
                'citations': citations,
                'context_used': len(documents) > 0,
                'latency': latency,
                'token_estimate': token_estimate
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                'answer': 'I apologize, but I encountered an error generating the answer.',
                'error': str(e),
                'context_used': False
            }
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM with context and query."""
        if not context or context.strip() == "":
            return f"""Answer the following question based ONLY on the provided context. If the context does not contain enough information to answer the question, respond with: "I don't have enough information in the provided documents."

Question: {query}

Context: No relevant context was found.

Answer:"""
        
        prompt = f"""Answer the following question based ONLY on the provided context. If the context does not contain enough information to answer the question, respond with: "I don't have enough information in the provided documents."

Question: {query}

Context:
{context}

Answer:"""
        
        return prompt
    
    def _prepare_citations(self, documents: list) -> list:
        """Prepare citation information from retrieved documents."""
        citations = []
        for doc in documents:
            citations.append({
                'chunk_id': doc['id'],
                'source': doc['metadata'].get('file_name', 'Unknown'),
                'page': doc['metadata'].get('page', 'N/A'),
                'category': doc['metadata'].get('category', 'N/A'),
                'similarity': doc.get('similarity', doc.get('score', 'N/A'))
            })
        return citations
    
    def check_model_available(self) -> bool:
        """Check if the configured model is available in Ollama."""
        try:
            models = self.client.list()
            available_models = [m['name'] for m in models['models']]
            return self.model in available_models
        except Exception as e:
            logger.error(f"Error checking model availability: {str(e)}")
            return False
    
    def pull_model(self) -> bool:
        """Pull the configured model from Ollama registry."""
        try:
            logger.info(f"Pulling model: {self.model}")
            self.client.pull(self.model)
            logger.info(f"Model {self.model} pulled successfully")
            return True
        except Exception as e:
            logger.error(f"Error pulling model: {str(e)}")
            return False


# Global generator instance
generator = RAGGenerator()


def get_generator() -> RAGGenerator:
    """Get the global generator instance."""
    return generator
