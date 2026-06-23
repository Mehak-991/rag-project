"""
Evaluation module for RAG system.
Implements retrieval and answer quality metrics.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import ndcg_score
from app.config import get_settings
from app.logger import logger
from app.retriever import get_retriever
from app.generator import get_generator

settings = get_settings()


class RetrievalMetrics:
    """Calculates retrieval quality metrics."""
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top documents to consider
            
        Returns:
            Recall@K score
        """
        retrieved_at_k = retrieved_ids[:k]
        relevant_retrieved = set(retrieved_at_k) & set(relevant_ids)
        return len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0.0
    
    @staticmethod
    def hit_rate(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        Calculate Hit Rate.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            
        Returns:
            Hit Rate score
        """
        return 1.0 if set(retrieved_ids) & set(relevant_ids) else 0.0
    
    @staticmethod
    def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            
        Returns:
            MRR score
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_scores: List[float], relevant_scores: List[float], k: int) -> float:
        """
        Calculate nDCG@K.
        
        Args:
            retrieved_scores: List of relevance scores for retrieved documents
            relevant_scores: List of ideal relevance scores
            k: Number of top documents to consider
            
        Returns:
            nDCG@K score
        """
        if len(retrieved_scores) == 0:
            return 0.0
        
        # Pad arrays to same length
        max_len = max(len(retrieved_scores), len(relevant_scores))
        retrieved_padded = retrieved_scores[:k] + [0] * (max(k, len(retrieved_scores)) - len(retrieved_scores[:k]))
        relevant_padded = relevant_scores[:k] + [0] * (max(k, len(relevant_scores)) - len(relevant_scores[:k]))
        
        try:
            return ndcg_score([relevant_padded], [retrieved_padded], k=k)
        except:
            return 0.0
    
    @staticmethod
    def context_precision(retrieved_docs: List[Dict], relevant_ids: List[str]) -> float:
        """
        Calculate Context Precision.
        Measures the fraction of retrieved documents that are relevant.
        
        Args:
            retrieved_docs: List of retrieved documents with metadata
            relevant_ids: List of relevant document IDs
            
        Returns:
            Context Precision score
        """
        if not retrieved_docs:
            return 0.0
        
        retrieved_ids = [doc['id'] for doc in retrieved_docs]
        relevant_retrieved = set(retrieved_ids) & set(relevant_ids)
        return len(relevant_retrieved) / len(retrieved_ids)


class AnswerMetrics:
    """Calculates answer quality metrics."""
    
    @staticmethod
    def faithfulness(answer: str, context: str) -> float:
        """
        Calculate Faithfulness score.
        Measures how well the answer is grounded in the context.
        
        This is a simplified implementation. In production, use an LLM to evaluate.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Faithfulness score (0-1)
        """
        # Simple heuristic: check if answer contains information from context
        if not context or not answer:
            return 0.0
        
        # Extract key terms from answer and check if they appear in context
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Calculate overlap
        overlap = answer_words & context_words
        if not answer_words:
            return 0.0
        
        return len(overlap) / len(answer_words)
    
    @staticmethod
    def groundedness(answer: str, context: str) -> float:
        """
        Calculate Groundedness score.
        Similar to faithfulness but focuses on factual accuracy.
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Groundedness score (0-1)
        """
        # For now, use the same implementation as faithfulness
        return AnswerMetrics.faithfulness(answer, context)
    
    @staticmethod
    def answer_relevance(answer: str, query: str) -> float:
        """
        Calculate Answer Relevance.
        Measures how relevant the answer is to the query.
        
        Args:
            answer: Generated answer
            query: Original query
            
        Returns:
            Answer Relevance score (0-1)
        """
        if not answer or not query:
            return 0.0
        
        # Simple heuristic: check if answer contains query terms
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        overlap = query_words & answer_words
        if not query_words:
            return 0.5  # Neutral score if query has no meaningful words
        
        return len(overlap) / len(query_words)


class RAGEvaluator:
    """Evaluates RAG system performance."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.retriever = get_retriever()
        self.generator = get_generator()
        self.retrieval_metrics = RetrievalMetrics()
        self.answer_metrics = AnswerMetrics()
    
    def evaluate_query(
        self,
        query: str,
        relevant_ids: List[str],
        k_values: List[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single query.
        
        Args:
            query: Query string
            relevant_ids: List of relevant document IDs
            k_values: List of K values for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        if k_values is None:
            k_values = settings.k_values
        
        results = {
            'query': query,
            'relevant_count': len(relevant_ids),
            'metrics': {}
        }
        
        # Retrieve documents
        retrieval_result = self.retriever.retrieve_with_rerank(
            query=query,
            top_k=max(k_values),
            rerank=True
        )
        
        retrieved_docs = retrieval_result['documents']
        retrieved_ids = [doc['id'] for doc in retrieved_docs]
        retrieved_scores = [doc.get('similarity', 0) for doc in retrieved_docs]
        
        # Calculate retrieval metrics for each K
        for k in k_values:
            results['metrics'][f'recall@{k}'] = self.retrieval_metrics.recall_at_k(
                retrieved_ids, relevant_ids, k
            )
            results['metrics'][f'ndcg@{k}'] = self.retrieval_metrics.ndcg_at_k(
                retrieved_scores, [1] * len(relevant_ids), k
            )
        
        results['metrics']['hit_rate'] = self.retrieval_metrics.hit_rate(
            retrieved_ids, relevant_ids
        )
        results['metrics']['mrr'] = self.retrieval_metrics.mrr(
            retrieved_ids, relevant_ids
        )
        results['metrics']['context_precision'] = self.retrieval_metrics.context_precision(
            retrieved_docs, relevant_ids
        )
        
        # Generate answer and calculate answer metrics
        generation_result = self.generator.generate_answer(
            query=query,
            context=retrieval_result['context'],
            documents=retrieved_docs
        )
        
        answer = generation_result['answer']
        context = retrieval_result['context']
        
        results['metrics']['faithfulness'] = self.answer_metrics.faithfulness(
            answer, context
        )
        results['metrics']['groundedness'] = self.answer_metrics.groundedness(
            answer, context
        )
        results['metrics']['answer_relevance'] = self.answer_metrics.answer_relevance(
            answer, query
        )
        
        results['answer'] = answer
        results['retrieved_count'] = len(retrieved_docs)
        
        return results
    
    def evaluate_dataset(
        self,
        questions_file: Path,
        output_file: Path = None
    ) -> Dict[str, Any]:
        """
        Evaluate a dataset of questions.
        
        Args:
            questions_file: Path to JSON file with questions and relevant IDs
            output_file: Path to save results
            
        Returns:
            Dictionary with aggregated results
        """
        # Load questions
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)
        
        all_results = []
        aggregated_metrics = {}
        
        for item in questions_data:
            query = item['query']
            relevant_ids = item.get('relevant_ids', [])
            
            result = self.evaluate_query(query, relevant_ids)
            all_results.append(result)
        
        # Aggregate metrics
        metric_keys = all_results[0]['metrics'].keys()
        for key in metric_keys:
            values = [r['metrics'][key] for r in all_results]
            aggregated_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        final_results = {
            'total_queries': len(all_results),
            'aggregated_metrics': aggregated_metrics,
            'individual_results': all_results
        }
        
        # Save results if output file provided
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
        
        return final_results


# Global evaluator instance
evaluator = RAGEvaluator()


def get_evaluator() -> RAGEvaluator:
    """Get the global evaluator instance."""
    return evaluator
