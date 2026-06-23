"""
Evaluation metrics module.
Provides standalone metric calculation functions.
"""

import numpy as np
from typing import List, Dict, Any


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Calculate Recall@K metric."""
    retrieved_at_k = retrieved_ids[:k]
    relevant_retrieved = set(retrieved_at_k) & set(relevant_ids)
    return len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0.0


def hit_rate(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Calculate Hit Rate metric."""
    return 1.0 if set(retrieved_ids) & set(relevant_ids) else 0.0


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Calculate Mean Reciprocal Rank metric."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_scores: List[float], relevant_scores: List[float], k: int) -> float:
    """Calculate nDCG@K metric."""
    if len(retrieved_scores) == 0:
        return 0.0
    
    max_len = max(len(retrieved_scores), len(relevant_scores))
    retrieved_padded = retrieved_scores[:k] + [0] * (max(k, len(retrieved_scores)) - len(retrieved_scores[:k]))
    relevant_padded = relevant_scores[:k] + [0] * (max(k, len(relevant_scores)) - len(relevant_scores[:k]))
    
    try:
        from sklearn.metrics import ndcg_score
        return ndcg_score([relevant_padded], [retrieved_padded], k=k)
    except:
        return 0.0


def context_precision(retrieved_docs: List[Dict], relevant_ids: List[str]) -> float:
    """Calculate Context Precision metric."""
    if not retrieved_docs:
        return 0.0
    
    retrieved_ids = [doc['id'] for doc in retrieved_docs]
    relevant_retrieved = set(retrieved_ids) & set(relevant_ids)
    return len(relevant_retrieved) / len(retrieved_ids)


def faithfulness(answer: str, context: str) -> float:
    """Calculate Faithfulness metric."""
    if not context or not answer:
        return 0.0
    
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    
    overlap = answer_words & context_words
    if not answer_words:
        return 0.0
    
    return len(overlap) / len(answer_words)


def groundedness(answer: str, context: str) -> float:
    """Calculate Groundedness metric."""
    return faithfulness(answer, context)


def answer_relevance(answer: str, query: str) -> float:
    """Calculate Answer Relevance metric."""
    if not answer or not query:
        return 0.0
    
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    
    overlap = query_words & answer_words
    if not query_words:
        return 0.5
    
    return len(overlap) / len(query_words)


def aggregate_metrics(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across multiple results."""
    if not results:
        return {}
    
    metric_keys = results[0].keys()
    aggregated = {}
    
    for key in metric_keys:
        values = [r[key] for r in results if key in r]
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    return aggregated
