"""
Evaluation runner script.
Runs evaluation on the RAG system using questions.json.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluator import get_evaluator
from app.config import get_settings

settings = get_settings()


def main():
    """Run evaluation on the RAG system."""
    print("Starting RAG system evaluation...")
    
    # Initialize evaluator
    evaluator = get_evaluator()
    
    # Define file paths
    questions_file = Path(settings.evaluation_dir) / "questions.json"
    results_file = Path(settings.evaluation_dir) / "results.json"
    
    # Check if questions file exists
    if not questions_file.exists():
        print(f"Questions file not found: {questions_file}")
        print("Please create questions.json with evaluation queries.")
        return
    
    # Run evaluation
    print(f"Loading questions from {questions_file}")
    results = evaluator.evaluate_dataset(
        questions_file=questions_file,
        output_file=results_file
    )
    
    # Print aggregated results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total queries evaluated: {results['total_queries']}")
    print("\nAggregated Metrics:")
    print("-" * 50)
    
    for metric_name, metric_values in results['aggregated_metrics'].items():
        print(f"\n{metric_name}:")
        print(f"  Mean: {metric_values['mean']:.4f}")
        print(f"  Std:  {metric_values['std']:.4f}")
        print(f"  Min:  {metric_values['min']:.4f}")
        print(f"  Max:  {metric_values['max']:.4f}")
    
    print(f"\nResults saved to: {results_file}")
    print("="*50)


if __name__ == "__main__":
    main()
