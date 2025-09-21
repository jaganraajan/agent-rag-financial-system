#!/usr/bin/env python3
"""
Test script for the 5 new query types mentioned in the problem statement.
"""

from src.agents.query_decomposer import QueryDecomposer
from src.agents.synthesis_engine import SynthesisEngine
import json

def test_new_query_types():
    """Test all 5 query types from the problem statement."""
    decomposer = QueryDecomposer()
    engine = SynthesisEngine()
    
    # The 5 query types from the problem statement
    test_queries = [
        "What was Microsoft's total revenue in 2023?",  # Basic Metrics
        "How did NVIDIA's data center revenue grow from 2022 to 2023?",  # YoY Comparison
        "Which company had the highest operating margin in 2023?",  # Cross-Company
        "What percentage of Google's revenue came from cloud in 2023?",  # Segment Analysis
        "Compare AI investments mentioned by all three companies in their 2024 10-Ks"  # AI Strategy
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {query}")
        print(f"{'='*80}")
        
        # Decompose query
        decomp_result = decomposer.decompose_query(query)
        print(f"Query Type: {decomp_result['query_type']}")
        print(f"Companies: {decomp_result['companies']}")
        print(f"Years: {decomp_result['years']}")
        print(f"Financial Metric: {decomp_result['financial_metric']}")
        print(f"Sub-queries: {decomp_result['sub_queries']}")
        
        # Mock RAG results for testing
        mock_rag_results = []
        for sub_query in decomp_result['sub_queries']:
            mock_rag_results.append({
                'question': sub_query,
                'results': [
                    {
                        'text': f'Mock financial data for {sub_query}...',
                        'metadata': {'company': 'MSFT', 'year': '2023'},
                        'similarity': 0.9
                    }
                ]
            })
        
        # Synthesize results
        synthesis_result = engine.synthesize_comparative_results(
            query, 
            decomp_result['sub_queries'], 
            mock_rag_results,
            decomp_result['query_type']
        )
        
        # Format and display results
        json_output = engine.format_json_output(synthesis_result)
        print("\nSynthesis Result:")
        print(json.dumps(json_output, indent=2))

if __name__ == "__main__":
    test_new_query_types()