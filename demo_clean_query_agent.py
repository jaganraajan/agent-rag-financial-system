#!/usr/bin/env python3
"""
Demo script showing the cleaned query agent that supports only 2 query types.
"""

from src.agents.query_decomposer import QueryDecomposer
from src.agents.synthesis_engine import SynthesisEngine
import json

def demo_query_agent():
    """Demonstrate the cleaned query agent with the two supported query types."""
    
    decomposer = QueryDecomposer()
    engine = SynthesisEngine()
    
    print("Financial Query Agent - Supporting 2 Query Types Only")
    print("=" * 60)
    print()
    
    # Example 1: Basic Metrics Query
    print("1. BASIC METRICS QUERY")
    print("-" * 30)
    query1 = "What was Microsoft's total revenue in 2023?"
    print(f"Query: {query1}")
    
    result1 = decomposer.decompose_query(query1)
    print(f"Type: {result1['query_type']}")
    print(f"Companies: {result1['companies']}")
    print(f"Metric: {result1['financial_metric']}")
    print(f"Sub-queries: {result1['sub_queries']}")
    
    # Mock RAG data for demo
    mock_rag_results1 = [{
        'question': 'MSFT revenue 2023',
        'results': [{
            'text': 'Microsoft Corporation reported total revenue of $211.9 billion for fiscal year 2023...',
            'metadata': {'company': 'MSFT', 'year': '2023'},
            'similarity': 0.9
        }]
    }]
    
    synthesis_result1 = engine.synthesize_comparative_results(
        query1, result1['sub_queries'], mock_rag_results1, result1['query_type']
    )
    
    print(f"Answer: {synthesis_result1.answer}")
    print(f"Reasoning: {synthesis_result1.reasoning}")
    print()
    
    # Example 2: Cross-Company Comparison Query
    print("2. CROSS-COMPANY COMPARISON QUERY")
    print("-" * 35)
    query2 = "Which company had the highest operating margin in 2023?"
    print(f"Query: {query2}")
    
    result2 = decomposer.decompose_query(query2)
    print(f"Type: {result2['query_type']}")
    print(f"Companies: {result2['companies']}")
    print(f"Metric: {result2['financial_metric']}")
    print(f"Sub-queries: {result2['sub_queries']}")
    
    # Mock RAG data for demo
    mock_rag_results2 = [
        {
            'question': 'MSFT operating margin 2023',
            'results': [{
                'text': 'Microsoft operating margin was 42.1% in fiscal year 2023...',
                'metadata': {'company': 'MSFT', 'year': '2023'},
                'similarity': 0.9
            }]
        },
        {
            'question': 'GOOGL operating margin 2023',
            'results': [{
                'text': 'Google operating margin was 25.2% in 2023...',
                'metadata': {'company': 'GOOGL', 'year': '2023'},
                'similarity': 0.85
            }]
        },
        {
            'question': 'NVDA operating margin 2023',
            'results': [{
                'text': 'NVIDIA operating margin was 32.1% in 2023...',
                'metadata': {'company': 'NVDA', 'year': '2023'},
                'similarity': 0.88
            }]
        }
    ]
    
    synthesis_result2 = engine.synthesize_comparative_results(
        query2, result2['sub_queries'], mock_rag_results2, result2['query_type']
    )
    
    print(f"Answer: {synthesis_result2.answer}")
    print(f"Reasoning: {synthesis_result2.reasoning}")
    print()
    
    # Example 3: Handling unsupported query types
    print("3. UNSUPPORTED QUERY TYPE (handled gracefully)")
    print("-" * 45)
    query3 = "How did NVIDIA's data center revenue grow from 2022 to 2023?"
    print(f"Query: {query3}")
    
    result3 = decomposer.decompose_query(query3)
    print(f"Classified as: {result3['query_type']} (simplified classification)")
    
    # Even with no RAG data, it handles gracefully
    synthesis_result3 = engine.synthesize_comparative_results(
        query3, result3['sub_queries'], [], result3['query_type']
    )
    
    print(f"Answer: {synthesis_result3.answer}")
    print()
    
    print("KEY CHANGES MADE:")
    print("-" * 20)
    print("✓ Removed all mock financial data (demo_financial_data)")
    print("✓ Query decomposer only supports 'simple' and 'comparative' types")
    print("✓ Synthesis engine uses actual RAG results instead of mock data")
    print("✓ Unsupported query types are handled gracefully")
    print("✓ System returns appropriate error messages when no data is available")
    print()
    print("SUPPORTED QUERY TYPES:")
    print("-" * 25)
    print("1. Basic Metrics: Single company, single metric (e.g., revenue)")
    print("2. Cross-Company: Compare metrics across multiple companies")


if __name__ == "__main__":
    demo_query_agent()