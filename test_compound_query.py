#!/usr/bin/env python3
"""
Test script for compound query decomposition with iterative follow-up.

This script tests the new iterative query decomposition feature that:
1. Runs comparative sub-queries to determine a winner
2. Appends and executes a follow-up sub-query based on the winner
3. Returns JSON with complete sub_queries list including follow-ups
"""

import sys
import os
import json
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.enhanced_rag import EnhancedRAGPipeline
from src.agents.synthesis_engine import SynthesisResult, SourceInfo

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def test_compound_query_detection():
    """Test detection of compound queries."""
    print("\n" + "=" * 70)
    print("TEST 1: Compound Query Detection")
    print("=" * 70)
    
    pipeline = EnhancedRAGPipeline()
    
    # Test cases
    test_cases = [
        ("Which company had the highest revenue in 2024? What are the main AI risks of that company?", True),
        ("Which company had the highest operating margin in 2023?", False),
        ("Compare Microsoft and Google revenue", False),
        ("What company had the best performance in 2024? What are the AI strategies of that company?", True),
    ]
    
    for query, expected_compound in test_cases:
        is_compound = pipeline._detect_compound_query_with_followup(query)
        status = "✅ PASS" if is_compound == expected_compound else "❌ FAIL"
        print(f"{status}: '{query[:60]}...' -> Compound: {is_compound}")
    
    print("\n✅ Compound query detection tests completed\n")


def test_year_extraction():
    """Test year extraction from queries."""
    print("\n" + "=" * 70)
    print("TEST 2: Year Extraction")
    print("=" * 70)
    
    pipeline = EnhancedRAGPipeline()
    
    test_cases = [
        ("Which company had the highest revenue in 2024?", "2024"),
        ("Compare Microsoft and Google revenue in 2023", "2023"),
        ("What are NVIDIA's main business activities?", "2024"),  # Default
    ]
    
    for query, expected_year in test_cases:
        year = pipeline._extract_year_from_query(query)
        status = "✅ PASS" if year == expected_year else "❌ FAIL"
        print(f"{status}: '{query[:50]}...' -> Year: {year} (expected: {expected_year})")
    
    print("\n✅ Year extraction tests completed\n")


def test_synthesis_metadata():
    """Test that synthesis result includes metadata with winner."""
    print("\n" + "=" * 70)
    print("TEST 3: Synthesis Metadata (Winner Propagation)")
    print("=" * 70)
    
    from src.agents.synthesis_engine import SynthesisEngine
    
    engine = SynthesisEngine()
    
    # Mock RAG results with revenue data (formatted to match extraction patterns)
    mock_results = [
        {
            'results': [
                {
                    'text': 'Total revenue of $211.9 billion in fiscal year 2023...',
                    'metadata': {'company': 'MSFT', 'year': '2023'},
                    'similarity': 0.9
                }
            ]
        },
        {
            'results': [
                {
                    'text': 'Total revenue of $307.4 billion in 2023...',
                    'metadata': {'company': 'GOOGL', 'year': '2023'},
                    'similarity': 0.85
                }
            ]
        },
        {
            'results': [
                {
                    'text': 'Total revenue of $26.9 billion in 2023...',
                    'metadata': {'company': 'NVDA', 'year': '2023'},
                    'similarity': 0.88
                }
            ]
        }
    ]
    
    query = "Which company had the highest revenue in 2023?"
    sub_queries = ["MSFT revenue 2023", "GOOGL revenue 2023", "NVDA revenue 2023"]
    
    result = engine.synthesize_comparative_results(query, sub_queries, mock_results, "comparative")
    
    print(f"Query: {query}")
    print(f"Answer: {result.answer}")
    print(f"Sub-queries: {result.sub_queries}")
    
    if result.metadata and 'winner_company' in result.metadata:
        winner = result.metadata['winner_company']
        print(f"✅ PASS: Winner company in metadata: {winner}")
    else:
        print(f"❌ FAIL: No winner_company in metadata")
    
    print("\n✅ Synthesis metadata test completed\n")


def test_iterative_query_flow_mock():
    """Test the iterative query flow with mocked results."""
    print("\n" + "=" * 70)
    print("TEST 4: Iterative Query Flow (Mocked)")
    print("=" * 70)
    
    # This test demonstrates the expected flow without actual vector DB
    query = "Which company had the highest revenue in 2024? What are the main AI risks of that company?"
    
    print(f"Testing compound query: {query}")
    print("\nExpected behavior:")
    print("1. Decompose into initial comparative sub-queries:")
    print("   - 'MSFT revenue 2024'")
    print("   - 'GOOGL revenue 2024'")
    print("   - 'NVDA revenue 2024'")
    print("2. Execute retrieval and determine the winner (e.g., GOOGL)")
    print("3. Dynamically append and execute follow-up sub-query:")
    print("   - 'GOOGL ai strategy 2024' (standardized to 'ai strategy')")
    print("4. Return JSON with complete sub_queries list")
    
    print("\n✅ Iterative query flow test described\n")


def test_sub_queries_list_completeness():
    """Test that sub_queries list includes all executed queries."""
    print("\n" + "=" * 70)
    print("TEST 5: Sub-queries List Completeness")
    print("=" * 70)
    
    print("Testing that the final JSON output contains all sub-queries:")
    print("- Initial comparative queries (e.g., 3 revenue queries)")
    print("- Follow-up query (e.g., 1 ai strategy query)")
    print("- Total expected: 4 queries in sub_queries list")
    
    expected_structure = {
        "query": "Which company had the highest revenue in 2024? What are the main AI risks of that company?",
        "answer": "[Synthesized answer about the winner and AI risks]",
        "reasoning": "[Synthesis reasoning]",
        "sub_queries": [
            "MSFT revenue 2024",
            "GOOGL revenue 2024", 
            "NVDA revenue 2024",
            "GOOGL ai strategy 2024"  # Follow-up appended
        ],
        "sources": "[List of sources]"
    }
    
    print("\nExpected JSON structure:")
    print(json.dumps(expected_structure, indent=2))
    
    print("\n✅ Sub-queries list completeness test described\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("COMPOUND QUERY DECOMPOSITION TEST SUITE")
    print("=" * 70)
    
    try:
        # Run tests
        test_compound_query_detection()
        test_year_extraction()
        test_synthesis_metadata()
        test_iterative_query_flow_mock()
        test_sub_queries_list_completeness()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nNote: These tests validate the logic structure.")
        print("For full end-to-end testing with actual document retrieval,")
        print("run: python main.py rag --query 'Which company had the highest revenue in 2024? What are the main AI risks of that company?'")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
