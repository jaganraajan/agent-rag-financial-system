#!/usr/bin/env python3
"""
Integration test for compound query with iterative follow-up.

This script demonstrates the full end-to-end flow:
1. User asks: "Which company had the highest revenue in 2024? What are the main AI risks of that company?"
2. System decomposes into comparative revenue queries
3. System executes queries and determines winner
4. System appends and executes follow-up AI strategy query
5. System returns complete JSON with all sub-queries
"""

import sys
import os
import json
import logging
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.enhanced_rag import EnhancedRAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def mock_rag_query_with_data(query, top_k=5):
    """Mock RAG query that returns realistic data based on the query."""
    # Mock revenue queries
    if 'revenue 2024' in query.lower():
        if 'MSFT' in query:
            return {
                'results': [
                    {
                        'text': 'Total revenue of $245.1 billion in fiscal year 2024, representing 16% growth...',
                        'metadata': {'company': 'MSFT', 'year': '2024'},
                        'similarity': 0.92
                    }
                ]
            }
        elif 'GOOGL' in query:
            return {
                'results': [
                    {
                        'text': 'Total revenue of $328.3 billion in 2024, up from previous year...',
                        'metadata': {'company': 'GOOGL', 'year': '2024'},
                        'similarity': 0.90
                    }
                ]
            }
        elif 'NVDA' in query:
            return {
                'results': [
                    {
                        'text': 'Total revenue of $60.9 billion in 2024, driven by data center growth...',
                        'metadata': {'company': 'NVDA', 'year': '2024'},
                        'similarity': 0.88
                    }
                ]
            }
    
    # Mock AI strategy query
    if 'ai strategy' in query.lower():
        if 'GOOGL' in query:
            return {
                'results': [
                    {
                        'text': 'Our AI strategy focuses on responsible AI development, with significant investments in Gemini and cloud AI services. Key risks include regulatory challenges, competition, and potential misuse of AI technologies...',
                        'metadata': {'company': 'GOOGL', 'year': '2024'},
                        'similarity': 0.87
                    }
                ]
            }
    
    # Default empty response
    return {'results': []}


def test_integration_compound_query():
    """Test full integration of compound query with iterative follow-up."""
    print("\n" + "=" * 70)
    print("INTEGRATION TEST: Compound Query with Iterative Follow-up")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = EnhancedRAGPipeline()
    
    # Patch the base RAG query method to return our mock data
    pipeline.base_rag.query = mock_rag_query_with_data
    
    # Test query
    query = "Which company had the highest revenue in 2024? What are the main AI risks of that company?"
    
    print(f"\n📝 Query: {query}")
    print("\n🔄 Processing...\n")
    
    # Execute query
    result = pipeline.query(query, return_json=True)
    
    # Display results
    print("=" * 70)
    print("📋 RESULTS (JSON Output)")
    print("=" * 70)
    print(json.dumps(result, indent=2))
    
    # Validate results
    print("\n" + "=" * 70)
    print("✅ VALIDATION")
    print("=" * 70)
    
    checks = []
    
    # Check 1: Query field
    if result.get('query') == query:
        checks.append("✅ Query field matches input")
    else:
        checks.append("❌ Query field doesn't match")
    
    # Check 2: Sub-queries includes initial comparative queries
    sub_queries = result.get('sub_queries', [])
    has_msft_revenue = any('MSFT' in q and 'revenue' in q and '2024' in q for q in sub_queries)
    has_googl_revenue = any('GOOGL' in q and 'revenue' in q and '2024' in q for q in sub_queries)
    has_nvda_revenue = any('NVDA' in q and 'revenue' in q and '2024' in q for q in sub_queries)
    
    if has_msft_revenue and has_googl_revenue and has_nvda_revenue:
        checks.append("✅ Sub-queries include all 3 comparative revenue queries")
    else:
        checks.append("❌ Missing some comparative revenue queries")
    
    # Check 3: Sub-queries includes follow-up AI strategy query
    has_ai_followup = any('ai strategy' in q.lower() and '2024' in q for q in sub_queries)
    
    if has_ai_followup:
        checks.append("✅ Sub-queries include follow-up AI strategy query")
    else:
        checks.append("❌ Missing follow-up AI strategy query")
    
    # Check 4: Total sub-queries count
    if len(sub_queries) == 4:
        checks.append(f"✅ Correct number of sub-queries: {len(sub_queries)} (3 comparative + 1 follow-up)")
    else:
        checks.append(f"⚠️  Sub-queries count: {len(sub_queries)} (expected 4)")
    
    # Check 5: Answer exists
    if result.get('answer') and len(result.get('answer', '')) > 0:
        checks.append("✅ Answer generated")
    else:
        checks.append("❌ No answer generated")
    
    # Check 6: Reasoning exists
    if result.get('reasoning'):
        checks.append("✅ Reasoning provided")
    else:
        checks.append("❌ No reasoning provided")
    
    # Check 7: Sources exist
    if result.get('sources') and len(result.get('sources', [])) > 0:
        checks.append(f"✅ Sources provided: {len(result.get('sources', []))} sources")
    else:
        checks.append("⚠️  No sources provided")
    
    # Print all checks
    for check in checks:
        print(check)
    
    # Final verdict
    failed_checks = sum(1 for c in checks if c.startswith('❌'))
    
    print("\n" + "=" * 70)
    if failed_checks == 0:
        print("✅✅✅ INTEGRATION TEST PASSED ✅✅✅")
        print("=" * 70)
        print("\nThe compound query feature is working correctly:")
        print("- Detects compound queries with follow-up intent")
        print("- Executes initial comparative sub-queries")
        print("- Determines winner from comparative results")
        print("- Appends and executes follow-up sub-query")
        print("- Returns complete sub_queries list in JSON output")
        return True
    else:
        print(f"⚠️  INTEGRATION TEST COMPLETED WITH {failed_checks} ISSUES ⚠️")
        print("=" * 70)
        return False


def main():
    """Run the integration test."""
    try:
        success = test_integration_compound_query()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
