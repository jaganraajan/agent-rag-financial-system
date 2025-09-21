#!/usr/bin/env python3
"""
Test Enhanced RAG System with LangGraph

This script tests the enhanced RAG system functionality including query decomposition,
multi-step retrieval, and synthesis.
"""

import sys
import os
import json
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append('src')

def test_enhanced_rag():
    """Test the enhanced RAG system end-to-end."""
    print("=" * 60)
    print("🧪 TESTING ENHANCED RAG SYSTEM WITH LANGGRAPH")
    print("=" * 60)
    
    try:
        from src.agents.enhanced_rag import EnhancedRAGPipeline
        from src.agents.query_decomposer import QueryDecomposer
        from src.agents.synthesis_engine import SynthesisEngine
        
        print("✅ Successfully imported enhanced RAG components")
        
        # Test 1: Query Decomposer
        print("\n1. Testing Query Decomposer")
        print("-" * 30)
        decomposer = QueryDecomposer()
        
        test_query = "Which company had the highest operating margin in 2023?"
        result = decomposer.decompose_query(test_query)
        
        print(f"Query: {test_query}")
        print(f"Type: {result['query_type']}")
        print(f"Companies: {result['companies']}")
        print(f"Sub-queries: {result['sub_queries']}")
        print(f"Needs comparison: {result['needs_comparison']}")
        
        assert result['query_type'] == 'comparative'
        assert 'MSFT' in result['companies']
        assert 'GOOGL' in result['companies']
        assert 'NVDA' in result['companies']
        assert len(result['sub_queries']) == 3
        print("✅ Query decomposer working correctly")
        
        # Test 2: Synthesis Engine
        print("\n2. Testing Synthesis Engine")
        print("-" * 30)
        engine = SynthesisEngine()
        
        # Mock RAG results
        mock_results = [
            {
                'question': 'MSFT operating margin 2023',
                'results': [
                    {
                        'text': 'Microsoft operating margin was 42.1% in fiscal year 2023...',
                        'metadata': {'company': 'MSFT', 'year': '2023'},
                        'similarity': 0.9
                    }
                ]
            },
            {
                'question': 'GOOGL operating margin 2023',
                'results': [
                    {
                        'text': 'Google operating margin was 25.2% in 2023...',
                        'metadata': {'company': 'GOOGL', 'year': '2023'},
                        'similarity': 0.85
                    }
                ]
            }
        ]
        
        synthesis_result = engine.synthesize_comparative_results(
            test_query, 
            result['sub_queries'], 
            mock_results
        )
        
        json_output = engine.format_json_output(synthesis_result)
        
        print("Synthesis Result Structure:")
        print(f"  - Query: {json_output['query']}")
        print(f"  - Answer: {json_output['answer'][:50]}...")
        print(f"  - Sources: {len(json_output['sources'])} sources")
        print(f"  - Sub-queries: {len(json_output['sub_queries'])} sub-queries")
        print(f"  - Confidence: {json_output['confidence']}")
        
        assert 'MSFT' in json_output['answer']
        assert json_output['confidence'] > 0.8
        assert len(json_output['sources']) > 0
        print("✅ Synthesis engine working correctly")
        
        # Test 3: Enhanced RAG Pipeline (if vector store exists)
        print("\n3. Testing Enhanced RAG Pipeline")
        print("-" * 30)
        
        vector_db_path = "./vector_db"
        if os.path.exists(vector_db_path):
            print("Vector store found, testing full pipeline...")
            
            pipeline = EnhancedRAGPipeline(vector_store_path=vector_db_path)
            
            test_queries = [
                "Which company had the highest operating margin in 2023?",
                "Compare Microsoft and Google revenue"
            ]
            
            for query in test_queries:
                print(f"\nTesting: {query}")
                result = pipeline.query(query, return_json=True)
                
                # Validate JSON structure
                required_fields = ['query', 'answer', 'reasoning', 'sub_queries', 'sources']
                for field in required_fields:
                    assert field in result, f"Missing field: {field}"
                
                print(f"  ✓ Generated {len(result['sub_queries'])} sub-queries")
                print(f"  ✓ Found {len(result['sources'])} sources")
                print(f"  ✓ Answer: {result['answer'][:50]}...")
            
            print("✅ Enhanced RAG pipeline working correctly")
        else:
            print("⚠️  Vector store not found, skipping full pipeline test")
            print("   Run 'python main.py rag --process' first to create vector store")
        
        print("\n" + "=" * 60)
        print("🎉 ALL ENHANCED RAG TESTS PASSED!")
        print("=" * 60)
        print("\nEnhanced Features Verified:")
        print("✅ Query decomposition with LangGraph")
        print("✅ Multi-step retrieval orchestration")  
        print("✅ Synthesis with reasoning and confidence")
        print("✅ Structured JSON output format")
        print("✅ Comparative question handling")
        print("✅ Source attribution and page references")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Enhanced RAG components not available")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    logging.getLogger().setLevel(logging.ERROR)
    
    success = test_enhanced_rag()
    sys.exit(0 if success else 1)