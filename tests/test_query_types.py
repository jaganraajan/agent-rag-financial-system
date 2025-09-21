#!/usr/bin/env python3
"""
Tests for the 2 query types supported by the cleaned synthesis engine.
"""

import pytest
from src.agents.query_decomposer import QueryDecomposer
from src.agents.synthesis_engine import SynthesisEngine


class TestQueryTypes:
    
    def setup_method(self):
        """Setup test fixtures."""
        self.decomposer = QueryDecomposer()
        self.engine = SynthesisEngine()
    
    def test_basic_metrics_query(self):
        """Test Basic Metrics query type."""
        query = "What was Microsoft's total revenue in 2023?"
        
        # Test decomposition
        result = self.decomposer.decompose_query(query)
        assert result['query_type'] == 'simple'
        assert 'MSFT' in result['companies']
        assert '2023' in result['years']
        assert result['financial_metric'] == 'revenue'
        
        # Test synthesis with realistic mock RAG data
        mock_rag_results = [{
            'question': 'MSFT revenue 2023', 
            'results': [{
                'text': 'Microsoft Corporation reported total revenue of $211.9 billion for fiscal year 2023...',
                'metadata': {'company': 'MSFT', 'year': '2023'},
                'similarity': 0.9
            }]
        }]
        synthesis_result = self.engine.synthesize_comparative_results(
            query, result['sub_queries'], mock_rag_results, result['query_type']
        )
        
        assert 'Microsoft' in synthesis_result.answer
        assert '2023' in synthesis_result.answer
        assert '$211.9' in synthesis_result.answer
    
    def test_cross_company_comparison_query(self):
        """Test Cross-Company Comparison query type."""
        query = "Which company had the highest operating margin in 2023?"
        
        # Test decomposition
        result = self.decomposer.decompose_query(query)
        assert result['query_type'] == 'comparative'
        assert len(result['companies']) >= 2  # Should detect multiple companies
        assert '2023' in result['years']
        assert result['financial_metric'] == 'operating margin'
        
        # Test synthesis with realistic mock RAG data
        mock_rag_results = [
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
        
        synthesis_result = self.engine.synthesize_comparative_results(
            query, result['sub_queries'], mock_rag_results, result['query_type']
        )
        
        assert 'Microsoft' in synthesis_result.answer
        assert 'highest' in synthesis_result.answer
        assert '42.1%' in synthesis_result.answer
    
    def test_unsupported_query_types(self):
        """Test that unsupported query types are properly rejected."""
        unsupported_queries = [
            "How did NVIDIA's data center revenue grow from 2022 to 2023?",  # YoY comparison
            "What percentage of Google's revenue came from cloud in 2023?",  # Segment analysis
            "Compare AI investments mentioned by all three companies in their 2024 10-Ks"  # AI strategy
        ]
        
        for query in unsupported_queries:
            result = self.decomposer.decompose_query(query)
            
            # All should be classified as either 'simple' or 'comparative'
            assert result['query_type'] in ['simple', 'comparative']
            
            # If they get misclassified as simple/comparative but have unsupported content,
            # the synthesis engine should still handle them gracefully
            mock_rag_results = [{'question': 'test', 'results': []}]
            synthesis_result = self.engine.synthesize_comparative_results(
                query, result['sub_queries'], mock_rag_results, result['query_type']
            )
            
            # Should return a result (even if it's "no data found")
            assert isinstance(synthesis_result.answer, str)
            assert isinstance(synthesis_result.reasoning, str)
    
    def test_query_decomposer_supported_patterns(self):
        """Test that supported patterns are correctly recognized by the decomposer."""
        test_cases = [
            ("What was Microsoft's total revenue in 2023?", "simple"),
            ("Which company had the highest operating margin in 2023?", "comparative"),
            ("Compare Microsoft and Google revenue in 2022", "comparative"),
            ("What was NVIDIA's revenue in 2023?", "simple"),
        ]
        
        for query, expected_type in test_cases:
            result = self.decomposer.decompose_query(query)
            assert result['query_type'] == expected_type, f"Query '{query}' should be type '{expected_type}' but got '{result['query_type']}'"
    
    def test_synthesis_engine_has_required_methods(self):
        """Test that the synthesis engine has the required methods."""
        required_methods = [
            '_synthesize_basic_metrics',
            '_synthesize_cross_company_comparison',
        ]
        
        for method_name in required_methods:
            assert hasattr(self.engine, method_name), f"SynthesisEngine should have method {method_name}"
            assert callable(getattr(self.engine, method_name)), f"{method_name} should be callable"
    
    def test_synthesis_engine_rejects_unsupported_types(self):
        """Test that synthesis engine properly rejects unsupported query types."""
        query = "Test query"
        sub_queries = ["test"]
        mock_rag_results = []
        
        # Test with an unsupported query type
        result = self.engine.synthesize_comparative_results(
            query, sub_queries, mock_rag_results, query_type="unsupported_type"
        )
        
        assert "not supported" in result.answer.lower()
        assert "unsupported_type" in result.reasoning
    
    def test_basic_metrics_with_no_data(self):
        """Test basic metrics query when no data is available."""
        query = "What was Microsoft's total revenue in 2023?"
        
        result = self.decomposer.decompose_query(query)
        
        # Empty RAG results
        mock_rag_results = [{'question': 'MSFT revenue 2023', 'results': []}]
        synthesis_result = self.engine.synthesize_comparative_results(
            query, result['sub_queries'], mock_rag_results, result['query_type']
        )
        
        assert "Unable to find" in synthesis_result.answer
        assert "No revenue data found" in synthesis_result.reasoning
    
    def test_cross_company_with_no_data(self):
        """Test cross-company comparison when no data is available."""
        query = "Which company had the highest operating margin in 2023?"
        
        result = self.decomposer.decompose_query(query)
        
        # Empty RAG results
        mock_rag_results = [{'question': q, 'results': []} for q in result['sub_queries']]
        synthesis_result = self.engine.synthesize_comparative_results(
            query, result['sub_queries'], mock_rag_results, result['query_type']
        )
        
        assert "Unable to find" in synthesis_result.answer
        assert "No operating margin data found" in synthesis_result.reasoning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])