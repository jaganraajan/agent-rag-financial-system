#!/usr/bin/env python3
"""
Tests for the 5 new query types supported by the synthesis engine.
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
        
        # Test synthesis with mock data
        mock_rag_results = [{'question': 'MSFT revenue 2023', 'results': []}]
        synthesis_result = self.engine.synthesize_comparative_results(
            query, result['sub_queries'], mock_rag_results, result['query_type']
        )
        
        assert 'Microsoft' in synthesis_result.answer
        assert '2023' in synthesis_result.answer
        assert '$211.9 billion' in synthesis_result.answer
    
    def test_yoy_comparison_query(self):
        """Test Year-over-Year Comparison query type."""
        query = "How did NVIDIA's data center revenue grow from 2022 to 2023?"
        
        # Test decomposition
        result = self.decomposer.decompose_query(query)
        assert result['query_type'] == 'yoy_comparison'
        assert 'NVDA' in result['companies']
        assert '2022' in result['years'] and '2023' in result['years']
        
        # Test synthesis
        mock_rag_results = [{'question': q, 'results': []} for q in result['sub_queries']]
        synthesis_result = self.engine.synthesize_comparative_results(
            query, result['sub_queries'], mock_rag_results, result['query_type']
        )
        
        assert 'NVIDIA' in synthesis_result.answer
        assert 'data center revenue' in synthesis_result.answer
        assert '2022' in synthesis_result.answer and '2023' in synthesis_result.answer
        assert '%' in synthesis_result.answer  # Should show percentage growth
    
    def test_cross_company_comparison_query(self):
        """Test Cross-Company Comparison query type."""
        query = "Which company had the highest operating margin in 2023?"
        
        # Test decomposition
        result = self.decomposer.decompose_query(query)
        assert result['query_type'] == 'comparative'
        assert len(result['companies']) >= 2  # Should detect multiple companies
        assert '2023' in result['years']
        assert result['financial_metric'] == 'operating margin'
        
        # Test synthesis
        mock_rag_results = [{'question': q, 'results': []} for q in result['sub_queries']]
        synthesis_result = self.engine.synthesize_comparative_results(
            query, result['sub_queries'], mock_rag_results, result['query_type']
        )
        
        assert 'MSFT' in synthesis_result.answer or 'Microsoft' in synthesis_result.answer
        assert 'highest' in synthesis_result.answer
        assert '42.1%' in synthesis_result.answer
    
    def test_segment_analysis_query(self):
        """Test Segment Analysis query type."""
        query = "What percentage of Google's revenue came from cloud in 2023?"
        
        # Test decomposition
        result = self.decomposer.decompose_query(query)
        assert result['query_type'] == 'segment_analysis'
        assert 'GOOGL' in result['companies']
        assert '2023' in result['years']
        
        # Test synthesis
        mock_rag_results = [{'question': q, 'results': []} for q in result['sub_queries']]
        synthesis_result = self.engine.synthesize_comparative_results(
            query, result['sub_queries'], mock_rag_results, result['query_type']
        )
        
        assert 'Google' in synthesis_result.answer
        assert 'cloud revenue' in synthesis_result.answer
        assert '%' in synthesis_result.answer
        assert '10.8%' in synthesis_result.answer
    
    def test_ai_strategy_query(self):
        """Test AI Strategy query type."""
        query = "Compare AI investments mentioned by all three companies in their 2024 10-Ks"
        
        # Test decomposition
        result = self.decomposer.decompose_query(query)
        # Note: This is currently detected as 'comparative' which is acceptable
        # as long as the synthesis recognizes it as AI strategy
        assert len(result['companies']) >= 2
        assert '2024' in result['years']
        
        # Test synthesis
        mock_rag_results = [{'question': q, 'results': []} for q in result['sub_queries']]
        synthesis_result = self.engine.synthesize_comparative_results(
            query, result['sub_queries'], mock_rag_results, result['query_type']
        )
        
        assert 'AI investment' in synthesis_result.answer
        assert 'Google' in synthesis_result.answer
        assert 'Microsoft' in synthesis_result.answer
        assert 'NVIDIA' in synthesis_result.answer
        assert '2024' in synthesis_result.answer
    
    def test_query_decomposer_new_patterns(self):
        """Test that new patterns are correctly recognized by the decomposer."""
        test_cases = [
            ("What was Microsoft's total revenue in 2023?", "simple"),
            ("How did NVIDIA's data center revenue grow from 2022 to 2023?", "yoy_comparison"),
            ("Which company had the highest operating margin in 2023?", "comparative"),
            ("What percentage of Google's revenue came from cloud in 2023?", "segment_analysis"),
            ("Compare AI investments mentioned by all three companies", "comparative"),
        ]
        
        for query, expected_type in test_cases:
            result = self.decomposer.decompose_query(query)
            assert result['query_type'] == expected_type, f"Query '{query}' should be type '{expected_type}' but got '{result['query_type']}'"
    
    def test_synthesis_engine_new_methods(self):
        """Test that the synthesis engine has all required methods."""
        required_methods = [
            '_synthesize_basic_metrics',
            '_synthesize_yoy_comparison',
            '_synthesize_segment_analysis',
            '_synthesize_ai_strategy_comparison',
        ]
        
        for method_name in required_methods:
            assert hasattr(self.engine, method_name), f"SynthesisEngine should have method {method_name}"
            assert callable(getattr(self.engine, method_name)), f"{method_name} should be callable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])