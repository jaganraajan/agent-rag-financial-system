#!/usr/bin/env python3
"""
Enhanced RAG Pipeline with LangGraph Integration

This module extends the basic RAG pipeline with query decomposition,
multi-step retrieval, and synthesis capabilities using LangGraph.
"""

import sys
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.rag.rag_pipeline import RAGPipeline as BaseRAGPipeline
from src.agents.query_decomposer import QueryDecomposer
from src.agents.synthesis_engine import SynthesisEngine


class EnhancedRAGPipeline:
    """Enhanced RAG Pipeline with LangGraph query decomposition and synthesis."""
    
    def __init__(self, vector_store_path: str = "./vector_db", use_openai: bool = False):
        """Initialize the enhanced RAG pipeline.
        
        Args:
            vector_store_path: Path to the vector database
            use_openai: Whether to use OpenAI API for enhanced capabilities
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize base RAG pipeline
        self.base_rag = BaseRAGPipeline(vector_store_path)
        
        # Initialize LangGraph components
        self.query_decomposer = QueryDecomposer(use_openai=use_openai)
        self.synthesis_engine = SynthesisEngine()
        
        self.logger.info("Enhanced RAG Pipeline with LangGraph initialized")
    
    def process_directory(self, filings_dir: str) -> Dict:
        """Process directory using the base RAG pipeline."""
        return self.base_rag.process_directory(filings_dir)
    
    def get_stats(self) -> Dict:
        """Get statistics from the base RAG pipeline."""
        return self.base_rag.get_stats()
    
    def query(self, question: str, top_k: int = 5, return_json: bool = True) -> Dict:
        """Enhanced query processing with decomposition and synthesis.
        
        Args:
            question: The user's question
            top_k: Number of top results to retrieve per sub-query
            return_json: Whether to return structured JSON output
            
        Returns:
            Enhanced results with decomposition, multi-step retrieval, and synthesis
        """
        try:
            self.logger.info(f"Processing enhanced query: {question}")
            
            # Step 1: Query Decomposition
            decomposition_result = self.query_decomposer.decompose_query(question)
            
            if decomposition_result.get('error'):
                self.logger.error(f"Query decomposition failed: {decomposition_result['error']}")
                # Fallback to basic RAG
                return self._fallback_query(question, top_k, return_json)
            
            # Step 2: Multi-step Retrieval
            sub_queries = decomposition_result['sub_queries']
            self.logger.info(f"Executing {len(sub_queries)} sub-queries")
            
            rag_results = []
            for sub_query in sub_queries:
                sub_result = self.base_rag.query(sub_query, top_k)
                rag_results.append(sub_result)
                self.logger.debug(f"Sub-query '{sub_query}' returned {len(sub_result.get('results', []))} results")
            
            # Step 3: Synthesis
            if decomposition_result['needs_comparison'] or decomposition_result['query_type'] == 'comparative':
                synthesis_result = self.synthesis_engine.synthesize_comparative_results(
                    question, sub_queries, rag_results, decomposition_result['query_type']
                )
            else:
                # For non-comparative queries, use simpler synthesis
                synthesis_result = self._synthesize_simple_query(question, sub_queries, rag_results)
            
            # Step 4: Format output
            if return_json:
                return self.synthesis_engine.format_json_output(synthesis_result)
            else:
                return self._format_traditional_output(synthesis_result, rag_results)
                
        except Exception as e:
            self.logger.error(f"Error in enhanced query processing: {e}")
            return self._fallback_query(question, top_k, return_json)
    
    def _fallback_query(self, question: str, top_k: int, return_json: bool) -> Dict:
        """Fallback to basic RAG query if enhanced processing fails."""
        self.logger.info("Falling back to basic RAG query")
        basic_result = self.base_rag.query(question, top_k)
        
        if return_json:
            # Convert basic result to JSON format
            sources = []
            for result in basic_result.get('results', []):
                metadata = result.get('metadata', {})
                sources.append({
                    "company": metadata.get('company', 'Unknown'),
                    "year": metadata.get('year', 'Unknown'),
                    "excerpt": result.get('text', '')[:200] + "..." if len(result.get('text', '')) > 200 else result.get('text', ''),
                    "page": 10  # Default page
                })
            
            return {
                "query": question,
                "answer": f"Found {len(basic_result.get('results', []))} relevant results for your query.",
                "reasoning": "Basic RAG retrieval without decomposition due to processing limitations.",
                "sub_queries": [question],
                "sources": sources
            }
        else:
            return basic_result
    
    def _synthesize_simple_query(self, query: str, sub_queries: List[str], rag_results: List[Dict]):
        """Synthesize results for non-comparative queries."""
        from src.agents.synthesis_engine import SynthesisResult, SourceInfo
        
        # Extract all sources
        sources = []
        total_results = 0
        
        for result in rag_results:
            for item in result.get('results', []):
                metadata = item.get('metadata', {})
                source = SourceInfo(
                    company=metadata.get('company', 'Unknown'),
                    year=metadata.get('year', 'Unknown'),
                    excerpt=item.get('text', '')[:200] + "..." if len(item.get('text', '')) > 200 else item.get('text', ''),
                    similarity=item.get('similarity', 0.0)
                )
                sources.append(source)
                total_results += 1
        
        # Create simple synthesis
        if sources:
            # Focus on highest similarity source
            best_source = max(sources, key=lambda x: x.similarity)
            answer = f"Based on the retrieved information from {best_source.company} ({best_source.year}), "
            answer += f"relevant details have been found. Analysis includes {total_results} relevant document sections."
        else:
            answer = "No specific information was found for the query."
        
        reasoning = f"Retrieved information from {len(sources)} document sections using {len(sub_queries)} targeted searches."
        
        return SynthesisResult(
            query=query,
            answer=answer,
            reasoning=reasoning,
            sub_queries=sub_queries,
            sources=sources[:5],  # Limit to top 5
        )
    
    def _format_traditional_output(self, synthesis_result, rag_results: List[Dict]) -> Dict:
        """Format output in traditional RAG format for backward compatibility."""
        # Combine all RAG results
        combined_results = []
        for result in rag_results:
            combined_results.extend(result.get('results', []))
        
        # Sort by similarity
        combined_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        return {
            'question': synthesis_result.query,
            'results': combined_results[:10],  # Top 10 results
            'top_k': 10,
            'synthesis': {
                'answer': synthesis_result.answer,
                'reasoning': synthesis_result.reasoning,
                'sub_queries': synthesis_result.sub_queries
            }
        }
    
    def query_basic(self, question: str, top_k: int = 5) -> Dict:
        """Basic query using the original RAG pipeline (for compatibility)."""
        return self.base_rag.query(question, top_k)


def test_enhanced_rag():
    """Test function for the enhanced RAG pipeline."""
    print("Testing Enhanced RAG Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = EnhancedRAGPipeline()
    
    # Test queries
    test_queries = [
        "Which company had the highest operating margin in 2023?",
        "Compare Microsoft and Google revenue",
        "What are NVIDIA's main business activities?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        # Test JSON output
        result = pipeline.query(query, return_json=True)
        print("JSON Output:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    test_enhanced_rag()