#!/usr/bin/env python3
"""
Enhanced RAG Pipeline with LangGraph Integration

This module extends the basic RAG pipeline with query decomposition,
multi-step retrieval, and synthesis capabilities using LangGraph.
"""

import sys
import os
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.rag.rag_pipeline import RAGPipeline as BaseRAGPipeline
from src.agents.query_decomposer import QueryDecomposer
from src.agents.synthesis_engine import SynthesisEngine


class EnhancedRAGPipeline:
    def _synthesize_yoy_comparison(self, query: str, sub_queries: List[str], rag_results: List[Dict]):
        """Synthesize year-over-year comparison using LLM from sub-query answers."""
        from src.agents.synthesis_engine import SynthesisResult, SourceInfo

        sub_answers = []
        all_sources = []
        for i, sub_query in enumerate(sub_queries):
            sub_result = rag_results[i] if i < len(rag_results) else {}
            simple_result = self._synthesize_simple_query(sub_query, [sub_query], [sub_result])
            sub_answers.append(f"{sub_query}: {simple_result.answer}")
            all_sources.extend(simple_result.sources)

        prompt = (
            f"You are a financial analyst assistant. "
            f"Given the following answers to sub-queries about year-over-year financial metrics, create a concise answer for the user's main question. "
            f"If you find specific values or growth rates, include them in a 1-2 line answer.\n\n"
            f"Main Question: {query}\n\n"
            f"Sub-query Answers:\n" + "\n".join(sub_answers)
        )

        try:
            from openai import AzureOpenAI
            import os
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
            model = os.getenv('AZURE_OPENAI_MODEL', 'gpt-4o-mini')
            client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=256
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"LLM call failed: {e}. Falling back to concatenated sub-query answers.\n" + "\n".join(sub_answers)

        reasoning = f"Synthesized YoY comparison using LLM from {len(sub_queries)} sub-query answers."

        return SynthesisResult(
            query=query,
            answer=answer,
            reasoning=reasoning,
            sub_queries=sub_queries,
            sources=all_sources[:5],
        )

    def _synthesize_segment_analysis(self, query: str, sub_queries: List[str], rag_results: List[Dict]):
        """Synthesize segment analysis using LLM from sub-query answers."""
        from src.agents.synthesis_engine import SynthesisResult, SourceInfo

        sub_answers = []
        all_sources = []
        for i, sub_query in enumerate(sub_queries):
            sub_result = rag_results[i] if i < len(rag_results) else {}
            simple_result = self._synthesize_simple_query(sub_query, [sub_query], [sub_result])
            sub_answers.append(f"{sub_query}: {simple_result.answer}")
            all_sources.extend(simple_result.sources)

        prompt = (
            f"You are a financial analyst assistant. "
            f"Given the following answers to sub-queries about financial segments, create a concise answer for the user's main question. "
            f"If you find specific values or percentages, include them in a 1-2 line answer.\n\n"
            f"Main Question: {query}\n\n"
            f"Sub-query Answers:\n" + "\n".join(sub_answers)
        )

        try:
            from openai import AzureOpenAI
            import os
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
            model = os.getenv('AZURE_OPENAI_MODEL', 'gpt-4o-mini')
            client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=256
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"LLM call failed: {e}. Falling back to concatenated sub-query answers.\n" + "\n".join(sub_answers)

        reasoning = f"Synthesized segment analysis using LLM from {len(sub_queries)} sub-query answers."

        return SynthesisResult(
            query=query,
            answer=answer,
            reasoning=reasoning,
            sub_queries=sub_queries,
            sources=all_sources[:5],
        )

    def _synthesize_ai_strategy(self, query: str, sub_queries: List[str], rag_results: List[Dict]):
        """Synthesize AI strategy comparison using LLM from sub-query answers."""
        from src.agents.synthesis_engine import SynthesisResult, SourceInfo

        sub_answers = []
        all_sources = []
        for i, sub_query in enumerate(sub_queries):
            sub_result = rag_results[i] if i < len(rag_results) else {}
            simple_result = self._synthesize_simple_query(sub_query, [sub_query], [sub_result])
            sub_answers.append(f"{sub_query}: {simple_result.answer}")
            all_sources.extend(simple_result.sources)

        prompt = (
            f"You are a financial analyst assistant. "
            f"Given the following answers to sub-queries about AI investments and strategy, create a concise comparative answer for the user's main question. "
            f"If you find specific values or investment amounts, include them in a 1-2 line answer.\n\n"
            f"Main Question: {query}\n\n"
            f"Sub-query Answers:\n" + "\n".join(sub_answers)
        )

        try:
            from openai import AzureOpenAI
            import os
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
            model = os.getenv('AZURE_OPENAI_MODEL', 'gpt-4o-mini')
            client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=256
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"LLM call failed: {e}. Falling back to concatenated sub-query answers.\n" + "\n".join(sub_answers)

        reasoning = f"Synthesized AI strategy comparison using LLM from {len(sub_queries)} sub-query answers."

        return SynthesisResult(
            query=query,
            answer=answer,
            reasoning=reasoning,
            sub_queries=sub_queries,
            sources=all_sources[:5],
        )
    
    def _synthesize_cross_company_comparison(self, query: str, sub_queries: List[str], rag_results: List[Dict]):
        """Synthesize cross-company comparison using LLM from sub-query answers."""
        from src.agents.synthesis_engine import SynthesisResult, SourceInfo

        # Get answers for each sub-query using _synthesize_simple_query
        sub_answers = []
        all_sources = []
        for i, sub_query in enumerate(sub_queries):
            sub_result = rag_results[i] if i < len(rag_results) else {}
            simple_result = self._synthesize_simple_query(sub_query, [sub_query], [sub_result])
            sub_answers.append(f"{sub_query}: {simple_result.answer}")
            all_sources.extend(simple_result.sources)

        # Prepare prompt for LLM
        prompt = (
            f"You are a financial analyst assistant. "
            f"Given the following answers to sub-queries about different companies, create a concise comparative answer for the user's main question. "
            f"If you find specific values, include them in a 1-2 line answer.\n\n"
            f"Main Question: {query}\n\n"
            f"Sub-query Answers:\n" + "\n".join(sub_answers)
        )

        # Call Azure OpenAI
        try:
            from openai import AzureOpenAI
            import os
            azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
            api_key = os.getenv('AZURE_OPENAI_API_KEY')
            api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
            model = os.getenv('AZURE_OPENAI_MODEL', 'gpt-4o-mini')
            client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=256
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"LLM call failed: {e}. Falling back to concatenated sub-query answers.\n" + "\n".join(sub_answers)

        reasoning = f"Synthesized comparative answer using LLM from {len(sub_queries)} sub-query answers."

        # Extract winner company from sources for compound query support
        winner_company = None
        year = self._extract_year_from_query(query)
        
        # Try to determine winner by finding company with highest metric
        if all_sources:
            # Group sources by company and extract values
            company_values = {}
            for source in all_sources:
                if source.year == year and source.company not in company_values:
                    # Try to extract numeric value from excerpt
                    import re
                    # Look for revenue patterns like "$123.4 billion"
                    match = re.search(r'\$(\d+\.?\d*)\s*billion', source.excerpt.lower())
                    if match:
                        company_values[source.company] = float(match.group(1))
            
            # Find company with highest value
            if company_values:
                winner_company = max(company_values, key=company_values.get)

        return SynthesisResult(
            query=query,
            answer=answer,
            reasoning=reasoning,
            sub_queries=sub_queries,
            sources=all_sources,  # Limit to top 5 sources
            metadata={'winner_company': winner_company, 'year': year} if winner_company else None
        )
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
    
    def _detect_compound_query_with_followup(self, query: str) -> bool:
        """Detect if query has both comparative and follow-up clauses with pronoun references.
        
        Returns True if the query contains:
        1. A comparative clause (e.g., "highest X in Y")
        2. A follow-up clause with pronoun reference (e.g., "that company")
        3. AI-related intent (ai risks/ai strategy)
        """
        query_lower = query.lower()
        
        # Check for comparative patterns
        comparative_patterns = [
            r'\b(highest|lowest|best|worst|top|bottom|most|least)\b',
            r'\bwhich\s+(company|organization)\b'
        ]
        has_comparative = any(re.search(pattern, query_lower) for pattern in comparative_patterns)
        
        # Check for follow-up with pronoun reference
        followup_patterns = [
            r'\bthat\s+(company|organization)\b',
            r'\bthe\s+(company|organization)\b',
            r'\bit\b'
        ]
        has_followup_pronoun = any(re.search(pattern, query_lower) for pattern in followup_patterns)
        
        # Check for AI-related intent
        ai_patterns = [r'ai\s+(risks?|strategy|strategies|investment)']
        has_ai_intent = any(re.search(pattern, query_lower) for pattern in ai_patterns)
        
        return has_comparative and has_followup_pronoun and has_ai_intent
    
    def _extract_year_from_query(self, query: str) -> str:
        """Extract year from query, defaulting to 2024."""
        year_match = re.search(r'\b(20\d{2})\b', query)
        return year_match.group(1) if year_match else '2024'
    
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
            
            # Check for compound query with follow-up
            is_compound = self._detect_compound_query_with_followup(question)
            
            # Step 1: Query Decomposition
            decomposition_result = self.query_decomposer.decompose_query(question)
            
            if decomposition_result.get('error'):
                self.logger.error(f"Query decomposition failed: {decomposition_result['error']}")
                # Fallback to basic RAG
                return self._fallback_query(question, top_k, return_json)
            
            # Step 2: Multi-step Retrieval (initial sub-queries)
            sub_queries = decomposition_result['sub_queries']
            self.logger.info(f"Executing {len(sub_queries)} sub-queries")
            
            rag_results = []
            for sub_query in sub_queries:
                sub_result = self.base_rag.query(sub_query, top_k)
                rag_results.append(sub_result)
                self.logger.debug(f"Sub-query '{sub_query}' returned {len(sub_result.get('results', []))} results")
            
            # Step 3: Synthesis (initial synthesis)
            query_type = decomposition_result['query_type']
            if query_type == 'comparative':
                synthesis_result = self._synthesize_cross_company_comparison(
                    question, sub_queries, rag_results
                )
            elif query_type == 'yoy_comparison':
                synthesis_result = self._synthesize_yoy_comparison(
                    question, sub_queries, rag_results
                )
            elif query_type == 'segment_analysis':
                synthesis_result = self._synthesize_segment_analysis(
                    question, sub_queries, rag_results
                )
            elif query_type == 'ai_strategy':
                synthesis_result = self._synthesize_ai_strategy(
                    question, sub_queries, rag_results
                )
            else:
                synthesis_result = self._synthesize_simple_query(question, sub_queries, rag_results)
            
            # Step 4: Iterative follow-up for compound queries
            if is_compound and synthesis_result.metadata and 'winner_company' in synthesis_result.metadata:
                winner = synthesis_result.metadata['winner_company']
                year = self._extract_year_from_query(question)
                
                self.logger.info(f"Detected compound query with winner: {winner}, appending follow-up sub-query")
                
                # Generate follow-up sub-query: "{WINNER} ai strategy {year}"
                followup_query = f"{winner} ai strategy {year}"
                self.logger.info(f"Follow-up sub-query: {followup_query}")
                
                # Execute follow-up retrieval
                followup_result = self.base_rag.query(followup_query, top_k)
                
                # Append to executed sub_queries and results
                synthesis_result.sub_queries.append(followup_query)
                rag_results.append(followup_result)
                
                # Re-synthesize with combined results using _synthesize_ai_strategy
                synthesis_result = self._synthesize_ai_strategy(
                    question, synthesis_result.sub_queries, rag_results
                )
            
            # Step 5: Format output
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
        
        # Create synthesis using Azure OpenAI LLM
        if sources:
            # Prepare context for LLM
            context_chunks = "\n\n".join([s.excerpt for s in sources[:5]])
            prompt = (
                f"You are a financial analyst assistant. "
                f"Given the following SEC filing excerpts, answer in 1-2 lines the user's question as precisely as possible. "
                f"If you find a specific value, include it in your answer.\n\n"
                f"Question: {query}\n\n"
                f"SEC Filing Excerpts:\n{context_chunks}"
            )

            # Call Azure OpenAI
            try:
                from openai import AzureOpenAI
                import os
                azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
                api_key = os.getenv('AZURE_OPENAI_API_KEY')
                api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
                model = os.getenv('AZURE_OPENAI_MODEL', 'gpt-4o-mini')
                client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version=api_version
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a financial analyst assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=256
                )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                answer = f"LLM call failed: {e}. Falling back to best retrieved chunk."
                best_source = max(sources, key=lambda x: x.similarity)
                answer += f" Based on the retrieved information from {best_source.company} ({best_source.year}), relevant details have been found. Analysis includes {total_results} relevant document sections."
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