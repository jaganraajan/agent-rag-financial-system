#!/usr/bin/env python3
"""
Synthesis Engine for Combining Multi-Step RAG Results

This module implements synthesis capabilities for combining results from multiple
sub-queries into a coherent answer with reasoning.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SourceInfo:
    """Information about a source document."""
    company: str
    year: str
    excerpt: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    similarity: float = 0.0


@dataclass
class SynthesisResult:
    """Result of synthesis process."""
    query: str
    answer: str
    reasoning: str
    sub_queries: List[str]
    sources: List[SourceInfo]


class SynthesisEngine:
    """Engine for synthesizing results from multiple RAG queries."""
    
    def __init__(self):
        """Initialize the synthesis engine."""
        self.logger = logging.getLogger(__name__)
        
        # Patterns for extracting financial metrics
        self.metric_patterns = {
            'operating_margin': [
                r'operating\s+margin\s+was\s+(\d+\.?\d*)%',
                r'operating\s+margin[:\s]*(\d+\.?\d*)%?',
                r'operating\s+income\s+margin[:\s]*(\d+\.?\d*)%?',
                r'(\d+\.?\d*)%?\s*operating\s+margin'
            ],
            'revenue': [
                r'total\s+revenue\s+of\s+\$(\d+\.?\d*)\s*(billion|million|trillion)?',
                r'revenue[:\s]*\$?(\d+\.?\d*)\s*(billion|million|trillion)?',
                r'total\s+revenue[:\s]*\$?(\d+\.?\d*)\s*(billion|million|trillion)?',
                r'net\s+revenue[:\s]*\$?(\d+\.?\d*)\s*(billion|million|trillion)?'
            ],
            'profit': [
                r'net\s+income[:\s]*\$?(\d+\.?\d*)\s*(billion|million|trillion)?',
                r'profit[:\s]*\$?(\d+\.?\d*)\s*(billion|million|trillion)?',
                r'net\s+profit[:\s]*\$?(\d+\.?\d*)\s*(billion|million|trillion)?'
            ]
        }
        
        # Company name mapping
        self.company_names = {'MSFT': 'Microsoft', 'GOOGL': 'Google', 'NVDA': 'NVIDIA'}
    
    def synthesize_comparative_results(self, query: str, sub_queries: List[str], 
                                     rag_results: List[Dict], query_type: str = "comparative") -> SynthesisResult:
        """Synthesize results for the two supported query types.
        
        Args:
            query: Original query
            sub_queries: List of sub-queries used
            rag_results: Results from RAG system for each sub-query
            query_type: Type of query being processed
            
        Returns:
            SynthesisResult with combined answer and reasoning
        """
        try:
            # Extract sources from RAG results
            sources = self._extract_sources(rag_results)
            
            # Only support two query types as per requirements
            if query_type == "simple":
                # Basic Metrics: Single company, single metric (e.g., "What was Microsoft's total revenue in 2023?")
                return self._synthesize_basic_metrics(query, sub_queries, sources)
            elif query_type == "comparative":
                # Cross-Company: Compare across companies (e.g., "Which company had the highest operating margin in 2023?")
                return self._synthesize_cross_company_comparison(query, sub_queries, sources)
            else:
                # Unsupported query type
                return SynthesisResult(
                    query=query,
                    answer="This query type is not supported. Please ask a basic metrics question about a specific company or a cross-company comparison question.",
                    reasoning=f"Query type '{query_type}' is not supported. Only basic metrics and cross-company comparisons are supported.",
                    sub_queries=sub_queries,
                    sources=[]
                )
                
        except Exception as e:
            self.logger.error(f"Error in synthesis: {e}")
            return SynthesisResult(
                query=query,
                answer="Unable to synthesize results due to an error.",
                reasoning=f"Synthesis failed: {str(e)}",
                sub_queries=sub_queries,
                sources=[]
            )
    
    def _extract_sources(self, rag_results: List[Dict]) -> List[SourceInfo]:
        """Extract and format source information from RAG results."""
        sources = []
        
        for result in rag_results:
            if 'results' not in result:
                continue
                
            for item in result.get('results', []):
                metadata = item.get('metadata', {})
                
                source = SourceInfo(
                    company=metadata.get('company', 'Unknown'),
                    year=metadata.get('year', 'Unknown'),
                    excerpt=item.get('text', '')[:200] + "..." if len(item.get('text', '')) > 200 else item.get('text', ''),
                    similarity=item.get('similarity', 0.0),
                    chunk_id=metadata.get('chunk_id')
                )
                sources.append(source)
        
        return sources
    
    def _extract_metric_value(self, text: str, metric_type: str) -> Optional[str]:
        """Extract a financial metric value from text.
        
        Args:
            text: The text to search
            metric_type: Type of metric to extract ('revenue', 'operating_margin', etc.)
            
        Returns:
            String value if found, None otherwise
        """
        patterns = self.metric_patterns.get(metric_type, [])
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                value = match.group(1)
                unit = match.group(2) if len(match.groups()) > 1 else None
                
                # Convert to standard format
                if metric_type == 'operating_margin':
                    return f"{value}%"
                elif metric_type == 'revenue':
                    if unit and 'billion' in unit:
                        return value
                    elif unit and 'million' in unit:
                        return str(float(value) / 1000)  # Convert to billions
                    else:
                        return value
                else:
                    return value
        
        return None

    def _synthesize_basic_metrics(self, query: str, sub_queries: List[str], 
                                sources: List[SourceInfo]) -> SynthesisResult:
        """Synthesize basic metrics results for a single company and year using actual RAG data."""
        # Extract company and year from query
        year_match = re.search(r'\b(20\d{2})\b', query)
        year = year_match.group(1) if year_match else '2023'
        
        # Extract company
        query_lower = query.lower()
        company = None
        for variant, ticker in {'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL', 'nvidia': 'NVDA'}.items():
            if variant in query_lower:
                company = ticker
                break
        
        if not company:
            return SynthesisResult(
                query=query,
                answer="Unable to identify the company in the query.",
                reasoning="Company not found in query.",
                sub_queries=sub_queries,
                sources=sources
            )
        
        # Extract financial data from RAG sources
        revenue_value = None
        relevant_sources = []
        
        for source in sources:
            if source.company == company and source.year == year:
                # Try to extract revenue value from the text excerpt
                revenue_match = self._extract_metric_value(source.excerpt, 'revenue')
                if revenue_match:
                    revenue_value = revenue_match
                    relevant_sources.append(source)
                    break
        
        if not revenue_value:
            return SynthesisResult(
                query=query,
                answer="Unable to find the requested financial data in the available sources.",
                reasoning=f"No revenue data found for {self.company_names.get(company, company)} in {year} from the RAG sources.",
                sub_queries=sub_queries,
                sources=sources
            )
        
        # Generate answer using actual data
        company_name = self.company_names.get(company, company)
        answer = f"{company_name}'s total revenue was ${revenue_value} billion in {year}."
        
        reasoning = f"Retrieved revenue data for {company_name} from their {year} financial filing."
        
        return SynthesisResult(
            query=query,
            answer=answer,
            reasoning=reasoning,
            sub_queries=sub_queries,
            sources=relevant_sources
        )

    def _synthesize_cross_company_comparison(self, query: str, sub_queries: List[str], 
                                           sources: List[SourceInfo]) -> SynthesisResult:
        """Synthesize cross-company comparison results using actual RAG data."""
        # Extract year from query
        year_match = re.search(r'\b(20\d{2})\b', query)
        year = year_match.group(1) if year_match else '2023'
        
        # Determine the metric being compared
        query_lower = query.lower()
        if 'operating margin' in query_lower:
            metric_type = 'operating_margin'
            metric_name = 'operating margin'
        elif 'revenue' in query_lower:
            metric_type = 'revenue'
            metric_name = 'revenue'
        else:
            metric_type = 'revenue'  # Default to revenue
            metric_name = 'revenue'
        
        # Extract data from RAG sources for each company
        company_data = {}
        relevant_sources = []
        
        for source in sources:
            if source.year == year:
                value = self._extract_metric_value(source.excerpt, metric_type)
                if value:
                    company_data[source.company] = {
                        'value': value,
                        'numeric_value': float(value.replace('%', '').replace('$', '')) if value else 0
                    }
                    relevant_sources.append(source)
        
        if not company_data:
            return SynthesisResult(
                query=query,
                answer=f"Unable to find {metric_name} data for the requested companies in {year}.",
                reasoning=f"No {metric_name} data found in the RAG sources for {year}.",
                sub_queries=sub_queries,
                sources=sources
            )
        
        # Find the highest value
        highest_company = max(company_data.keys(), key=lambda x: company_data[x]['numeric_value'])
        highest_value = company_data[highest_company]['value']
        
        # Generate answer
        company_name = self.company_names.get(highest_company, highest_company)
        
        if metric_type == 'operating_margin':
            answer = f"{company_name} had the highest {metric_name} at {highest_value} in {year}"
        else:
            answer = f"{company_name} had the highest {metric_name} at ${highest_value} billion in {year}"
        
        # Add context with other companies if available
        if len(company_data) > 1:
            sorted_companies = sorted(company_data.items(), key=lambda x: x[1]['numeric_value'], reverse=True)
            if len(sorted_companies) > 1:
                second_company, second_data = sorted_companies[1]
                second_name = self.company_names.get(second_company, second_company)
                second_value = second_data['value']
                if metric_type == 'operating_margin':
                    answer += f", followed by {second_name} at {second_value}"
                else:
                    answer += f", followed by {second_name} at ${second_value} billion"
        
        answer += "."
        
        reasoning = f"Compared {metric_name} data for {len(company_data)} companies from their {year} financial filings."
        
        return SynthesisResult(
            query=query,
            answer=answer,
            reasoning=reasoning,
            sub_queries=sub_queries,
            sources=relevant_sources
        )
    
    def format_json_output(self, synthesis_result: SynthesisResult) -> Dict[str, Any]:
        """Format synthesis result as JSON matching the required output format."""
        sources_json = []
        for source in synthesis_result.sources:
            sources_json.append({
                "company": source.company,
                "year": source.year,
                "excerpt": source.excerpt,
                "page": source.page or 10
            })

        return {
            "query": synthesis_result.query,
            "answer": synthesis_result.answer,
            "reasoning": synthesis_result.reasoning,
            "sub_queries": synthesis_result.sub_queries,
            "sources": sources_json
        }


def test_synthesis_engine():
    """Test function for the synthesis engine."""
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
    
    query = "Which company had the highest operating margin in 2023?"
    sub_queries = ["MSFT operating margin 2023", "GOOGL operating margin 2023", "NVDA operating margin 2023"]
    
    result = engine.synthesize_comparative_results(query, sub_queries, mock_results)
    json_output = engine.format_json_output(result)
    
    print("Synthesis Result:")
    import json
    print(json.dumps(json_output, indent=2))


if __name__ == "__main__":
    test_synthesis_engine()