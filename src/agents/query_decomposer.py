#!/usr/bin/env python3
"""
Query Decomposer using LangGraph for Complex Financial Questions

This module implements query decomposition for breaking down complex financial questions
into simpler sub-queries that can be executed against the RAG system.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class QueryState:
    """State object for the query decomposition workflow."""
    original_query: str
    sub_queries: List[str]
    query_type: str
    companies: List[str]
    years: List[str]
    financial_metric: Optional[str] = None
    needs_comparison: bool = False
    error: Optional[str] = None


# Allowed query types for classification
ALLOWED_QUERY_TYPES = {"yoy_comparison", "segment_analysis", "ai_strategy", "comparative", "simple"}


class QueryDecomposer:
    """LangGraph-powered query decomposer for financial questions."""
    
    def __init__(self, use_openai: bool = False, llm_classifier: Optional[Callable[[str], str]] = None):
        """Initialize the query decomposer.
        
        Args:
            use_openai: Whether to use OpenAI API for classification (requires credentials)
            llm_classifier: Optional custom LLM classifier function that takes a query string
                          and returns a classification label
        """
        self.logger = logging.getLogger(__name__)
        self.use_openai = use_openai
        self.llm_classifier = llm_classifier
        
        # Company mapping for common variations
        self.company_mapping = {
            'microsoft': 'MSFT',
            'google': 'GOOGL',
            'alphabet': 'GOOGL',
            'nvidia': 'NVDA',
            'nvda': 'NVDA',
            'msft': 'MSFT',
            'googl': 'GOOGL'
        }
        
        # Financial metrics patterns (expanded for new query types)
        self.financial_metrics = {
            'operating margin': ['operating margin', 'operating income margin'],
            'revenue': ['revenue', 'total revenue', 'net revenue', 'sales'],
            'growth': ['growth', 'growth rate', 'yoy', 'year-over-year'],
            'segment': ['segment', 'business segment', 'division', 'category'],
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'ai investment', 'ai strategy']
        }
        
        # Initialize LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        """Build the LangGraph workflow for query decomposition."""
        from langgraph.graph import StateGraph, END
        
        workflow = StateGraph(dict)
        
        # Define nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("extract_entities", self._extract_entities)
        workflow.add_node("generate_subqueries", self._generate_subqueries)
        workflow.add_node("validate_queries", self._validate_queries)
        
        # Define edges
        workflow.add_edge("analyze_query", "extract_entities")
        workflow.add_edge("extract_entities", "generate_subqueries")
        workflow.add_edge("generate_subqueries", "validate_queries")
        workflow.add_edge("validate_queries", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        return workflow.compile()
    
    def _classify_with_llm(self, query: str) -> Optional[str]:
        """Classify query using LLM (custom or Azure OpenAI).
        
        Args:
            query: The original query string
            
        Returns:
            Classification label if successful and valid, None otherwise
        """
        # Try custom classifier first
        if self.llm_classifier is not None:
            try:
                self.logger.info("Attempting classification with custom LLM classifier")
                result = self.llm_classifier(query)
                if result:
                    label = result.strip().lower()
                    if label in ALLOWED_QUERY_TYPES:
                        self.logger.info(f"Custom LLM classifier returned: {label}")
                        return label
                    else:
                        self.logger.warning(f"Custom LLM classifier returned invalid label: {label}")
                        return None
            except Exception as e:
                self.logger.error(f"Custom LLM classifier failed: {e}")
                return None
        
        # Try Azure OpenAI if use_openai is enabled
        elif self.use_openai:
            try:
                import os
                from openai import AzureOpenAI
                
                self.logger.info("Attempting classification with Azure OpenAI")
                
                # Check for required environment variables
                azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
                api_key = os.getenv('AZURE_OPENAI_API_KEY')
                
                if not azure_endpoint or not api_key:
                    self.logger.warning("Azure OpenAI credentials not found in environment")
                    return None
                
                api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')
                model = os.getenv('AZURE_OPENAI_MODEL', 'gpt-4o-mini')
                
                # Initialize Azure OpenAI client
                client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version=api_version
                )
                
                # Call the model with strict classification prompt
                system_prompt = (
                    "You are a classifier. Classify the user query into one of: "
                    "yoy_comparison, segment_analysis, ai_strategy, comparative, simple. "
                    "Respond with only the label."
                )
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    temperature=0.0,
                    max_tokens=5
                )
                
                result = response.choices[0].message.content.strip().lower()
                
                # Validate the result
                if result in ALLOWED_QUERY_TYPES:
                    self.logger.info(f"Azure OpenAI classification: {result}")
                    return result
                else:
                    self.logger.warning(f"Azure OpenAI returned invalid label: {result}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Azure OpenAI classification failed: {e}")
                return None
        
        return None
    
    def _analyze_query(self, state: Dict) -> Dict:
        """Analyze the query type and complexity, including new query types."""
        original_query = state['original_query']
        query = original_query.lower()
        
        # Attempt LLM classification first if enabled
        llm_label = None
        if self.llm_classifier is not None or self.use_openai:
            llm_label = self._classify_with_llm(original_query)
        
        # If LLM classification succeeded, use it
        if llm_label is not None and llm_label in ALLOWED_QUERY_TYPES:
            state['query_type'] = llm_label
            state['needs_comparison'] = (llm_label == "comparative")
            self.logger.info(f"Using LLM classification - Query type: {state['query_type']}, Needs comparison: {state['needs_comparison']}")
            return state
        
        # Fallback to regex-based classification
        if llm_label is not None:
            self.logger.info("LLM classification attempted but invalid, falling back to regex")
        elif self.llm_classifier is not None or self.use_openai:
            self.logger.info("LLM classification failed, falling back to regex")
        
        # Patterns for new query types
        yoy_patterns = [r"year[- ]?over[- ]?year", r"yoy", r"growth from", r"growth rate", r"change from", r"change"]
        segment_patterns = [r"segment analysis", r"business segment", r"revenue by segment", r"division", r"category"]
        ai_patterns = [r"ai risks", r"ai strategy", r"ai investment", r"artificial intelligence", r"machine learning"]

        comparative_patterns = [
            r'\b(highest|lowest|best|worst|top|bottom|most|least|compare|versus|vs)\b',
            r'\bwhich\s+(company|organization)\b',
            r'\b(more|less|greater|smaller)\s+than\b'
        ]

        if any(re.search(pattern, query) for pattern in yoy_patterns):
            state['query_type'] = "yoy_comparison"
        elif any(re.search(pattern, query) for pattern in segment_patterns):
            state['query_type'] = "segment_analysis"
        elif any(re.search(pattern, query) for pattern in ai_patterns):
            state['query_type'] = "ai_strategy"
        elif any(re.search(pattern, query) for pattern in comparative_patterns):
            state['query_type'] = "comparative"
            state['needs_comparison'] = True
        else:
            state['query_type'] = "simple"
            state['needs_comparison'] = False

        self.logger.info(f"Using regex classification - Query type: {state['query_type']}, Needs comparison: {state['needs_comparison']}")
        return state
    
    def _extract_entities(self, state: Dict) -> Dict:
        """Extract companies, years, and financial metrics from the query."""
        query = state['original_query'].lower()
        
        # Extract companies
        companies = set()
        for variant, ticker in self.company_mapping.items():
            if variant in query:
                companies.add(ticker)
        
        # If no specific companies mentioned but it's comparative, include all
        if state['needs_comparison'] and not companies:
            companies = {'MSFT', 'GOOGL', 'NVDA'}
        
        state['companies'] = list(companies)
        
        # Extract years
        years = re.findall(r'\b(20\d{2})\b', state['original_query'])
        state['years'] = list(set(years)) if years else ['2023']  # Default to 2023
        
        # Extract financial metrics
        state['financial_metric'] = None
        for metric_type, patterns in self.financial_metrics.items():
            if any(pattern in query for pattern in patterns):
                state['financial_metric'] = metric_type
                break
        
        self.logger.info(f"Extracted - Companies: {state['companies']}, Years: {state['years']}, Metric: {state['financial_metric']}")
        return state
    
    def _generate_subqueries(self, state: Dict) -> Dict:
        """Generate sub-queries for all supported query types."""
        sub_queries = []

        if state['query_type'] == "comparative":
            companies = state['companies'] if state['companies'] else ['MSFT', 'GOOGL', 'NVDA']
            for company in companies:
                for year in state['years']:
                    if state['financial_metric']:
                        sub_query = f"{company} {state['financial_metric']} {year}"
                    else:
                        sub_query = f"{company} financial performance {year}"
                    sub_queries.append(sub_query)

        elif state['query_type'] == "yoy_comparison":
            # Year-over-year comparison: generate queries for each company and year
            companies = state['companies'] if state['companies'] else ['MSFT', 'GOOGL', 'NVDA']
            years = sorted(state['years']) if state['years'] else ['2022', '2023', '2024']
            if companies and state['financial_metric'] and len(years) >= 2:
                for company in companies:
                    for year in years:
                        sub_queries.append(f"{company} {state['financial_metric']} {year}")
            else:
                sub_queries.append(state['original_query'])

        elif state['query_type'] == "segment_analysis":
            # Segment analysis: generate queries for each segment if possible
            if state['companies'] and state['financial_metric']:
                company = state['companies'][0]
                year = state['years'][0] if state['years'] else '2023'
                sub_queries.append(f"{company} segment analysis {year}")
            else:
                sub_queries.append(state['original_query'])

        elif state['query_type'] == "ai_strategy":
            # AI strategy: generate queries for each company/year
            companies = state['companies'] if state['companies'] else ['MSFT', 'GOOGL', 'NVDA']
            year = state['years'][0] if state['years'] else '2023'
            for company in companies:
                sub_queries.append(f"{company} ai strategy {year}")

        else:
            # Simple/Basic metrics query - use as-is or enhance slightly
            if state['companies'] and state['financial_metric']:
                company = state['companies'][0]
                year = state['years'][0] if state['years'] else '2023'
                sub_queries.append(f"{company} {state['financial_metric']} {year}")
            else:
                sub_queries.append(state['original_query'])

        state['sub_queries'] = sub_queries
        self.logger.info(f"Generated {len(sub_queries)} sub-queries")
        return state
    
    def _validate_queries(self, state: Dict) -> Dict:
        """Validate and clean up the generated sub-queries."""
        validated_queries = []
        
        for query in state['sub_queries']:
            # Clean up the query
            cleaned = query.strip()
            if len(cleaned) > 5 and cleaned not in validated_queries:
                validated_queries.append(cleaned)
        
        state['sub_queries'] = validated_queries[:10]  # Limit to max 10 sub-queries
        
        if not state['sub_queries']:
            state['error'] = "No valid sub-queries could be generated"
        
        return state
    
    def decompose_query(self, query: str) -> Dict[str, Any]:
        """Decompose a complex query into sub-queries.
        
        Args:
            query: The original complex query
            
        Returns:
            Dictionary containing decomposition results
        """
        try:
            # Initialize state
            initial_state = {
                'original_query': query,
                'sub_queries': [],
                'query_type': "",
                'companies': [],
                'years': [],
                'financial_metric': None,
                'needs_comparison': False,
                'error': None
            }
            
            # Run the workflow
            result = self.workflow.invoke(initial_state)
            
            return {
                'original_query': result['original_query'],
                'query_type': result['query_type'],
                'sub_queries': result['sub_queries'],
                'companies': result['companies'],
                'years': result['years'],
                'financial_metric': result['financial_metric'],
                'needs_comparison': result['needs_comparison'],
                'error': result.get('error')
            }
            
        except Exception as e:
            self.logger.error(f"Error in query decomposition: {e}")
            return {
                'original_query': query,
                'query_type': 'error',
                'sub_queries': [query],  # Fallback to original
                'companies': [],
                'years': [],
                'financial_metric': None,
                'needs_comparison': False,
                'error': str(e)
            }


def test_query_decomposer():
    """Test function for the query decomposer."""
    print("=" * 70)
    print("TEST 1: Default Mode (Regex-only classification)")
    print("=" * 70)
    decomposer = QueryDecomposer()
    
    test_queries = [
        "Which company had the highest operating margin in 2023?",
        "Compare Microsoft and Google revenue in 2022",
        "What was NVIDIA's growth rate from 2022 to 2023?",
        "Tell me about Apple's financial performance"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = decomposer.decompose_query(query)
        print(f"Type: {result['query_type']}")
        print(f"Companies: {result['companies']}")
        print(f"Sub-queries: {result['sub_queries']}")
    
    print("\n" + "=" * 70)
    print("TEST 2: Custom LLM Classifier Mode (Local, no network)")
    print("=" * 70)
    
    # Custom classifier that always returns "comparative"
    def custom_classifier(query: str) -> str:
        return "comparative"
    
    decomposer_with_llm = QueryDecomposer(use_openai=False, llm_classifier=custom_classifier)
    
    # Test that custom classifier is used
    test_query = "Tell me about Apple's financial performance"
    print(f"\nQuery: {test_query}")
    print("(Expected: custom classifier should classify as 'comparative')")
    result = decomposer_with_llm.decompose_query(test_query)
    print(f"Type: {result['query_type']}")
    print(f"Companies: {result['companies']}")
    print(f"Sub-queries: {result['sub_queries']}")
    
    print("\n" + "=" * 70)
    print("TEST 3: Custom LLM Classifier with various labels")
    print("=" * 70)
    
    # Test different query type labels
    test_cases = [
        ("What is Microsoft's revenue?", "simple"),
        ("Compare companies", "comparative"),
        ("Year over year growth", "yoy_comparison"),
        ("Segment analysis for NVDA", "segment_analysis"),
        ("AI strategy investments", "ai_strategy"),
    ]
    
    for query_text, expected_label in test_cases:
        decomposer_test = QueryDecomposer(
            use_openai=False, 
            llm_classifier=lambda q: expected_label
        )
        print(f"\nQuery: {query_text}")
        print(f"Custom classifier returns: {expected_label}")
        result = decomposer_test.decompose_query(query_text)
        print(f"Actual type: {result['query_type']}")
        assert result['query_type'] == expected_label, f"Expected {expected_label}, got {result['query_type']}"
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_query_decomposer()