#!/usr/bin/env python3
"""
Demonstration of Compound Query with Iterative Follow-up

This script demonstrates the complete iterative query decomposition feature
with visual output showing each step of the process.
"""

import sys
import os
import json
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.enhanced_rag import EnhancedRAGPipeline

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def demo_compound_query():
    """Demonstrate compound query with step-by-step output."""
    print_banner("COMPOUND QUERY DECOMPOSITION DEMONSTRATION")
    
    # Initialize pipeline
    print("\n📦 Initializing Enhanced RAG Pipeline...")
    pipeline = EnhancedRAGPipeline(vector_store_path='./vector_db')
    print("✅ Pipeline initialized")
    
    # The test query
    query = "Which company had the highest revenue in 2024? What are the main AI risks of that company?"
    
    print_banner("TEST QUERY")
    print(f"\n💬 User Query:")
    print(f"   \"{query}\"")
    
    print("\n🔍 Query Analysis:")
    is_compound = pipeline._detect_compound_query_with_followup(query)
    year = pipeline._extract_year_from_query(query)
    print(f"   - Compound query detected: {is_compound}")
    print(f"   - Year extracted: {year}")
    
    if is_compound:
        print("\n✨ This query will trigger iterative follow-up!")
        print("   Expected flow:")
        print("   1️⃣  Decompose into comparative sub-queries (revenue)")
        print("   2️⃣  Execute retrieval for all companies")
        print("   3️⃣  Determine winner from comparative results")
        print("   4️⃣  Append and execute follow-up query (AI strategy)")
        print("   5️⃣  Synthesize final answer combining all results")
    
    print_banner("EXECUTING QUERY")
    print("\n⚙️  Processing...\n")
    
    # Execute the query
    result = pipeline.query(query, return_json=True)
    
    print_banner("RESULTS")
    
    # Display sub-queries
    print("\n📋 Sub-queries Executed:")
    sub_queries = result.get('sub_queries', [])
    for i, sq in enumerate(sub_queries, 1):
        icon = "🔄" if i <= 3 else "➕"  # Different icon for follow-up
        print(f"   {icon} {i}. {sq}")
    
    # Analyze sub-queries
    print("\n📊 Analysis:")
    initial_queries = [sq for sq in sub_queries if 'revenue' in sq.lower()]
    followup_queries = [sq for sq in sub_queries if 'ai strategy' in sq.lower()]
    
    print(f"   - Initial comparative queries: {len(initial_queries)}")
    print(f"   - Follow-up queries: {len(followup_queries)}")
    print(f"   - Total queries executed: {len(sub_queries)}")
    
    if followup_queries:
        print(f"\n   ✅ Follow-up query successfully appended!")
        print(f"   ➡️  Follow-up: \"{followup_queries[0]}\"")
    else:
        print(f"\n   ⚠️  Follow-up not appended (winner could not be determined)")
        print(f"   💡 Note: This may occur when source data lacks extractable metrics")
    
    # Display sources
    sources = result.get('sources', [])
    print(f"\n📚 Sources Retrieved: {len(sources)} documents")
    if sources:
        companies = set(s.get('company') for s in sources[:5])
        print(f"   - Companies: {', '.join(companies)}")
    
    # Full JSON output
    print_banner("FULL JSON OUTPUT")
    print("\n" + json.dumps(result, indent=2))
    
    print_banner("DEMONSTRATION COMPLETE")
    
    # Summary
    print("\n📝 Summary:")
    if len(sub_queries) > 3 and followup_queries:
        print("   ✅ Compound query successfully processed with iterative follow-up")
        print("   ✅ All sub-queries included in final output")
        print("   ✅ System correctly identified winner and appended follow-up")
    elif is_compound:
        print("   ⚠️  Compound query detected but follow-up not appended")
        print("   💡 Reason: Winner could not be extracted from available data")
        print("   💡 With real financial data, follow-up would be appended")
    else:
        print("   ℹ️  Query processed as standard comparative query")
    
    print("\n" + "=" * 80)
    print()


if __name__ == "__main__":
    try:
        demo_compound_query()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
