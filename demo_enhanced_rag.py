#!/usr/bin/env python3
"""
Enhanced RAG Financial System Demo

This script demonstrates the new LangGraph-powered capabilities including
query decomposition, multi-step retrieval, and synthesis.
"""

import sys
import os
import json
import time

def demo_enhanced_rag():
    """Demonstrate the enhanced RAG system capabilities."""
    print("🚀 Enhanced RAG Financial System Demo")
    print("=" * 60)
    print("LangGraph-Powered Query Decomposition & Synthesis")
    print("=" * 60)
    
    # Demo queries showcasing different capabilities
    demo_queries = [
        {
            "query": "Which company had the highest operating margin in 2023?",
            "description": "🔍 Comparative Analysis",
            "features": ["Query decomposition", "Multi-company comparison", "Financial metric extraction"]
        },
        {
            "query": "Compare Microsoft and Google revenue in 2022",
            "description": "📊 Multi-Company Revenue Comparison", 
            "features": ["Entity extraction", "Multi-step retrieval", "Synthesis with reasoning"]
        },
        {
            "query": "What was NVIDIA's growth rate from 2022 to 2023?",
            "description": "📈 Temporal Financial Analysis",
            "features": ["Company identification", "Temporal analysis", "Growth calculation"]
        }
    ]
    
    print("Demo Overview:")
    print("✅ Query Decomposition with LangGraph")
    print("✅ Multi-step Retrieval for Complex Questions") 
    print("✅ Synthesis with Reasoning and Confidence Scoring")
    print("✅ Structured JSON Output Format")
    print("✅ Source Attribution with Page References")
    print()
    
    # Check if vector store exists
    if not os.path.exists("./vector_db"):
        print("⚠️  Vector store not found. Setting up demo environment...")
        os.system("python demo_scraper.py")
        os.system("python main.py rag --process --input-dir demo_filings")
        print("✅ Demo environment ready!")
        print()
    
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n{demo['description']}")
        print("=" * 50)
        print(f"Query: {demo['query']}")
        print(f"Features Demonstrated: {', '.join(demo['features'])}")
        print()
        
        # Execute the query
        print("🔄 Processing with Enhanced RAG...")
        cmd = f'python main.py rag --query "{demo["query"]}" --top-k 3'
        
        print(f"Command: {cmd}")
        print("-" * 50)
        
        # Run the command and capture output
        exit_code = os.system(cmd)
        
        if exit_code != 0:
            print("❌ Query failed")
        else:
            print("✅ Query completed successfully")
        
        if i < len(demo_queries):
            print("\n" + "⏱️  Pausing before next demo..." + "\n")
            time.sleep(2)
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced RAG Demo Complete!")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("🧠 Intelligent query decomposition")
    print("🔍 Multi-step retrieval orchestration")
    print("📋 Structured JSON output with reasoning")
    print("📊 Comparative financial analysis")
    print("🏢 Multi-company data synthesis")
    print("📝 Source attribution and confidence scoring")
    
    print("\nTechnical Implementation:")
    print("• LangGraph workflows for query processing")
    print("• Automated entity extraction (companies, years, metrics)")
    print("• Comparative question detection and handling")
    print("• Financial data synthesis with reasoning")
    print("• Confidence scoring for answer reliability")
    
    print(f"\nFor more advanced usage, see: README.md")


def show_json_output_example():
    """Show example of the enhanced JSON output format."""
    print("\n📋 Enhanced JSON Output Format")
    print("=" * 40)
    
    example_output = {
        "query": "Which company had the highest operating margin in 2023?",
        "answer": "MSFT had the highest operating margin at 42.1% in 2023, followed by NVDA at 32.1%",
        "reasoning": "Retrieved operating margins for 3 companies from their 2023 10-K filings. Compared the following: MSFT: 42.1%, NVDA: 32.1%, GOOGL: 25.2%",
        "sub_queries": [
            "MSFT operating margin 2023",
            "GOOGL operating margin 2023", 
            "NVDA operating margin 2023"
        ],
        "sources": [
            {
                "company": "MSFT",
                "year": "2023",
                "excerpt": "Operating margin was 42.1%...",
                "page": 10
            },
            {
                "company": "NVDA", 
                "year": "2023",
                "excerpt": "Operating margin was 32.1%...",
                "page": 10
            },
            {
                "company": "GOOGL",
                "year": "2023", 
                "excerpt": "Operating margin was 25.2%...",
                "page": 10
            }
        ],
        "confidence": 0.9
    }
    
    print(json.dumps(example_output, indent=2))
    
    print("\nJSON Structure Benefits:")
    print("✅ Machine-readable format for downstream processing")
    print("✅ Standardized structure across all query types")
    print("✅ Detailed reasoning for transparency")
    print("✅ Source attribution for verification")
    print("✅ Confidence scoring for reliability assessment")


if __name__ == "__main__":
    demo_enhanced_rag()
    show_json_output_example()