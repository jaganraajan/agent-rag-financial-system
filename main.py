#!/usr/bin/env python3
"""
Main script for SEC EDGAR Financial RAG System.

This script supports two modes:
1. Scraper mode: Downloads 10-K filings from SEC EDGAR
2. RAG mode: Processes filings and provides query interface
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from compound_query_planner import CompoundQueryPlanner
from src.scrapers.sec_edgar_scraper import SECEdgarScraper

# Try to import enhanced RAG, fall back to basic if not available
try:
    sys.path.append('src')
    from src.agents.enhanced_rag import EnhancedRAGPipeline
    ENHANCED_RAG_AVAILABLE = True
except ImportError:
    from rag_pipeline import RAGPipeline
    ENHANCED_RAG_AVAILABLE = False
    print("Enhanced RAG not available, using basic RAG pipeline")


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def run_scraper_mode(args):
    """Run the SEC filing scraper."""
    print("SEC EDGAR 10-K Filings Scraper")
    print("=" * 40)
    print(f"Companies: {', '.join(args.companies)}")
    print(f"Years: {', '.join(map(str, args.years))}")
    print(f"Output directory: {args.output_dir}")
    print(f"Expected total files: {len(args.companies) * len(args.years)}")
    print("=" * 40)
    print()
    
    # Create scraper instance
    scraper = SECEdgarScraper(user_agent=args.user_agent)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Track results
    all_results = {}
    total_downloaded = 0
    
    try:
        # Download filings for each company
        for company in args.companies:
            print(f"\nProcessing {company}...")
            files = scraper.scrape_company_10k_filings(company, args.years, args.output_dir)
            all_results[company] = files
            total_downloaded += len(files)
            
        # Print summary
        print("\n" + "=" * 50)
        print("DOWNLOAD SUMMARY")
        print("=" * 50)
        
        for company, files in all_results.items():
            print(f"\n{company} ({scraper.companies[company]['name']}):")
            print(f"  Files downloaded: {len(files)}")
            
            for file_path in files:
                filename = os.path.basename(file_path)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                print(f"    - {filename} ({file_size:,} bytes)")
        
        print(f"\nTotal files downloaded: {total_downloaded}")
        print(f"Expected files: {len(args.companies) * len(args.years)}")
        
        if total_downloaded == len(args.companies) * len(args.years):
            print("✅ All expected files downloaded successfully!")
        else:
            print("⚠️  Some files may be missing. Check the logs above for details.")
            
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during download: {e}")
        sys.exit(1)


def run_rag_mode(args):
    """Run the RAG pipeline."""
    print("SEC EDGAR Financial RAG System")
    print("=" * 40)
    print(f"Input directory: {args.input_dir}")
    print(f"Vector store path: {args.vector_store}")
    print("=" * 40)
    print()
    
    # Initialize RAG pipeline
    try:
        if ENHANCED_RAG_AVAILABLE:
            rag = EnhancedRAGPipeline(vector_store_path=args.vector_store)
            print("🚀 Using Enhanced RAG with LangGraph")
        else:
            rag = RAGPipeline(vector_store_path=args.vector_store)
            print("📊 Using Basic RAG Pipeline")
        
        if args.process:
            # Process documents
            print("Processing documents...")
            results = rag.process_directory(args.input_dir)
            
            print("\n" + "=" * 50)
            print("PROCESSING SUMMARY")
            print("=" * 50)
            print(f"Files processed: {results['processed_files']}/{results['total_files']}")
            print(f"Total chunks created: {results['total_chunks']}")
            print("✅ Documents processed successfully!")
            print()
        
        # Get system stats
        stats = rag.get_stats()
        print(f"Vector store contains: {stats['total_chunks']} chunks")
        print()
        
        if args.query:
            # Single query mode
            print(f"Query: {args.query}")
            print("=" * 60)

            def is_compound_query(query: str) -> bool:
                """Use Azure OpenAI to classify if a query is compound."""
                try:
                    from openai import AzureOpenAI
                    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
                    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
                    azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
                    azure_model = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o-mini")
                    client = AzureOpenAI(
                        azure_endpoint=azure_endpoint,
                        api_key=azure_api_key,
                        api_version=azure_api_version,
                    )
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a financial assistant. Classify the user's query as 'compound' if it contains multiple distinct sub-questions, dependencies, or requires multi-step reasoning. Otherwise, classify as 'simple'. Return STRICT JSON: {\"type\": \"compound\" or \"simple\"}."
                        },
                        {
                            "role": "user",
                            "content": query
                        }
                    ]
                    resp = client.chat.completions.create(
                        model=azure_model,
                        temperature=0.0,
                        messages=messages,
                    )
                    raw = resp.choices[0].message.content or "{}"
                    import json as _json
                    result = _json.loads(raw)
                    return result.get("type", "simple") == "compound"
                except Exception as e:
                    print(f"⚠️  Compound query classification failed: {e}. Defaulting to simple.")
                    return False

            if ENHANCED_RAG_AVAILABLE:
                # Use Azure OpenAI to classify query type
                compound = is_compound_query(args.query)
                if compound:
                    planner = CompoundQueryPlanner(max_steps=4, top_k=5)
                    result = planner.run(args.query)
                    print("\n=== Compound Query Result ===")
                    print(json.dumps(result, indent=2))
                else:
                    result = rag.query(args.query, top_k=args.top_k, return_json=True)
                    print("\n📋 Enhanced Query Result (JSON):")
                    print("=" * 40)
                    print(json.dumps(result, indent=2))
            else:
                # Use basic query  
                logging.basicConfig(level=logging.INFO)
                result = rag.query(args.query, top_k=args.top_k)
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                    return
                print(f"Found {len(result['results'])} relevant chunks:")
                print()
                for i, chunk in enumerate(result['results'], 1):
                    print(f"Result {i} (Similarity: {chunk['similarity']:.3f}):")
                    print(f"Company: {chunk['metadata'].get('company', 'Unknown')}")
                    print(f"Year: {chunk['metadata'].get('year', 'Unknown')}")
                    print(f"Text: {chunk['text'][:200]}...")
                    print("-" * 30)
                    print()
        
        elif not args.process:
            # Interactive query mode
            print("🤖 Interactive Query Mode")
            print("Type your questions about the financial documents.")
            print("Type 'quit' or 'exit' to stop.")
            print()
            
            while True:
                try:
                    query = input("💬 Your question: ").strip()
                    
                    if query.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if not query:
                        continue
                    
                    print(f"\n🔍 Searching for: {query}")
                    print("-" * 50)
                    
                    result = rag.query(query, top_k=args.top_k)
                    
                    if 'error' in result:
                        print(f"❌ Error: {result['error']}")
                        continue
                    
                    if not result['results']:
                        print("No relevant information found.")
                        print()
                        continue
                    
                    print(f"Found {len(result['results'])} relevant chunks:")
                    print()
                    
                    for i, chunk in enumerate(result['results'], 1):
                        print(f"📄 Result {i} (Similarity: {chunk['similarity']:.3f}):")
                        print(f"   Company: {chunk['metadata'].get('company', 'Unknown')}")
                        print(f"   Year: {chunk['metadata'].get('year', 'Unknown')}")
                        print(f"   Text: {chunk['text'][:300]}...")
                        print()
                    
                    print("-" * 50)
                    print()
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    print()
            
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        sys.exit(1)
def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='SEC EDGAR Financial RAG System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download filings (scraper mode)
  python main.py scrape --companies GOOGL MSFT --years 2023 2024
  
  # Process downloaded filings for RAG
  python main.py rag --process --input-dir filings
  
  # Query the RAG system
  python main.py rag --query "What are the main revenue sources?"
  
  # Interactive query mode
  python main.py rag --input-dir demo_filings
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Scraper mode
    scraper_parser = subparsers.add_parser('scrape', help='Download SEC filings')
    scraper_parser.add_argument('--companies', nargs='+', default=['GOOGL', 'MSFT', 'NVDA'],
                               choices=['GOOGL', 'MSFT', 'NVDA'],
                               help='Company symbols to download (default: all)')
    scraper_parser.add_argument('--years', nargs='+', type=int, default=[2022, 2023, 2024],
                               help='Years to download filings for (default: 2022 2023 2024)')
    scraper_parser.add_argument('--output-dir', default='filings',
                               help='Output directory for downloaded filings (default: filings)')
    scraper_parser.add_argument('--user-agent', 
                               default='Financial Analysis Tool 1.0 (jaganraajan@gmail.com)',
                               help='User agent for SEC requests')
    
    # RAG mode
    rag_parser = subparsers.add_parser('rag', help='RAG pipeline operations')
    rag_parser.add_argument('--input-dir', default='demo_filings',
                           help='Directory containing HTML filings (default: demo_filings)')
    rag_parser.add_argument('--vector-store', default='./vector_db',
                           help='Path for vector store persistence (default: ./vector_db)')
    rag_parser.add_argument('--process', action='store_true',
                           help='Process documents and build vector store')
    rag_parser.add_argument('--query', type=str,
                           help='Single query to execute')
    rag_parser.add_argument('--top-k', type=int, default=5,
                           help='Number of top results to return (default: 5)')
    rag_parser.add_argument('--enhanced', action='store_true', default=True,
                           help='Use enhanced RAG with LangGraph (default: True if available)')
    rag_parser.add_argument('--basic', action='store_true',
                           help='Force use of basic RAG pipeline')
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # If no mode specified, show help
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Route to appropriate mode
    if args.mode == 'scrape':
        run_scraper_mode(args)
    elif args.mode == 'rag':
        run_rag_mode(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()