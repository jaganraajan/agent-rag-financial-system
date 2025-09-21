# Agent RAG Financial System

A comprehensive financial analysis system that combines **Enhanced RAG (Retrieval-Augmented Generation)** capabilities with **LangGraph query decomposition** and a web scraper for SEC EDGAR 10-K filings. The system processes SEC filings, creates searchable vector embeddings, and provides intelligent answers to complex financial questions about Google, Microsoft, and NVIDIA using their recent 10-K filings.

## 🚀 New Features with LangGraph Integration

### 🧠 Query Decomposition
- **Automatic Question Breaking**: Complex questions are automatically decomposed into simpler sub-queries
- **Multi-step Retrieval**: Executes multiple targeted searches for comprehensive answers
- **Comparative Analysis**: Handles "which company" and comparison questions intelligently

### 📊 Enhanced Query Types
- **Comparative Queries**: "Which company had the highest operating margin in 2023?"
- **Multi-Company Analysis**: "Compare Microsoft and Google revenue"
- **Temporal Analysis**: "What was NVIDIA's growth from 2022 to 2023?"

### 📋 Structured JSON Output
```json
{
  "query": "Which company had the highest operating margin in 2023?",
  "answer": "MSFT had the highest operating margin at 42.1% in 2023, followed by NVDA at 32.1%",
  "reasoning": "Retrieved operating margins for 3 companies from their 2023 10-K filings...",
  "sub_queries": ["MSFT operating margin 2023", "GOOGL operating margin 2023", "NVDA operating margin 2023"],
  "sources": [
    {
      "company": "MSFT",
      "year": "2023", 
      "excerpt": "Operating margin was 42.1%...",
      "page": 10
    }
  ],
  "confidence": 0.9
}
```

## Features

### 🔍 SEC EDGAR Web Scraper
- Downloads 10-K filings for GOOGL, MSFT, and NVDA (2022-2024)
- Handles rate limiting and SEC compliance
- Robust error handling and retry logic
- Demo mode for offline testing

### 🤖 Enhanced RAG Pipeline
- **Query Decomposition**: LangGraph-powered analysis of complex questions
- **Multi-step Retrieval**: Executes multiple targeted sub-queries
- **Synthesis Engine**: Combines results with reasoning and confidence scoring
- **Text Extraction**: Parses HTML filings and extracts clean text
- **Semantic Chunking**: Splits documents into 50-1000 token chunks with semantic boundaries
- **Embeddings**: Uses Azure OpenAI embeddings (with offline mock fallback)
- **Vector Storage**: ChromaDB for efficient similarity search
- **Dual Interface**: Enhanced JSON output and traditional query modes
- **Metadata**: Extracts company, year, and filing information

### 💡 Key Capabilities
- **Intelligent Question Analysis**: Automatically detects comparative, temporal, and simple queries
- **Multi-Company Comparisons**: Handles cross-company financial analysis
- **Structured Reasoning**: Provides explanations for how answers were derived
- **Source Attribution**: Tracks and cites specific document sections
- **Confidence Scoring**: Assigns confidence levels to generated answers
- **Fallback Support**: Gracefully handles errors with basic RAG fallback

- **SEC EDGAR Web Scraper**: Automated downloading of 10-K filings from the SEC EDGAR database
- **Multi-Company Support**: Covers Google (GOOGL), Microsoft (MSFT), and NVIDIA (NVDA)
- **Multi-Year Coverage**: Downloads filings for 2022, 2023, and 2024
- **Rate-Limited API Access**: Respectful scraping with proper delays and user-agent headers
- **Comprehensive Error Handling**: Robust error handling for network issues and missing data
- **Structured Output**: Organized file naming and directory structure

## SEC EDGAR Web Scraper

### Requirements Met

The web scraper satisfies the following data scope requirements:

| Aspect | Requirement | Implementation |
|--------|-------------|----------------|
| **Companies** | Google (GOOGL), Microsoft (MSFT), NVIDIA (NVDA) | ✅ Implemented with CIK codes |
| **Documents** | Annual 10-K filings only | ✅ Filters for 10-K form type |
| **Years** | 2022, 2023, 2024 | ✅ Configurable year range |
| **Total Files** | 9 documents (3 companies × 3 years) | ✅ Downloads all combinations |
| **Source** | SEC EDGAR database | ✅ Uses official SEC API |

### Company CIK Codes

- **GOOGL (Alphabet Inc.)**: 1652044
- **MSFT (Microsoft Corporation)**: 789019
- **NVDA (NVIDIA Corporation)**: 1045810

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jaganraajan/agent-rag-financial-system.git
cd agent-rag-financial-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up Azure OpenAI credentials for production embeddings:
```bash
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_API_KEY="your-api-key"
```

## Usage

### 🚀 Quick Start - RAG Pipeline

#### 1. Generate Demo Files
```bash
python demo_scraper.py
```

#### 2. Process Documents
```bash
python main.py rag --process --input-dir demo_filings
```

#### 3. Query the System
```bash
# Enhanced comparative queries (new!)
python main.py rag --query "Which company had the highest operating margin in 2023?"

# Multi-company comparisons
python main.py rag --query "Compare Microsoft and Google revenue"

# Traditional single queries
python main.py rag --query "What are NVIDIA's main risk factors?"

# Interactive mode
python main.py rag
```

#### 🆕 Enhanced Query Examples
```bash
# Operating margin comparison
python main.py rag --query "Which company had the highest operating margin in 2023?"

# Revenue analysis
python main.py rag --query "Compare MSFT and GOOGL revenue in 2022"

# Growth analysis  
python main.py rag --query "What was NVIDIA's growth rate from 2022 to 2023?"

# Force basic mode (without LangGraph features)
python main.py rag --basic --query "Your question"
```

### 📊 SEC Scraper Mode

#### Download All Filings
```bash
python main.py scrape
```

#### Custom Downloads
```bash
# Download specific companies
python main.py scrape --companies GOOGL MSFT --years 2023 2024

# Custom output directory
python main.py scrape --output-dir my_filings

# Custom user agent
python main.py scrape --user-agent "My Analysis Tool 1.0"
```

### 🔧 RAG Pipeline Examples

```bash
# Process real downloaded filings
python main.py rag --process --input-dir filings

# Query with custom parameters
python main.py rag --query "NVIDIA revenue growth" --top-k 3

# Use custom vector store location
python main.py rag --vector-store ./my_vector_db --process
```

## File Structure

```
agent-rag-financial-system/
├── README.md                    # Project documentation  
├── requirements.txt             # Python dependencies (includes LangGraph)
├── .gitignore                  # Git ignore rules
├── main.py                     # Main CLI interface (enhanced + basic modes)
├── sec_edgar_scraper.py        # SEC EDGAR scraper implementation  
├── rag_pipeline.py             # Original RAG pipeline components
├── demo_scraper.py            # Demo/mock version for testing
├── src/                       # Organized source code structure
│   ├── agents/                # LangGraph agents and enhanced RAG
│   │   ├── query_decomposer.py     # Query decomposition with LangGraph
│   │   ├── synthesis_engine.py     # Result synthesis and reasoning
│   │   └── enhanced_rag.py         # Enhanced RAG pipeline orchestrator
│   ├── rag/                   # Core RAG components
│   │   └── rag_pipeline.py         # Original RAG pipeline (moved)
│   ├── scrapers/              # Web scraping modules
│   │   ├── sec_edgar_scraper.py    # SEC scraper (moved)
│   │   └── demo_scraper.py         # Demo scraper (moved)
│   └── utils/                 # Utility functions
├── tests/                     # Test modules
│   ├── test_scraper.py            # Basic scraper functionality tests
│   └── test_rag_system.py         # Comprehensive RAG system tests
├── filings/                   # Downloaded 10-K filings (created on scrape)
├── demo_filings/              # Demo filings (created by demo_scraper.py)
└── vector_db/                 # ChromaDB vector store (created on RAG processing)
```

### Example File Contents
```
demo_filings/
├── GOOGL_10K_2022_[accession].htm
├── GOOGL_10K_2023_[accession].htm
├── GOOGL_10K_2024_[accession].htm
├── MSFT_10K_2022_[accession].htm
├── MSFT_10K_2023_[accession].htm
├── MSFT_10K_2024_[accession].htm
├── NVDA_10K_2022_[accession].htm
├── NVDA_10K_2023_[accession].htm
└── NVDA_10K_2024_[accession].htm
```

## Technical Details

### RAG Pipeline Architecture

#### 1. Text Extraction (`TextExtractor`)
- Parses HTML SEC filings using BeautifulSoup
- Removes scripts, styles, and formatting
- Cleans and normalizes text content

#### 2. Chunking (`TextChunker`) 
- **Token Range**: 50-1000 tokens per chunk
- **Semantic Boundaries**: Splits on paragraphs and sentences
- **Token Counting**: tiktoken (with word-based fallback)
- **Metadata**: Preserves company, year, filename information

#### 3. Embeddings (`EmbeddingService`)
- **Primary**: Azure OpenAI text-embedding-ada-002
- **Fallback**: Deterministic mock embeddings (1536-dim)
- **Offline Support**: Full functionality without internet

#### 4. Vector Storage (`VectorStore`)
- **Database**: ChromaDB with persistent storage
- **Features**: Similarity search, metadata filtering
- **Performance**: Efficient retrieval with distance scoring

#### 5. Query Pipeline (`RAGPipeline`)
- **Processing**: End-to-end document ingestion
- **Search**: Semantic similarity with top-k results
- **Metadata**: Rich context for each result

### Dependencies

#### Core RAG Dependencies
- `chromadb>=0.4.15` - Vector database for embeddings
- `openai>=1.0.0` - Azure OpenAI API integration
- `tiktoken>=0.5.0` - Token counting and text processing

#### Enhanced LangGraph Dependencies (New!)
- `langgraph>=0.6.0` - LangGraph workflow orchestration
- `langchain>=0.3.0` - LangChain core components
- `langchain-openai>=0.3.0` - LangChain OpenAI integration

#### Web Scraping Dependencies
- `requests>=2.31.0` - HTTP requests to SEC API
- `pandas>=2.0.0` - Data manipulation
- `beautifulsoup4>=4.12.0` - HTML parsing for SEC pages
- `lxml>=4.9.0` - XML/HTML parser backend
- `python-dateutil>=2.8.0` - Date parsing utilities

### SEC EDGAR API

The scraper uses the official SEC EDGAR API:
- **Submissions API**: `https://data.sec.gov/submissions/CIK{CIK}.json`
- **Archives**: `https://www.sec.gov/Archives/edgar/data/{CIK}/{accession}/`

### Rate Limiting

- Implements 0.1-second delays between requests (well below SEC's 10 requests/second limit)
- Additional delays between companies (2 seconds) and downloads (1 second)
- Automatic retry logic with exponential backoff

### Error Handling

- Network connectivity issues
- Missing filings for specific years
- Malformed SEC responses
- File system errors

## Development

### Testing

Run comprehensive system tests:
```bash
python test_rag_system.py
```

Run basic scraper tests:
```bash
python test_scraper.py
```

### Demo Mode

For testing without internet access:
```bash
python demo_scraper.py  # Generate demo files
python main.py rag --process --input-dir demo_filings  # Process them
python main.py rag  # Interactive query mode
```

## API Reference

### CLI Commands

```bash
# Main interface
python main.py {scrape,rag} [options]

# Scraper mode
python main.py scrape --companies GOOGL MSFT NVDA --years 2022 2023 2024 --output-dir filings

# RAG mode  
python main.py rag --process --input-dir demo_filings --vector-store ./vector_db
python main.py rag --query "Your question" --top-k 5
```

### Python API

#### RAG Pipeline
```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline(vector_store_path="./vector_db")

# Process documents
results = rag.process_directory("demo_filings")

# Query system
response = rag.query("What are the risk factors?", top_k=5)
```

#### SEC Scraper
```python
from sec_edgar_scraper import SECEdgarScraper

# Initialize scraper
scraper = SECEdgarScraper(user_agent="Your App 1.0")

# Download filings
files = scraper.scrape_company_10k_filings("GOOGL", [2023, 2024], "output_dir")
```

## Future Enhancements

- **Enhanced Question Answering**: Integration with large language models for natural language responses
- **Advanced Analytics**: Financial ratio calculations and trend analysis
- **More Document Types**: Support for 10-Q, 8-K, and other SEC filings
- **Real-time Updates**: Automatic fetching of new filings
- **Web Interface**: Browser-based query interface
- **Multi-modal Search**: Charts, tables, and text-based retrieval
- **Comparative Analysis**: Cross-company financial comparisons

## Performance Notes

- **Processing Speed**: ~1-2 seconds per document for demo files
- **Storage**: Vector database grows ~50MB per 100 documents
- **Memory Usage**: ~200MB baseline + ~50MB per 1000 chunks
- **Query Speed**: Sub-second response for similarity search

## License

This project is for educational and research purposes. Please ensure compliance with SEC terms of service when using the scraper.
