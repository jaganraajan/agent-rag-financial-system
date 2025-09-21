# ChromaDB Inspector UI

A web-based interface for inspecting and analyzing ChromaDB contents and chunking quality for the RAG Financial System.

## Features

- **Dashboard**: Overview of collection statistics, token distribution, and content breakdown
- **Browse Chunks**: Detailed inspection of individual chunks with filtering capabilities
- **Semantic Search**: Query interface for finding relevant chunks with similarity scoring

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Process some documents** (optional, if no data exists):
   ```bash
   python main.py rag --process --input-dir demo_filings
   ```

3. **Start the UI server**:
   ```bash
   python chromadb_ui.py
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:8080
   ```

## Usage

### Dashboard
- View total number of chunks and token statistics
- See distribution of content by section type (business, financial, risk, etc.)
- Analyze content types (text vs financial tables)

### Browse Chunks
- Filter chunks by section type
- View detailed metadata for each chunk including:
  - Section title and type
  - Token count
  - Content type (text/financial_table)
- Inspect actual chunk content

### Semantic Search
- Enter natural language queries
- View similarity scores for matching chunks
- Adjust number of results returned (5, 10, or 20)

## Enhanced 10-K Chunking Features

The system now includes enhanced chunking specifically optimized for SEC 10-K filings:

- **Section-aware chunking**: Preserves 10-K document structure (PART I, PART II, Items)
- **Financial table extraction**: Special handling for financial statements and tables
- **Rich metadata**: Each chunk includes section title, type, and hierarchical level
- **Content classification**: Automatic detection of business, financial, risk, and legal content

## API Endpoints

The UI provides REST API endpoints for programmatic access:

- `GET /api/stats` - Collection statistics
- `GET /api/chunks` - List all chunks (with optional filtering)
- `GET /api/search` - Semantic search

Example:
```bash
# Get collection stats
curl http://localhost:8080/api/stats

# Search for chunks
curl "http://localhost:8080/api/search?query=revenue&top_k=3"
```

## Configuration

The UI connects to the ChromaDB instance at `./vector_db` by default. This can be configured by modifying the `ChromaDBInspector` initialization in `chromadb_ui.py`.