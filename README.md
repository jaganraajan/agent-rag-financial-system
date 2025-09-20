# Agent RAG Financial System

A comprehensive financial analysis system that combines RAG (Retrieval-Augmented Generation) capabilities with a web scraper for SEC EDGAR 10-K filings. The system can answer both simple and comparative financial questions about Google, Microsoft, and NVIDIA using their recent 10-K filings, demonstrating query decomposition and multi-step reasoning for complex questions.

## Features

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

## Usage

### Quick Start - Download All Filings

```bash
python main.py
```

This will download all 10-K filings for all three companies (GOOGL, MSFT, NVDA) for years 2022-2024.

### Custom Usage

```bash
# Download specific companies
python main.py --companies GOOGL MSFT

# Download specific years
python main.py --years 2023 2024

# Custom output directory
python main.py --output-dir my_filings

# Custom user agent
python main.py --user-agent "My Financial Analysis Tool 1.0"
```

### Demo Mode

For testing or demonstration purposes (works without internet access):

```bash
python demo_scraper.py
```

This creates mock 10-K filings that demonstrate the expected output structure.

## File Structure

```
agent-rag-financial-system/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── sec_edgar_scraper.py        # Main scraper implementation
├── main.py                     # CLI interface
├── demo_scraper.py            # Demo/mock version
├── test_scraper.py            # Basic functionality tests
└── filings/                   # Downloaded 10-K filings (created on run)
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

Run basic functionality tests:
```bash
python test_scraper.py
```

### Dependencies

- `requests>=2.31.0` - HTTP requests to SEC API
- `pandas>=2.0.0` - Data manipulation (future RAG features)
- `beautifulsoup4>=4.12.0` - HTML parsing for SEC pages
- `lxml>=4.9.0` - XML/HTML parser backend
- `python-dateutil>=2.8.0` - Date parsing utilities

## API Reference

### SECEdgarScraper Class

Main scraper class with the following key methods:

- `get_company_filings(cik)` - Fetch all filings for a company
- `find_10k_filings(cik, years)` - Find 10-K filings for specific years
- `download_10k_filing(filing_info, output_dir)` - Download a single filing
- `scrape_company_10k_filings(symbol, years, output_dir)` - Scrape all filings for a company
- `scrape_all_companies(years, output_dir)` - Scrape all configured companies

## Future Enhancements

- RAG system integration for financial question answering
- Support for additional SEC filing types (10-Q, 8-K)
- Database storage for filing metadata
- Text extraction and preprocessing pipeline
- Financial data extraction and analysis tools

## License

This project is for educational and research purposes. Please ensure compliance with SEC terms of service when using the scraper.
