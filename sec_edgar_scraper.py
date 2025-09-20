"""
SEC EDGAR Web Scraper for 10-K Filings

This module provides functionality to download 10-K filings from the SEC EDGAR database
for specified companies and years.
"""

import os
import re
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
import pandas as pd
from bs4 import BeautifulSoup


class SECEdgarScraper:
    """
    A web scraper for downloading 10-K filings from SEC EDGAR database.
    """
    
    def __init__(self, user_agent: str = "Financial Analysis Tool 1.0"):
        """
        Initialize the SEC EDGAR scraper.
        
        Args:
            user_agent: User agent string for SEC API requests
        """
        self.base_url = "https://data.sec.gov"
        self.archives_url = "https://www.sec.gov/Archives"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'application/json, text/html, */*',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Company information
        self.companies = {
            'GOOGL': {'cik': '1652044', 'name': 'Alphabet Inc.'},
            'MSFT': {'cik': '789019', 'name': 'Microsoft Corporation'},
            'NVDA': {'cik': '1045810', 'name': 'NVIDIA Corporation'}
        }
        
        # Rate limiting - SEC recommends no more than 10 requests per second
        self.rate_limit_delay = 0.1
    
    def _make_request(self, url: str, retries: int = 3) -> Optional[requests.Response]:
        """
        Make a rate-limited request to the SEC API.
        
        Args:
            url: URL to request
            retries: Number of retry attempts
            
        Returns:
            Response object or None if failed
        """
        for attempt in range(retries):
            try:
                time.sleep(self.rate_limit_delay)
                response = self.session.get(url)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limited
                    self.logger.warning(f"Rate limited, waiting longer... (attempt {attempt + 1})")
                    time.sleep(1.0 * (attempt + 1))
                    continue
                else:
                    self.logger.error(f"HTTP {response.status_code} for {url}")
                    
            except requests.RequestException as e:
                self.logger.error(f"Request failed for {url}: {e}")
                
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))
        
        return None
    
    def get_company_filings(self, cik: str) -> Optional[Dict]:
        """
        Get all filings for a company by CIK.
        
        Args:
            cik: Central Index Key for the company
            
        Returns:
            Dictionary containing filing information or None if failed
        """
        # Pad CIK to 10 digits
        cik_padded = cik.zfill(10)
        url = f"{self.base_url}/submissions/CIK{cik_padded}.json"
        
        self.logger.info(f"Fetching filings for CIK {cik}")
        response = self._make_request(url)
        
        if response:
            try:
                return response.json()
            except ValueError as e:
                self.logger.error(f"Failed to parse JSON for CIK {cik}: {e}")
        
        return None
    
    def find_10k_filings(self, cik: str, target_years: List[int]) -> List[Dict]:
        """
        Find 10-K filings for specific years.
        
        Args:
            cik: Central Index Key for the company
            target_years: List of years to search for
            
        Returns:
            List of filing dictionaries with accession numbers and filing dates
        """
        filings_data = self.get_company_filings(cik)
        if not filings_data:
            return []
        
        found_filings = []
        recent_filings = filings_data.get('filings', {}).get('recent', {})
        
        if not recent_filings:
            self.logger.warning(f"No recent filings found for CIK {cik}")
            return []
        
        forms = recent_filings.get('form', [])
        filing_dates = recent_filings.get('filingDate', [])
        accession_numbers = recent_filings.get('accessionNumber', [])
        
        for i, form in enumerate(forms):
            if form == '10-K' and i < len(filing_dates):
                filing_date = filing_dates[i]
                filing_year = int(filing_date.split('-')[0])
                
                if filing_year in target_years:
                    found_filings.append({
                        'form': form,
                        'filing_date': filing_date,
                        'filing_year': filing_year,
                        'accession_number': accession_numbers[i] if i < len(accession_numbers) else None,
                        'cik': cik
                    })
        
        # Sort by filing year
        found_filings.sort(key=lambda x: x['filing_year'])
        return found_filings
    
    def download_10k_filing(self, filing_info: Dict, output_dir: str) -> Optional[str]:
        """
        Download a 10-K filing document.
        
        Args:
            filing_info: Dictionary containing filing information
            output_dir: Directory to save the downloaded file
            
        Returns:
            Path to downloaded file or None if failed
        """
        accession_number = filing_info.get('accession_number')
        if not accession_number:
            self.logger.error("No accession number provided")
            return None
        
        # Remove hyphens from accession number for URL
        accession_clean = accession_number.replace('-', '')
        cik_padded = filing_info['cik'].zfill(10)
        
        # Construct the URL for the filing document
        # SEC EDGAR URL format: /Archives/edgar/data/CIK/ACCESSIONNUMBER/ACCESSIONNUMBER-index.htm
        base_filing_url = f"{self.archives_url}/edgar/data/{filing_info['cik']}/{accession_clean}"
        
        # First, try to get the index page to find the actual 10-K document
        index_url = f"{base_filing_url}/{accession_number}-index.htm"
        
        self.logger.info(f"Downloading 10-K filing for {filing_info['filing_date']}")
        response = self._make_request(index_url)
        
        if not response:
            self.logger.error(f"Failed to fetch index page: {index_url}")
            return None
        
        # Parse the index page to find the 10-K document
        soup = BeautifulSoup(response.content, 'html.parser')
        document_link = None
        
        # Look for links to the 10-K document
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('.htm') and '10-k' in href.lower():
                document_link = href
                break
        
        if not document_link:
            # Fallback: try common naming pattern
            document_link = f"{accession_number}.txt"
        
        # Download the actual document
        document_url = f"{base_filing_url}/{document_link}"
        doc_response = self._make_request(document_url)
        
        if not doc_response:
            self.logger.error(f"Failed to download document: {document_url}")
            return None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        company_symbol = next((symbol for symbol, info in self.companies.items() 
                              if info['cik'] == filing_info['cik']), filing_info['cik'])
        filename = f"{company_symbol}_10K_{filing_info['filing_year']}_{accession_number}.htm"
        filepath = os.path.join(output_dir, filename)
        
        # Save the document
        try:
            with open(filepath, 'wb') as f:
                f.write(doc_response.content)
            
            self.logger.info(f"Successfully downloaded: {filepath}")
            return filepath
            
        except IOError as e:
            self.logger.error(f"Failed to save file {filepath}: {e}")
            return None
    
    def scrape_company_10k_filings(self, symbol: str, years: List[int], output_dir: str) -> List[str]:
        """
        Scrape all 10-K filings for a company for specified years.
        
        Args:
            symbol: Company stock symbol (GOOGL, MSFT, NVDA)
            years: List of years to download filings for
            output_dir: Directory to save downloaded files
            
        Returns:
            List of paths to downloaded files
        """
        if symbol not in self.companies:
            self.logger.error(f"Unknown company symbol: {symbol}")
            return []
        
        cik = self.companies[symbol]['cik']
        company_name = self.companies[symbol]['name']
        
        self.logger.info(f"Starting 10-K scraping for {company_name} ({symbol}) - CIK: {cik}")
        
        # Find 10-K filings for the target years
        filings = self.find_10k_filings(cik, years)
        
        if not filings:
            self.logger.warning(f"No 10-K filings found for {symbol} in years {years}")
            return []
        
        self.logger.info(f"Found {len(filings)} 10-K filings for {symbol}")
        
        downloaded_files = []
        for filing in filings:
            filepath = self.download_10k_filing(filing, output_dir)
            if filepath:
                downloaded_files.append(filepath)
            
            # Add delay between downloads to be respectful
            time.sleep(1.0)
        
        return downloaded_files
    
    def scrape_all_companies(self, years: List[int] = [2022, 2023, 2024], 
                           output_dir: str = "filings") -> Dict[str, List[str]]:
        """
        Scrape 10-K filings for all configured companies.
        
        Args:
            years: List of years to download filings for
            output_dir: Directory to save downloaded files
            
        Returns:
            Dictionary mapping company symbols to lists of downloaded file paths
        """
        self.logger.info(f"Starting comprehensive 10-K scraping for years: {years}")
        
        results = {}
        total_files = 0
        
        for symbol in self.companies.keys():
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Processing {symbol}")
            self.logger.info(f"{'='*50}")
            
            downloaded_files = self.scrape_company_10k_filings(symbol, years, output_dir)
            results[symbol] = downloaded_files
            total_files += len(downloaded_files)
            
            # Add delay between companies
            time.sleep(2.0)
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"SCRAPING COMPLETE")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Total files downloaded: {total_files}")
        
        for symbol, files in results.items():
            self.logger.info(f"{symbol}: {len(files)} files")
        
        return results


if __name__ == "__main__":
    # Example usage
    scraper = SECEdgarScraper()
    results = scraper.scrape_all_companies()
    
    print("\nDownload Summary:")
    for company, files in results.items():
        print(f"{company}: {len(files)} files downloaded")
        for file in files:
            print(f"  - {os.path.basename(file)}")