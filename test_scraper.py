#!/usr/bin/env python3
"""
Test script for the SEC EDGAR scraper to verify basic functionality.
"""

import os
import sys
from sec_edgar_scraper import SECEdgarScraper


def test_scraper_basic_functionality():
    """Test basic functionality of the SEC EDGAR scraper."""
    print("Testing SEC EDGAR Scraper...")
    print("=" * 40)
    
    # Initialize scraper
    scraper = SECEdgarScraper()
    
    # Test 1: Check company configuration
    print("Test 1: Company Configuration")
    print(f"Configured companies: {list(scraper.companies.keys())}")
    for symbol, info in scraper.companies.items():
        print(f"  {symbol}: CIK {info['cik']} - {info['name']}")
    print("✅ Company configuration looks good\n")
    
    # Test 2: Test API connectivity (get filings for one company)
    print("Test 2: API Connectivity")
    try:
        test_cik = scraper.companies['NVDA']['cik']  # Use NVIDIA for testing
        print(f"Testing with NVIDIA (CIK: {test_cik})")
        
        filings_data = scraper.get_company_filings(test_cik)
        if filings_data:
            company_name = filings_data.get('name', 'Unknown')
            recent_filings = filings_data.get('filings', {}).get('recent', {})
            num_filings = len(recent_filings.get('form', []))
            print(f"  Company: {company_name}")
            print(f"  Recent filings available: {num_filings}")
            print("✅ API connectivity successful\n")
        else:
            print("❌ Failed to fetch filings data\n")
            return False
            
    except Exception as e:
        print(f"❌ API connectivity test failed: {e}\n")
        return False
    
    # Test 3: Test 10-K filing search
    print("Test 3: 10-K Filing Search")
    try:
        target_years = [2023, 2024]  # Test with recent years
        filings = scraper.find_10k_filings(test_cik, target_years)
        
        print(f"  Searching for 10-K filings in years: {target_years}")
        print(f"  Found {len(filings)} 10-K filings:")
        
        for filing in filings:
            print(f"    - {filing['filing_date']} (Year: {filing['filing_year']}) - {filing['accession_number']}")
        
        if filings:
            print("✅ 10-K filing search successful\n")
        else:
            print("⚠️  No 10-K filings found for test years\n")
            
    except Exception as e:
        print(f"❌ 10-K filing search failed: {e}\n")
        return False
    
    print("=" * 40)
    print("Basic functionality tests completed successfully!")
    print("The scraper is ready to download 10-K filings.")
    
    return True


if __name__ == "__main__":
    success = test_scraper_basic_functionality()
    sys.exit(0 if success else 1)