#!/usr/bin/env python3
"""
Main script to run the SEC EDGAR 10-K filing scraper.

This script downloads 10-K filings for Google (GOOGL), Microsoft (MSFT), 
and NVIDIA (NVDA) for the years 2022, 2023, and 2024.
"""

import os
import sys
import argparse
from datetime import datetime
from sec_edgar_scraper import SECEdgarScraper


def main():
    parser = argparse.ArgumentParser(description='Download 10-K filings from SEC EDGAR')
    parser.add_argument('--companies', nargs='+', default=['GOOGL', 'MSFT', 'NVDA'],
                        choices=['GOOGL', 'MSFT', 'NVDA'],
                        help='Company symbols to download (default: all)')
    parser.add_argument('--years', nargs='+', type=int, default=[2022, 2023, 2024],
                        help='Years to download filings for (default: 2022 2023 2024)')
    parser.add_argument('--output-dir', default='filings',
                        help='Output directory for downloaded filings (default: filings)')
    parser.add_argument('--user-agent', default='Financial Analysis Tool 1.0 (jaganraajan@gmail.com)',
                        help='User agent for SEC requests')
    
    args = parser.parse_args()
    
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


if __name__ == "__main__":
    main()