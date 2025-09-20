#!/usr/bin/env python3
"""
Mock/Demo version of the SEC EDGAR scraper for testing in environments without internet access.
This creates sample files to demonstrate the scraper functionality.
"""

import os
import json
from datetime import datetime
from sec_edgar_scraper import SECEdgarScraper


def create_mock_filing(company: str, year: int, output_dir: str) -> str:
    """Create a mock 10-K filing for demonstration purposes."""
    
    # Mock accession number format
    accession_number = f"000{year}012345-{year:02d}-00001"
    
    # Mock filing content (simplified HTML structure)
    mock_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>FORM 10-K - {company} - Annual Report</title>
</head>
<body>
    <div class="info">
        <h1>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</h1>
        <h2>FORM 10-K</h2>
        <h3>ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934</h3>
    </div>
    
    <div class="company-info">
        <h2>Company Information</h2>
        <p><strong>Company:</strong> {company}</p>
        <p><strong>Filing Year:</strong> {year}</p>
        <p><strong>Filing Date:</strong> {year}-03-15</p>
        <p><strong>Accession Number:</strong> {accession_number}</p>
    </div>
    
    <div class="business-section">
        <h2>PART I</h2>
        <h3>Item 1. Business</h3>
        <p>This is a mock 10-K filing for demonstration purposes. In a real filing, this would contain detailed business information, financial statements, and regulatory disclosures.</p>
        
        <h3>Item 1A. Risk Factors</h3>
        <p>Mock risk factors would be listed here...</p>
        
        <h3>Item 2. Properties</h3>
        <p>Property information would be detailed here...</p>
    </div>
    
    <div class="financial-section">
        <h2>PART II</h2>
        <h3>Item 8. Financial Statements and Supplementary Data</h3>
        <p>Financial statements and notes would appear here in a real filing...</p>
    </div>
    
    <div class="footer">
        <p><em>This is a mock filing created for demonstration purposes only.</em></p>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>"""
    
    # Create filename
    filename = f"{company}_10K_{year}_{accession_number}.htm"
    filepath = os.path.join(output_dir, filename)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write mock filing
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(mock_content)
    
    return filepath


def demo_scraper():
    """Demonstrate the scraper functionality with mock data."""
    print("SEC EDGAR Scraper - Demo Mode")
    print("=" * 40)
    print("Creating mock 10-K filings for demonstration...")
    print()
    
    companies = ['GOOGL', 'MSFT', 'NVDA']
    years = [2022, 2023, 2024]
    output_dir = "demo_filings"
    
    results = {}
    total_files = 0
    
    for company in companies:
        print(f"Creating mock filings for {company}...")
        company_files = []
        
        for year in years:
            filepath = create_mock_filing(company, year, output_dir)
            company_files.append(filepath)
            print(f"  ✅ Created: {os.path.basename(filepath)}")
        
        results[company] = company_files
        total_files += len(company_files)
    
    print(f"\n{'='*50}")
    print("DEMO COMPLETE")
    print(f"{'='*50}")
    print(f"Total mock files created: {total_files}")
    print(f"Files saved in: {output_dir}/")
    
    for company, files in results.items():
        print(f"\n{company}: {len(files)} files")
        for file_path in files:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"  - {os.path.basename(file_path)} ({file_size:,} bytes)")
    
    print(f"\n📁 Output directory: {os.path.abspath(output_dir)}")
    
    return results


if __name__ == "__main__":
    demo_scraper()