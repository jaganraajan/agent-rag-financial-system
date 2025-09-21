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
    
    # Mock financial data that can be extracted by the synthesis engine
    financial_data = {
        'MSFT': {
            2022: {'operating_margin': 41.5, 'revenue': 198.3, 'growth': 18.0, 'cloud_revenue': 91.2},
            2023: {'operating_margin': 42.1, 'revenue': 211.9, 'growth': 7.2, 'cloud_revenue': 111.6},
            2024: {'operating_margin': 43.0, 'revenue': 245.0, 'growth': 15.6, 'cloud_revenue': 135.0}
        },
        'GOOGL': {
            2022: {'operating_margin': 23.8, 'revenue': 282.8, 'growth': 10.6, 'cloud_revenue': 26.3},
            2023: {'operating_margin': 25.2, 'revenue': 307.4, 'growth': 8.7, 'cloud_revenue': 33.1},
            2024: {'operating_margin': 26.1, 'revenue': 334.7, 'growth': 8.9, 'cloud_revenue': 38.5}
        },
        'NVDA': {
            2022: {'operating_margin': 15.3, 'revenue': 27.0, 'growth': 0.8, 'datacenter_revenue': 15.0},
            2023: {'operating_margin': 32.1, 'revenue': 60.9, 'growth': 126.0, 'datacenter_revenue': 47.5},
            2024: {'operating_margin': 35.5, 'revenue': 79.8, 'growth': 31.0, 'datacenter_revenue': 64.2}
        }
    }
    
    data = financial_data.get(company, {}).get(year, {})
    operating_margin = data.get('operating_margin', 'N/A')
    revenue = data.get('revenue', 'N/A')
    growth = data.get('growth', 'N/A')
    cloud_revenue = data.get('cloud_revenue', 'N/A')
    datacenter_revenue = data.get('datacenter_revenue', 'N/A')
    
    # Mock filing content with realistic financial data
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
        <p>This section contains business overview and operations for {company}.</p>
        
        <h3>Item 1A. Risk Factors</h3>
        <p>Key business risks and market factors affecting operations.</p>
        
        <h3>Item 2. Properties</h3>
        <p>Information about corporate facilities and properties.</p>
    </div>
    
    <div class="financial-section">
        <h2>PART II</h2>
        <h3>Item 8. Financial Statements and Supplementary Data</h3>
        <h4>Consolidated Statements of Income</h4>
        <p>For the fiscal year ended June 30, {year}:</p>
        <p><strong>Total Revenue:</strong> ${revenue} billion</p>
        <p><strong>Revenue Growth:</strong> {growth}% year-over-year</p>
        <p><strong>Operating Margin:</strong> {operating_margin}%</p>
        
        {f'<p><strong>Cloud Services Revenue:</strong> ${cloud_revenue} billion</p>' if cloud_revenue != 'N/A' else ''}
        {f'<p><strong>Data Center Revenue:</strong> ${datacenter_revenue} billion</p>' if datacenter_revenue != 'N/A' else ''}
        
        <h4>Management Discussion and Analysis</h4>
        <p>During fiscal {year}, the company achieved total revenue of ${revenue} billion with an operating margin of {operating_margin}%.</p>
        <p>Year-over-year revenue growth was {growth}%, reflecting strong business performance.</p>
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