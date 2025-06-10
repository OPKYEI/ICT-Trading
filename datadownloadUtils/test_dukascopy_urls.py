#!/usr/bin/env python3
"""
Test script to verify which Dukascopy URL format is currently working
"""

import requests
import time
from datetime import datetime, date

def test_url(url, description):
    """Test a single URL and report the results"""
    print(f"\n--- Testing {description} ---")
    print(f"URL: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/octet-stream,*/*',
        'Accept-Language': 'en-US,en;q=0.9'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'Not specified')}")
        print(f"Content-Length: {response.headers.get('content-length', 'Not specified')}")
        
        if response.status_code == 200:
            content_size = len(response.content)
            print(f"Actual Content Size: {content_size} bytes")
            if content_size > 0:
                print("✅ SUCCESS: Data received!")
                return True
            else:
                print("⚠️  Empty response (might be normal for weekends/holidays)")
                return False
        else:
            print(f"❌ Failed with status {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    print("=== Dukascopy URL Format Testing ===")
    print(f"Current date: {datetime.now()}")
    
    # Test with recent data (a few days ago to ensure market was open)
    test_date = date(2025, 5, 30)  # A Friday in May 2025 (should have data)
    symbol = "EURUSD"
    hour = 10  # 10 AM UTC, market should be active
    
    # Format 1: Original format from your code
    url1 = f"https://www.dukascopy.com/datafeed/{symbol}/{test_date.year}/{test_date.month-1:02d}/{test_date.day:02d}/{hour:02d}h_ticks.bi5"
    
    # Format 2: Alternative format seen in some examples
    url2 = f"https://datafeed.dukascopy.com/datafeed/{symbol}/{test_date.year}/{test_date.month-1:02d}/{test_date.day:02d}/{hour:02d}h_ticks.bi5"
    
    # Format 3: Different subdomain variation
    url3 = f"https://www.dukascopy.com/datafeed/{symbol}/{test_date.year}/{test_date.month:02d}/{test_date.day:02d}/{hour:02d}h_ticks.bi5"
    
    # Test all formats
    success1 = test_url(url1, "Original format (month-1)")
    time.sleep(2)  # Be respectful with delays
    
    success2 = test_url(url2, "Datafeed subdomain")
    time.sleep(2)
    
    success3 = test_url(url3, "Regular month indexing")
    time.sleep(2)
    
    # Try with an even more recent date
    recent_date = date(2025, 6, 6)  # Very recent Friday
    url4 = f"https://www.dukascopy.com/datafeed/{symbol}/{recent_date.year}/{recent_date.month-1:02d}/{recent_date.day:02d}/{hour:02d}h_ticks.bi5"
    success4 = test_url(url4, f"Recent date ({recent_date})")
    
    print("\n=== Summary ===")
    if success1:
        print("✅ Original format works")
    if success2:
        print("✅ Datafeed subdomain works")
    if success3:
        print("✅ Regular month indexing works")
    if success4:
        print("✅ Recent date works")
        
    if not any([success1, success2, success3, success4]):
        print("❌ None of the URL formats worked!")
        print("\nPossible reasons:")
        print("1. Dukascopy may have changed their API")
        print("2. They may be blocking automated requests")
        print("3. The dates tested might not have data (weekends/holidays)")
        print("4. Authentication might now be required")
        print("\nRecommended alternatives:")
        print("- Use the official Historical Data Export widget")
        print("- Try the dukascopy-node package (Node.js)")
        print("- Use a commercial data provider")
    
    # Test the manual interface
    print("\n--- Testing Manual Interface ---")
    manual_url = "https://www.dukascopy.com/trading-tools/widgets/quotes/historical_data_feed"
    try:
        response = requests.get(manual_url, timeout=30)
        if response.status_code == 200:
            print("✅ Manual interface is accessible")
        else:
            print(f"❌ Manual interface returned {response.status_code}")
    except Exception as e:
        print(f"❌ Error accessing manual interface: {e}")

if __name__ == "__main__":
    main()
